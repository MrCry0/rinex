//! RINEX compression module

use crate::{
    epoch::format as epoch_format,
    error::FormattingError,
    hatanaka::{NumDiff, TextDiff},
    observation::{HeaderFields, Record},
    prelude::{Constellation, Observable, RinexType, SV},
    BufWriter,
};

use std::{collections::HashMap, io::Write};

use itertools::Itertools;

pub type Compressor = CompressorExpert<5>;

/// Compression order applied to every numerical field, matching what
/// the historical RNX2CRX tool uses.
const ORDER: usize = 3;

/// Compression state of one vehicle: its per-observable kernels and its
/// LLI/SNR flags kernel. Dropped (and rebuilt from scratch) when the
/// vehicle is absent from an epoch, so it is republished in full the
/// way RNX2CRX does when a vehicle returns.
struct SvState<const M: usize> {
    kernels: HashMap<Observable, NumDiff<M>>,
    flags: TextDiff,
}

impl<const M: usize> Default for SvState<M> {
    fn default() -> Self {
        Self {
            kernels: HashMap::with_capacity(8),
            flags: TextDiff::new(""),
        }
    }
}

pub struct CompressorExpert<const M: usize> {
    /// True (by default) if this a CRINEX3 compressor.
    /// Modify this before getting started!
    pub v3: bool,
    /// True when epoch descriptor should be compressed.
    /// True on first epoch.
    epoch_compression: bool,
    /// Readable Epoch being compressed
    epoch_buf: String,
    /// Observation line (values, then LLI/SNR flags) being compressed
    line_buf: String,
    /// Readable flags being compressed
    flags_buf: String,
    /// Epoch [TextDiff]
    epoch_diff: TextDiff,
    /// Receiver clock offset kernel, None while no offset is being reported
    clock_diff: Option<NumDiff<M>>,
    /// Compression state (observable kernels, LLI/SNR flags), per SV
    /// present in the previous epoch
    sv_states: HashMap<SV, SvState<M>>,
}

impl<const M: usize> Default for CompressorExpert<M> {
    fn default() -> Self {
        Self {
            v3: true,
            epoch_compression: false,
            epoch_diff: TextDiff::new(""),
            epoch_buf: String::with_capacity(128),
            line_buf: String::with_capacity(128),
            flags_buf: String::with_capacity(128),
            clock_diff: None,
            sv_states: HashMap::with_capacity(8),
        }
    }
}

impl<const M: usize> CompressorExpert<M> {
    /// Format [Record] using mutable [CompressorExpert].
    /// Compressed bytes are dumped in mutable [BufWriter].
    /// This permits the RNX2CRX compression ops.
    pub fn format<W: Write>(
        &mut self,
        w: &mut BufWriter<W>,
        record: &Record,
        header: &HeaderFields,
    ) -> Result<(), FormattingError> {
        // RINEX 3 formats the clock offset as F15.12, RINEX 2 as F12.9:
        // the CRINEX integer is the offset with the decimal point removed.
        let clock_scaling = if self.v3 { 1.0E12 } else { 1.0E9 };

        for (k, v) in record.iter() {
            if !k.flag.is_ok() {
                // TODO not 100% correct, verify > 1
                self.epoch_compression = false;
            }

            // form unique SV list
            let svnn = v
                .signals
                .iter()
                .map(|sig| sig.sv)
                .unique()
                .collect::<Vec<_>>();

            if !self.epoch_compression {
                if self.v3 {
                    write!(w, "> ")?;
                } else {
                    write!(w, "&")?;
                }
            } else {
                if self.v3 {
                    write!(w, "  ")?;
                } else {
                    write!(w, " ")?;
                }
            }

            let revision = if self.v3 { 3 } else { 2 };

            // RINEX 3 reserves 6 columns between the satellite count and
            // the SV list on the epoch descriptor line, RINEX 2 does not.
            let sat_list_pad = if self.v3 { "      " } else { "" };

            self.epoch_buf.push_str(&format!(
                "{}  {}{:3}{}",
                epoch_format(k.epoch, RinexType::ObservationData, revision),
                k.flag,
                svnn.len(),
                sat_list_pad,
            ));

            // Append each SV to epoch description
            for sv in svnn.iter() {
                self.epoch_buf.push_str(&format!("{:x}", sv));
            }

            // Epoch compression
            if !self.epoch_compression {
                self.epoch_diff.force_init(&self.epoch_buf);
                writeln!(w, "{}", self.epoch_buf.trim_end())?;
            } else {
                let compressed = self.epoch_diff.compress(&self.epoch_buf);
                writeln!(w, "{}", compressed.trim_end())?;
            }

            // Receiver clock offset: goes through its own order 3 kernel,
            // reset ("m&value") on the first sample and whenever the
            // offset was missing on the previous epoch, like RNX2CRX does.
            match v.clock {
                Some(clock) => {
                    let value = (clock.offset_s * clock_scaling).round() as i64;

                    match &mut self.clock_diff {
                        Some(kernel) => {
                            let compressed = kernel.compress(value)?;
                            writeln!(w, "{}", compressed)?;
                        },
                        None => {
                            writeln!(w, "{}&{}", ORDER, value)?;
                            self.clock_diff = Some(NumDiff::<M>::new(value, ORDER));
                        },
                    }
                },
                None => {
                    // No clock: BLANKed line, kernel rebuilt on next offset
                    writeln!(w)?;
                    self.clock_diff = None;
                },
            }

            // For each SV
            for sv in svnn.iter() {
                // Following header specs
                let sv_observables = header.codes.get(&sv.constellation);

                let sv_observables = match sv_observables {
                    Some(observables) => observables, // correctly identified,
                    None => {
                        // handles SBAS case
                        if sv.constellation.is_sbas() {
                            match header.codes.get(&Constellation::SBAS) {
                                Some(observables) => observables,
                                None => {
                                    // correctly formatted RINEX will never
                                    // end up here
                                    continue;
                                },
                            }
                        } else {
                            // correctly formatted RINEX will never
                            // end up here
                            continue;
                        }
                    },
                };

                // vehicles that were not present in the previous epoch
                // start with fresh kernels, and their flags are
                // republished in full (blanks as '&') on this line
                let state = self.sv_states.entry(*sv).or_default();

                self.line_buf.clear();
                self.flags_buf.clear();

                for observable in sv_observables.iter() {
                    if let Some(signal) = v
                        .signals
                        .iter()
                        .filter(|sig| sig.sv == *sv && &sig.observable == observable)
                        .reduce(|k, _| k)
                    {
                        let quantized = (signal.value * 1000.0).round() as i64;

                        // retrieve or build compression kernel
                        if let Some(kernel) = state.kernels.get_mut(observable) {
                            let compressed = kernel.compress(quantized)?;
                            self.line_buf.push_str(&format!("{} ", compressed));
                        } else {
                            // first encounter: build kernel
                            state
                                .kernels
                                .insert(observable.clone(), NumDiff::<M>::new(quantized, ORDER));

                            self.line_buf.push_str(&format!("{}&{} ", ORDER, quantized));
                        }

                        if let Some(lli) = signal.lli {
                            self.flags_buf.push_str(&format!("{}", lli.bits() as u8));
                        } else {
                            self.flags_buf.push_str(" ");
                        }

                        if let Some(snr) = signal.snr {
                            self.flags_buf.push_str(&format!("{}", snr as u8));
                        } else {
                            self.flags_buf.push_str(" ");
                        }
                    } else {
                        // missing observation: kernel reinitialized on next sample
                        state.kernels.remove(observable);
                        self.line_buf.push(' ');
                        self.flags_buf.push_str("  ");
                    }
                }

                // LLI/SNR flags compression, appended to the line
                let compressed = state.flags.compress(&self.flags_buf);
                self.line_buf.push_str(compressed);

                writeln!(w, "{}", self.line_buf.trim_end())?;
            }

            // vehicles missing from this epoch are reinitialized on their return
            self.sv_states.retain(|sv, _| svnn.contains(sv));

            // prepare for next epoch
            self.epoch_compression = true;
            self.epoch_buf.clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        hatanaka::Decompressor,
        observation::{ClockObservation, EpochFlag, ObsKey, Observations, SignalObservation},
        prelude::{Constellation, Epoch, Observable, SV},
        tests::formatting::Utf8Buffer,
    };

    use std::{collections::BTreeMap, str::FromStr};

    /// Compresses a two-epoch record carrying a non zero, varying receiver
    /// clock offset and decompresses it back, to catch scale mismatches
    /// between the compressor and the decompressor: an offset scaled by
    /// the wrong power of ten still round-trips to itself bit for bit if
    /// both sides share the same (wrong) scale, but does not survive
    /// comparison against the original F15.12 value.
    #[test]
    fn clock_offset_v3_round_trip() {
        let c1c = Observable::from_str("C1C").unwrap();
        let sv = SV::from_str("G01").unwrap();

        let mut codes = HashMap::new();
        codes.insert(Constellation::GPS, vec![c1c.clone()]);

        let header = HeaderFields {
            codes,
            ..Default::default()
        };

        let offsets_s = [-0.123456789012, -0.123456789098];
        let epochs = [
            Epoch::from_str("2021-01-01T00:00:00 GPST").unwrap(),
            Epoch::from_str("2021-01-01T00:00:30 GPST").unwrap(),
        ];

        let mut record: Record = BTreeMap::new();

        for (epoch, offset_s) in epochs.iter().zip(offsets_s.iter()) {
            let key = ObsKey {
                epoch: *epoch,
                flag: EpochFlag::Ok,
            };

            let obs = Observations {
                clock: Some(ClockObservation::default().with_offset_s(*epoch, *offset_s)),
                signals: vec![SignalObservation {
                    sv,
                    observable: c1c.clone(),
                    value: 20_000_000.0,
                    lli: None,
                    snr: None,
                }],
            };

            record.insert(key, obs);
        }

        let mut compressor = CompressorExpert::<5>::default();
        compressor.v3 = true;

        let mut buf = BufWriter::new(Utf8Buffer::new(1024));
        compressor.format(&mut buf, &record, &header).unwrap();

        let compressed = buf.into_inner().unwrap().to_ascii_utf8();

        // the first clock sample resets the kernel: "3&" followed by the
        // offset scaled by 1E12 (F15.12), not 1E9 or 1E3.
        let expected_first = (offsets_s[0] * 1.0E12).round() as i64;
        assert!(
            compressed.contains(&format!("3&{}", expected_first)),
            "unexpected first clock line in:\n{}",
            compressed
        );

        // decompress it back and recover both offsets within 1ps
        let mut gnss_observables = HashMap::new();
        gnss_observables.insert(Constellation::GPS, vec![c1c.clone()]);

        let mut decompressor = Decompressor::new(true, Constellation::GPS, gnss_observables);

        let mut out = [0u8; 4096];
        let mut recovered = Vec::new();

        let out_len = out.len();
        for line in compressed.lines() {
            let size = decompressor
                .decompress(line, line.len(), &mut out, out_len)
                .unwrap_or_else(|e| panic!("decompression failed on \"{}\": {}", line, e));

            let text = std::str::from_utf8(&out[..size]).unwrap();

            for decoded in text.lines() {
                if decoded.starts_with('>') {
                    let value = decoded
                        .rsplit_once(char::is_whitespace)
                        .expect("no clock field in decoded epoch line")
                        .1
                        .parse::<f64>()
                        .expect("clock field is not a valid float");
                    recovered.push(value);
                }
            }
        }

        assert_eq!(recovered.len(), offsets_s.len(), "missing recovered epoch");

        for (recovered, expected) in recovered.iter().zip(offsets_s.iter()) {
            assert!(
                (recovered - expected).abs() < 1.0E-12,
                "recovered clock offset {} does not match model {}",
                recovered,
                expected
            );
        }
    }
}
