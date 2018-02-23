extern crate tempfile;

use base::*;
use std::cmp;
use std::io::Read;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std;
use time::TimeDelta;
use filters;

use self::tempfile::NamedTempFile;

pub struct BruteFirChannel {
    child: Child,
}

impl BruteFirChannel {
    pub fn new(
        file: &str,
        sample_rate: usize,
        part_length: usize,
        parts: usize,
    ) -> Result<BruteFirChannel> {
        let mut config = NamedTempFile::new()?;
        config.write_fmt(format_args!(
            "sampling_rate: {};
filter_length: {},{};
show_progress: false;

coeff 0 {{
	filename: \"{}\";
	format: \"FLOAT_LE\";
}};

input 0 {{
  device: \"file\" {{ path: \"/dev/stdin\"; }};
  sample: \"FLOAT_LE\";
  channels: 1;
}};

output 0 {{
  device: \"file\" {{ path: \"/dev/stdout\"; }};
  sample: \"FLOAT_LE\";
  dither: false;
  channels: 1;
}};

filter 0 {{
  from_inputs: 0;
  to_outputs: 0;
  coeff: 0;
}};",
            sample_rate, part_length, parts, file
        ))?;

        let process = Command::new("brutefir")
            .arg(config.path().as_os_str())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let mut result = BruteFirChannel { child: process };

        // Pass an empty block through brutefir to make sure it's finished
        // starting and was able to read the config file. |config| may be
        // dropped after that, which will also delete the temp file.
        let mut buf = vec![0.0; part_length];
        result.write(&buf[..])?;
        result.read(&mut buf[..])?;

        Ok(result)
    }

    fn write(&mut self, data: &[f32]) -> Result<()> {
        let data_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data[..].as_ptr() as *const u8, data.len() * 4) };
        self.child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(&data_bytes[..])?;
        self.child.stdin.as_mut().unwrap().flush()?;
        Ok(())
    }

    fn read(&mut self, data: &mut [f32]) -> Result<()> {
        let data_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data[..].as_ptr() as *mut u8, data.len() * 4) };

        self.child
            .stdout
            .as_mut()
            .unwrap()
            .read_exact(&mut data_bytes[..])?;

        Ok(())
    }
}

pub struct BruteFir {
    files: Vec<String>,
    channels: Vec<BruteFirChannel>,
    part_length: usize,
    sample_rate: usize,
    parts: usize,
    samples_buffered: usize,
    failed: bool,
}

impl BruteFir {
    pub fn new(
        files: Vec<String>,
        sample_rate: usize,
        period_duration: TimeDelta,
        filter_length: usize,
    ) -> Result<BruteFir> {
        // Brute fir requries that partition length and filter length are power of 2.
        // Calculate the largest partition length that's shorter than period_duration and shortest
        // filter length not shorter than filter_length.
        let period_size = (period_duration * (sample_rate as i64)) / TimeDelta::seconds(1);
        let part_length = 1 << (32 - (period_size as u32).leading_zeros() - 1);
        let filter_length = 1 << (32 - (filter_length as u32 - 1).leading_zeros());

        println!(
            "INFO: Initializing brutefir with part_length = {}, total_length = {}",
            part_length, filter_length
        );
        Ok(BruteFir {
            files: files,
            channels: vec![],
            part_length: part_length,
            sample_rate: sample_rate,
            parts: filter_length / part_length,
            samples_buffered: 0,
            failed: false,
        })
    }

    fn apply_internal(&mut self, input: &Frame) -> Result<Frame> {
        if self.channels.is_empty() {
            self.channels = Vec::with_capacity(self.files.len());
            for c in 0..self.files.len() {
                self.channels.push(BruteFirChannel::new(
                    &self.files[c],
                    self.sample_rate,
                    self.part_length,
                    self.parts,
                )?);
            }
        }

        assert!(input.channels.len() == self.channels.len());

        let total_samples = input.len() + self.samples_buffered;
        let result_samples = (total_samples / self.part_length) * self.part_length;
        let mut result = Frame::new_stereo(
            self.sample_rate as f32,
            input.timestamp - samples_to_timedelta(self.sample_rate as f32, self.samples_buffered as i64),
            result_samples,
        );

        let mut input_pos = 0;
        let mut output_pos = 0;
        while input_pos < input.len() {
            let samples_to_write = cmp::min(
                input.len() - input_pos,
                self.part_length - self.samples_buffered,
            );

            for c in 0..self.channels.len() {
                self.channels[c]
                    .write(&input.channels[c].pcm[input_pos..(input_pos + samples_to_write)])?;
            }

            self.samples_buffered += samples_to_write;
            input_pos += samples_to_write;

            assert!(self.samples_buffered <= self.part_length);
            if self.samples_buffered == self.part_length {
                for c in 0..self.channels.len() {
                    self.channels[c].read(
                        &mut result.channels[c].pcm[output_pos..(output_pos + self.part_length)],
                    )?;
                }
                self.samples_buffered -= self.part_length;
                output_pos += self.part_length;
            }
        }
        assert!(output_pos == result.len());

        Ok(result)
    }

    fn reset_internal(&mut self) -> Result<()> {
        for _ in 0..self.parts {
            let mut buf = vec![0.0; self.part_length];
            for c in &mut self.channels {
                c.write(&(buf)[..])?;
            }
            for c in &mut self.channels {
                c.read(&mut (buf)[..])?;
            }
        }

        Ok(())
    }
}

impl filters::StreamFilter for BruteFir {
    fn apply(&mut self, frame: Frame) -> Frame {
        if self.failed {
            return frame;
        }
        match self.apply_internal(&frame) {
            Ok(r) => r,
            Err(e) => {
                println!("ERROR: BruteFir failed: {:?}", e);
                self.failed = true;
                frame
            }
        }
    }

    fn reset(&mut self) {
        if self.failed {
            self.failed = false;
            self.channels.clear();
        } else {
            match self.reset_internal() {
                Ok(_) => (),
                Err(_) => self.channels.clear(),
            }
        }
    }
}
