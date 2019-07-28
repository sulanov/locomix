use crate::base::*;
use crate::filters;
use crate::time::{Time, TimeDelta};
use std;
use std::cmp;
use std::io::Read;
use std::io::Write;
use std::mem;
use std::process::{Child, Command, Stdio};
use tempfile::NamedTempFile;

pub struct BruteFirChannel {
    child: Child,
}

impl BruteFirChannel {
    pub fn new(
        file: &str,
        sample_rate: usize,
        part_size: usize,
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
            sample_rate, part_size, parts, file
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
        let mut buf = vec![0.0; part_size];
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
    channels: PerChannel<BruteFirChannel>,
    part_size: usize,
    sample_rate: f64,
    parts: usize,
    samples_buffered: usize,
    buf_frame: Frame,
    failed: bool,
}

impl BruteFir {
    pub fn new(
        mut files: PerChannel<String>,
        sample_rate: usize,
        period_duration: TimeDelta,
        filter_length: usize,
    ) -> Result<BruteFir> {
        // Brute fir requries that partition length and filter length are power of 2.
        // Calculate the largest partition length that's shorter than period_duration and shortest
        // filter length not shorter than filter_length.
        let period_size = (period_duration * (sample_rate as i64)) / TimeDelta::seconds(1);
        let part_size = 1 << (32 - (period_size as u32).leading_zeros() - 1);
        let filter_length = 1 << (32 - (filter_length as u32 - 1).leading_zeros());
        let parts = filter_length / part_size;
        println!(
            "INFO: Initializing brutefir with part_size = {}, total_length = {}",
            part_size, filter_length
        );

        let mut channels = PerChannel::new();
        for (c, f) in files.iter() {
            channels.set(c, BruteFirChannel::new(f, sample_rate, part_size, parts)?);
        }

        Ok(BruteFir {
            channels: channels,
            part_size: part_size,
            sample_rate: sample_rate as f64,
            parts: parts,
            samples_buffered: 0,
            buf_frame: Frame::new(sample_rate as f64, Time::zero(), part_size),
            failed: false,
        })
    }

    fn process_frame(
        &mut self,
        input: &mut Frame,
        input_pos: usize,
        output: &mut Frame,
        output_pos: usize,
    ) -> Result<()> {
        for (c, f) in self.channels.iter() {
            let pcm = input.ensure_channel(c);
            f.write(&pcm[input_pos..(input_pos + self.part_size)])?
        }

        for (c, f) in self.channels.iter() {
            let pcm = output.ensure_channel(c);
            f.read(&mut pcm[output_pos..(output_pos + self.part_size)])?;
        }

        // Copy unfiltered channels.
        for (c, inp_pcm) in input.iter_channels() {
            if !self.channels.have_channel(c) {
                output.ensure_channel(c)[output_pos..(output_pos + self.part_size)]
                    .copy_from_slice(&mut inp_pcm[input_pos..(input_pos + self.part_size)]);
            }
        }

        Ok(())
    }

    fn apply_internal(&mut self, mut input: &mut Frame) -> Result<Frame> {
        let total_samples = input.len() + self.samples_buffered;
        let result_samples = (total_samples / self.part_size) * self.part_size;
        let mut result = Frame::new(
            self.sample_rate,
            input.timestamp - samples_to_timedelta(self.sample_rate, self.samples_buffered as i64),
            result_samples,
        );

        result.gain = input.gain;

        let mut input_pos = 0;
        let mut output_pos = 0;

        // Pass input data through buf_frame if we have any leftovers from the previous frames
        // or if we don't have enough data for the next frame
        if self.samples_buffered > 0 || input.len() < self.part_size {
            let append_samples = cmp::min(self.part_size - self.samples_buffered, input.len());
            for (c, inp_pcm) in input.iter_channels() {
                self.buf_frame.ensure_channel(c)
                    [self.samples_buffered..(self.samples_buffered + append_samples)]
                    .copy_from_slice(&mut inp_pcm[0..append_samples]);
            }

            input_pos += append_samples;
            self.samples_buffered += append_samples;

            if self.samples_buffered == self.part_size {
                let mut buf_frame = Frame::new(self.sample_rate, Time::zero(), self.part_size);
                mem::swap(&mut self.buf_frame, &mut buf_frame);
                self.process_frame(&mut buf_frame, 0, &mut result, output_pos)?;
                output_pos += self.part_size;
                self.samples_buffered = 0;
            }
        }

        // Process whole parts.
        while input_pos + self.part_size <= input.len() {
            assert!(self.samples_buffered == 0);
            self.process_frame(&mut input, input_pos, &mut result, output_pos)?;
            input_pos += self.part_size;
            output_pos += self.part_size;
        }

        // Move rest of the frame to buf_frame.
        if input_pos < input.len() {
            assert!(self.samples_buffered == 0);
            let append_samples = input.len() - input_pos;
            for (c, inp_pcm) in input.iter_channels() {
                self.buf_frame.ensure_channel(c)[0..append_samples]
                    .copy_from_slice(&mut inp_pcm[input_pos..(input_pos + append_samples)]);
            }
            self.samples_buffered = append_samples
        }

        assert!(output_pos == result.len());

        Ok(result)
    }

    fn reset_internal(&mut self) -> Result<()> {
        for _ in 0..self.parts {
            let mut buf = vec![0.0; self.part_size];
            for (_c, f) in self.channels.iter() {
                f.write(&(buf)[..])?;
            }
            for (_c, f) in self.channels.iter() {
                f.read(&mut (buf)[..])?;
            }
        }

        Ok(())
    }
}

impl filters::StreamFilter for BruteFir {
    fn apply(&mut self, mut frame: Frame) -> Frame {
        if self.failed {
            return frame;
        }
        match self.apply_internal(&mut frame) {
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
