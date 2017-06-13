use base::*;
use resampler::{ResamplerFactory, FastResampler};

pub trait Input: Send {
    fn read(&mut self) -> Result<Option<Frame>>;
    fn samples_buffered(&mut self) -> Result<usize>;

    // Synchronized streams blocks read so the sample rate matches target value. For example audio
    // device input is synchronized with the device).
    // In non-synchronized streams read() always returns quickly. It may still block for short
    // period of time, e.g. for file IO.
    fn is_synchronized(&self) -> bool;
}

pub struct InputResampler {
    input: Box<Input>,
    current_input_rate: usize,
    sample_rate: usize,
    resampler_factory: ResamplerFactory,
    resamplers: Option<[FastResampler; 2]>,
}

impl InputResampler {
    pub fn new(input: Box<Input>, sample_rate: usize) -> InputResampler {
        InputResampler {
            input: input,
            current_input_rate: 0,
            sample_rate: sample_rate,
            resampler_factory: ResamplerFactory::new(),
            resamplers: None,
        }
    }
}

impl Input for InputResampler {
    fn read(&mut self) -> Result<Option<Frame>> {
        Ok(try!(self.input.read()).map(|frame| {
            if self.current_input_rate != frame.sample_rate {
                self.current_input_rate = frame.sample_rate;
                self.resamplers =
                    Some([self.resampler_factory
                              .create_resampler(frame.sample_rate, self.sample_rate, 100),
                          self.resampler_factory
                              .create_resampler(frame.sample_rate, self.sample_rate, 100)]);
            };
            Frame {
                sample_rate: self.current_input_rate,
                left: self.resamplers.as_mut().unwrap()[0].resample(&frame.left),
                right: self.resamplers.as_mut().unwrap()[1].resample(&frame.right),
            }
        }))
    }

    fn samples_buffered(&mut self) -> Result<usize> {
        let irate = if self.current_input_rate > 0 {
            self.current_input_rate
        } else {
            self.sample_rate
        };
        Ok(try!(self.input.samples_buffered()) * self.sample_rate / irate)
    }

    fn is_synchronized(&self) -> bool {
        self.input.is_synchronized()
    }
}
