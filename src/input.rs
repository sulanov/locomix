use base::*;
use resampler::StreamResampler;

pub trait Input: Send {
    fn read(&mut self) -> Result<Option<Frame>>;
}

pub struct InputResampler {
    input: Box<Input>,
    resampler: StreamResampler,
}

impl InputResampler {
    pub fn new(input: Box<Input>, sample_rate: usize, window_size: usize) -> InputResampler {
        InputResampler {
            input: input,
            resampler: StreamResampler::new(sample_rate, window_size),
        }
    }
}

impl Input for InputResampler {
    fn read(&mut self) -> Result<Option<Frame>> {
        Ok(match try!(self.input.read()) {
            Some(f) => self.resampler.resample(f),
            None => None,
        })
    }
}
