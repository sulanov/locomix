use base::*;
use resampler::StreamResampler;
use time::TimeDelta;

pub trait Input: Send {
    fn read(&mut self) -> Result<Option<Frame>>;
    fn min_delay(&self) -> TimeDelta;
}

pub struct InputResampler {
    input: Box<Input>,
    resampler: StreamResampler,
}

impl InputResampler {
    pub fn new(input: Box<Input>, output_rate: f64, window_size: usize) -> InputResampler {
        InputResampler {
            input: input,
            resampler: StreamResampler::new(output_rate, window_size),
        }
    }
}

impl Input for InputResampler {
    fn read(&mut self) -> Result<Option<Frame>> {
        Ok(match try!(self.input.read()) {
            Some(f) => self.resampler.resample(f),
            None => {
                self.resampler.reset();
                None
            }
        })
    }

    fn min_delay(&self) -> TimeDelta {
        self.input.min_delay() + self.resampler.delay()
    }
}
