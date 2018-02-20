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
    delay: TimeDelta,
}

impl InputResampler {
    pub fn new(input: Box<Input>, sample_rate: usize, window_size: usize) -> InputResampler {
        InputResampler {
            input: input,
            resampler: StreamResampler::new(sample_rate as f64, sample_rate, window_size),
            delay: samples_to_timedelta(sample_rate, window_size as i64),
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

    fn min_delay(&self) -> TimeDelta {
        self.input.min_delay() + self.delay
    }
}
