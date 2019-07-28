use crate::base::*;
use crate::resampler::StreamResampler;
use crate::time::TimeDelta;

pub trait Input: Send {
    fn read(&mut self) -> Result<Option<Frame>>;
    fn min_delay(&self) -> TimeDelta;
}

pub struct InputResampler {
    input: Box<dyn Input>,
    resampler: StreamResampler,
    pos_tracker: StreamPositionTracker,
    input_sample_rate: f64,
}

impl InputResampler {
    pub fn new(input: Box<dyn Input>, output_rate: f64, window_size: usize) -> InputResampler {
        InputResampler {
            input: input,
            resampler: StreamResampler::new(output_rate, window_size),
            pos_tracker: StreamPositionTracker::new(output_rate),
            input_sample_rate: 0.0,
        }
    }

    fn update_frame(&mut self, mut frame: Frame) -> Frame {
        if self.input_sample_rate != frame.sample_rate
            || (self.pos_tracker.pos() - frame.timestamp).abs() > TimeDelta::milliseconds(20)
        {
            self.input_sample_rate = frame.sample_rate;
            self.pos_tracker.reset(frame.timestamp, frame.sample_rate);
        }
        let estimated_end = frame.end_timestamp();

        frame.timestamp = self.pos_tracker.pos();
        self.pos_tracker.add_samples(frame.len(), estimated_end);

        frame
    }
}

impl Input for InputResampler {
    fn read(&mut self) -> Result<Option<Frame>> {
        Ok(match self.input.read()? {
            Some(f) => {
                let updated = self.update_frame(f);
                self.resampler.resample(updated)
            }
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
