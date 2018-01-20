use base::*;
use filters::*;

struct ChannelCrossover {
    high_filter: BiquadFilter,
    low_filter: BiquadFilter,
}

impl ChannelCrossover {
    fn new(sample_rate: usize, frequency: f32) -> ChannelCrossover {
        ChannelCrossover {
            high_filter: BiquadFilter::new(BiquadParams::high_pass_filter(
                sample_rate as FCoef,
                frequency as FCoef,
                1.0,
            )),
            low_filter: BiquadFilter::new(BiquadParams::low_pass_filter(
                sample_rate as FCoef,
                frequency as FCoef,
                1.0,
            )),
        }
    }
}

pub struct CrossoverFilter {
    frequency: f32,
    channels: Vec<ChannelCrossover>,
}

impl CrossoverFilter {
    pub fn new(frequency: f32) -> CrossoverFilter {
        CrossoverFilter {
            frequency: frequency,
            channels: Vec::new(),
        }
    }

    pub fn apply(&mut self, mut frame: Frame) -> Frame {
        let mut bass = vec![0.0; frame.len()];
        for c in 0..frame.channels.len() {
            if self.channels.len() <= c {
                self.channels
                    .push(ChannelCrossover::new(frame.sample_rate, self.frequency));
            }

            let crossover = &mut self.channels[c];
            let pcm = &mut frame.channels[c].pcm;

            for i in 0..bass.len() {
                bass[i] += crossover.low_filter.apply_one(pcm[i]);
            }

            crossover.high_filter.apply_multi(&mut pcm[..]);
        }

        // Add the sub channel.
        frame.channels.push(ChannelData {
            pos: ChannelPos::Sub,
            pcm: bass,
        });

        frame
    }
}
