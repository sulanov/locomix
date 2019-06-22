use base::*;
use filters::*;

pub struct CrossoverFilter {
    frequency: f32,
    channels: PerChannel<CascadingFilter>,
}

impl CrossoverFilter {
    pub fn new(frequency: f32) -> CrossoverFilter {
        CrossoverFilter {
            frequency: frequency,
            channels: PerChannel::new(),
        }
    }

    pub fn apply(&mut self, mut frame: Frame) -> Frame {
        let mut bass = vec![0.0; frame.len()];
        let frequency = self.frequency;
        let sample_rate = frame.sample_rate;
        for (c, pcm) in frame.iter_channels() {
            if c == ChannelPos::Sub {
                continue;
            }

            for i in 0..bass.len() {
                bass[i] += pcm[i];
            }

            self.channels
                .get_or_insert(c, || {
                    CascadingFilter::new(&BiquadParams::high_pass_filter(
                        sample_rate as BqCoef,
                        frequency as BqCoef,
                        1.0,
                    ))
                })
                .apply_multi(pcm);
        }

        let filter = self.channels.get_or_insert(ChannelPos::Sub, || {
            CascadingFilter::new(&BiquadParams::low_pass_filter(
                frame.sample_rate as BqCoef,
                frequency as BqCoef,
                1.0,
            ))
        });

        filter.apply_multi(&mut bass[..]);

        frame.mix_channel(ChannelPos::Sub, bass, 1.0 / SUBWOOFER_LEVEL);

        frame
    }
}
