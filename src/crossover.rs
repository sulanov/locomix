use base::*;
use filters::*;
use std::collections::BTreeMap;

pub struct CrossoverFilter {
    frequency: f32,
    channels: BTreeMap<ChannelPos, CascadingFilter>,
}

impl CrossoverFilter {
    pub fn new(frequency: f32) -> CrossoverFilter {
        CrossoverFilter {
            frequency: frequency,
            channels: BTreeMap::new(),
        }
    }

    pub fn apply(&mut self, mut frame: Frame) -> Frame {
        let mut bass = vec![0.0; frame.len()];
        let mut sub_index: i32 = -1;
        let frequency = self.frequency;
        for c in 0..frame.channels.len() {
            {
                let pcm = &mut frame.channels[c].pcm;
                for i in 0..bass.len() {
                    bass[i] += pcm[i];
                }
            }

            let pos = frame.channels[c].pos;
            if pos == ChannelPos::Sub {
                sub_index = c as i32;
                continue;
            }

            let filter = self.channels.entry(pos).or_insert_with(|| {
                CascadingFilter::new(BiquadParams::high_pass_filter(
                    frame.sample_rate as FCoef,
                    frequency as FCoef,
                    1.0,
                ))
            });

            let pcm = &mut frame.channels[c].pcm;
            filter.apply_multi(&mut pcm[..]);
        }

        let filter = self.channels.entry(ChannelPos::Sub).or_insert_with(|| {
            CascadingFilter::new(BiquadParams::low_pass_filter(
                frame.sample_rate as FCoef,
                frequency as FCoef,
                1.0,
            ))
        });

        filter.apply_multi(&mut bass[..]);

        let sub_channel = ChannelData {
            pos: ChannelPos::Sub,
            pcm: bass,
        };

        if sub_index >= 0 {
            frame.channels[sub_index as usize] = sub_channel
        } else {
            frame.channels.push(sub_channel);
        }

        frame
    }
}
