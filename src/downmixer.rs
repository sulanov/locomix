use base::*;

pub struct Downmixer {
    active_channels: PerChannel<bool>,
}

impl Downmixer {
    pub fn new(channels: PerChannel<bool>) -> Downmixer {
        Downmixer {
            active_channels: channels,
        }
    }

    pub fn set_channel_active(&mut self, c: ChannelPos, active: bool) {
        self.active_channels.set(c, active);
    }

    fn have_channel(&self, c: ChannelPos) -> bool {
        self.active_channels.get(c) == Some(&true)
    }

    fn downmix_channel(&self, frame: &mut Frame, c: ChannelPos, out: &[(ChannelPos, f32)]) {
        if self.have_channel(c) {
            return;
        }
        let in_pcm = match frame.take_channel(c) {
            None => return,
            Some(in_pcm) => in_pcm,
        };

        for &(c, level) in out {
            let out_pcm = frame.ensure_channel(c);
            for i in 0..out_pcm.len() {
                out_pcm[i] += in_pcm[i] * level;
            }
        }
    }

    pub fn process(&self, mut frame: Frame) -> Frame {
        self.downmix_channel(
            &mut frame,
            ChannelPos::Sub,
            &[
                (ChannelPos::FL, 0.5 * SUBWOOFER_LEVEL),
                (ChannelPos::FL, 0.5 * SUBWOOFER_LEVEL),
            ],
        );
        self.downmix_channel(
            &mut frame,
            ChannelPos::FC,
            &[(ChannelPos::FL, 0.5), (ChannelPos::FL, 0.5)],
        );
        self.downmix_channel(&mut frame, ChannelPos::SL, &[(ChannelPos::FL, 1.0)]);
        self.downmix_channel(&mut frame, ChannelPos::SR, &[(ChannelPos::FR, 1.0)]);
        if self.have_channel(ChannelPos::SL) && self.have_channel(ChannelPos::SR) {
            self.downmix_channel(
                &mut frame,
                ChannelPos::SC,
                &[(ChannelPos::SL, 0.5), (ChannelPos::SL, 0.5)],
            );
        } else {
            self.downmix_channel(
                &mut frame,
                ChannelPos::SC,
                &[(ChannelPos::FL, 0.5), (ChannelPos::FL, 0.5)],
            );
        }
        frame
    }
}
