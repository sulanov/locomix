use crate::base::*;

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
        self.downmix_channel(&mut frame, CHANNEL_LFE, &[(CHANNEL_FC, SUBWOOFER_LEVEL)]);

        self.downmix_channel(
            &mut frame,
            CHANNEL_SC,
            &[(CHANNEL_SL, 0.5), (CHANNEL_SR, 0.5)],
        );
        self.downmix_channel(&mut frame, CHANNEL_SL, &[(CHANNEL_FL, 1.0)]);
        self.downmix_channel(&mut frame, CHANNEL_SR, &[(CHANNEL_FR, 1.0)]);

        if self.have_channel(CHANNEL_FC) {
            self.downmix_channel(&mut frame, CHANNEL_FL, &[(CHANNEL_FC, 1.0)]);
            self.downmix_channel(&mut frame, CHANNEL_FR, &[(CHANNEL_FC, 1.0)]);
        } else {
            self.downmix_channel(
                &mut frame,
                CHANNEL_FC,
                &[(CHANNEL_FL, 0.5), (CHANNEL_FR, 0.5)],
            );
        }

        frame
    }
}
