use base::*;
use filters::*;
use output;
use ui;

pub struct CrossoverFilter {
    left_high_filter: BiquadFilter,
    left_low_filter: BiquadFilter,
    right_high_filter: BiquadFilter,
    right_low_filter: BiquadFilter,
}

impl CrossoverFilter {
    pub fn new(sample_rate: usize, frequency: f32) -> CrossoverFilter {
        CrossoverFilter {
            left_high_filter: BiquadFilter::new(BiquadParams::high_pass_filter(
                sample_rate as FCoef,
                frequency as FCoef,
                1.0,
            )),
            left_low_filter: BiquadFilter::new(BiquadParams::low_pass_filter(
                sample_rate as FCoef,
                frequency as FCoef,
                1.0,
            )),
            right_high_filter: BiquadFilter::new(BiquadParams::high_pass_filter(
                sample_rate as FCoef,
                frequency as FCoef,
                1.0,
            )),
            right_low_filter: BiquadFilter::new(BiquadParams::low_pass_filter(
                sample_rate as FCoef,
                frequency as FCoef,
                1.0,
            )),
        }
    }

    pub fn apply(&mut self, mut frame: Frame) -> Frame {
        assert!(frame.channel_layout == ChannelLayout::Stereo);

        let mut left_bass = frame.data[0].clone();
        self.left_low_filter.apply_multi(&mut left_bass[..]);

        let mut right_bass = frame.data[1].clone();
        self.right_low_filter.apply_multi(&mut right_bass[..]);

        self.left_high_filter.apply_multi(&mut frame.data[0][..]);
        self.right_high_filter.apply_multi(&mut frame.data[1][..]);

        // Mix two sub channels.
        for i in 0..left_bass.len() {
            left_bass[i] += right_bass[i];
        }

        // Add the sub channel.
        frame.data.push(left_bass);
        frame.channel_layout = ChannelLayout::StereoSub;

        frame
    }
}

pub struct SubwooferCrossoverOutput {
    output: Box<output::Output>,
    crossover: Option<CrossoverFilter>,
    ui_msg_receiver: ui::UiMessageReceiver,
}

impl SubwooferCrossoverOutput {
    pub fn new(output: Box<output::Output>, shared_state: &ui::SharedState) -> Box<output::Output> {
        Box::new(SubwooferCrossoverOutput {
            output: output,
            crossover: None,
            ui_msg_receiver: shared_state.lock().add_observer(),
        })
    }
}

impl output::Output for SubwooferCrossoverOutput {
    fn write(&mut self, frame: Frame) -> Result<()> {
        for msg in self.ui_msg_receiver.try_iter() {
            match msg {
                ui::UiMessage::SetSubwooferConfig { config } => {
                    self.crossover = if config.enabled {
                        Some(CrossoverFilter::new(
                            frame.sample_rate,
                            config.crossover_frequency,
                        ))
                    } else {
                        None
                    }
                }
                _ => (),
            }
        }

        let frame = match self.crossover.as_mut() {
            Some(c) => c.apply(frame),
            None => frame,
        };

        self.output.write(frame)
    }

    fn deactivate(&mut self) {
        self.output.deactivate();
    }

    fn sample_rate(&self) -> usize {
        self.output.sample_rate()
    }

    fn period_size(&self) -> usize {
        self.output.sample_rate()
    }

    fn measured_sample_rate(&self) -> f64 {
        self.output.measured_sample_rate()
    }
}
