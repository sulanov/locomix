use base::*;
use filters::*;
use output;
use time;
use ui;

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
    sample_rate: usize,
    frequency: f32,
    channels: Vec<ChannelCrossover>,
}

impl CrossoverFilter {
    pub fn new(sample_rate: usize, frequency: f32) -> CrossoverFilter {
        CrossoverFilter {
            sample_rate: sample_rate,
            frequency: frequency,
            channels: Vec::new(),
        }
    }

    pub fn apply(&mut self, mut frame: Frame) -> Frame {
        let mut bass = vec![0.0; frame.len()];
        for c in 0..frame.channels.len() {
            if self.channels.len() <= c {
                self.channels
                    .push(ChannelCrossover::new(self.sample_rate, self.frequency));
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

    fn min_delay(&self) -> time::TimeDelta {
        self.output.min_delay()
    }

    fn measured_sample_rate(&self) -> f64 {
        self.output.measured_sample_rate()
    }
}
