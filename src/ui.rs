use base::Gain;
use serde;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};

pub const VOLUME_MIN: f32 = -90.0;
pub const VOLUME_MAX: f32 = 0.0;

pub type DeviceId = usize;

impl serde::ser::Serialize for Gain {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_f32(self.db)
    }
}

#[derive(Serialize)]
pub struct InputState {
    pub name: String,
    pub gain: Gain,
}

#[derive(Serialize, Copy, Clone)]
pub struct SubwooferConfig {
    pub crossover_frequency: f32,
}

#[derive(Serialize)]
pub struct OutputState {
    pub name: String,
    pub gain: Gain,
    pub subwoofer: Option<SubwooferConfig>,
    pub drc_supported: bool,
}

#[derive(Serialize, Copy, Clone)]
pub struct LoudnessConfig {
    pub enabled: bool,
    pub auto: bool,
    pub level: f32,
}

impl LoudnessConfig {
    pub fn default() -> LoudnessConfig {
        LoudnessConfig {
            enabled: true,
            auto: true,
            level: 0.5,
        }
    }

    pub fn get_level(&self, volume: Gain) -> Gain {
        match (self.enabled, self.auto) {
            (false, _) => Gain { db: 0.0 },
            (true, false) => Gain {
                db: 20.0 * self.level,
            },

            // No loudness compensation when volume = -5dB.
            (true, true) => Gain {
                db: (-5.0 - volume.db) * self.level,
            },
        }
    }
}

#[derive(Serialize, Copy, Clone)]
pub struct CrossfeedConfig {
    pub enabled: bool,
    pub level: f32,
    pub delay_ms: f32,
}

impl CrossfeedConfig {
    pub fn default() -> CrossfeedConfig {
        CrossfeedConfig {
            enabled: false,
            level: 0.3,
            delay_ms: 0.2,
        }
    }

    pub fn get_level(&self) -> f32 {
        if self.enabled {
            self.level
        } else {
            0.0
        }
    }
}

#[derive(Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum MuxMode {
    Exclusive,
    Mixer,
}

#[derive(Eq, PartialEq, Clone, Copy, Serialize)]
pub enum StreamState {
    Active,
    Inactive,
    Standby,
}

impl StreamState {
    pub fn as_str(&self) -> &'static str {
        match *self {
            StreamState::Active => "active",
            StreamState::Inactive => "inactive",
            StreamState::Standby => "standby",
        }
    }
}

#[derive(Serialize)]
pub struct State {
    pub inputs: Vec<InputState>,
    pub outputs: Vec<OutputState>,
    pub output: usize,
    pub mux_mode: MuxMode,
    pub loudness: LoudnessConfig,
    pub enable_drc: Option<bool>,
    pub enable_subwoofer: Option<bool>,
    pub crossfeed: Option<CrossfeedConfig>,
    pub stream_state: StreamState,
}

#[derive(Copy, Clone)]
pub enum StateChange {
    SelectOutput { output: usize },
    SetMasterVolume { volume: Gain, loudness: Gain },
    SetInputGain { device: DeviceId, gain: Gain },
    SetMuxMode { mux_mode: MuxMode },
    SetEnableDrc { enable: bool },
    SetEnableSubwoofer { enable: bool },
    SetCrossfeed { config: CrossfeedConfig },
    SetStreamState { stream_state: StreamState },
}

pub type StateObserver = mpsc::Receiver<StateChange>;

pub struct StateController {
    state: State,
    observers: Vec<mpsc::Sender<StateChange>>,
}

impl StateController {
    pub fn new() -> StateController {
        StateController {
            state: State {
                inputs: Vec::new(),
                outputs: Vec::new(),
                output: 0,
                mux_mode: MuxMode::Exclusive,
                loudness: LoudnessConfig::default(),
                enable_drc: None,
                enable_subwoofer: None,
                crossfeed: None,
                stream_state: StreamState::Active,
            },
            observers: Vec::new(),
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn add_input(&mut self, state: InputState) {
        self.state.inputs.push(state);
    }

    pub fn add_output(&mut self, state: OutputState) {
        if self.state.enable_drc.is_none() && state.drc_supported {
            self.state.enable_drc = Some(true);
        }
        if self.state.enable_subwoofer.is_none() && state.subwoofer.is_some() {
            self.state.enable_subwoofer = Some(true);
        }
        self.state.outputs.push(state);
    }

    pub fn current_output(&self) -> &OutputState {
        &self.state.outputs[self.state.output]
    }

    pub fn mut_current_output(&mut self) -> &mut OutputState {
        &mut self.state.outputs[self.state.output]
    }

    pub fn volume(&self) -> Gain {
        self.current_output().gain
    }

    fn get_volume_message(&self) -> StateChange {
        StateChange::SetMasterVolume {
            volume: self.volume(),
            loudness: self.state.loudness.get_level(self.volume()),
        }
    }

    pub fn add_observer(&mut self) -> StateObserver {
        let (sender, receiver) = mpsc::channel();
        sender.send(self.get_volume_message()).unwrap();
        self.observers.push(sender);
        receiver
    }

    fn broadcast(&mut self, msg: StateChange) {
        for o in self.observers.iter() {
            o.send(msg).unwrap();
        }
    }

    fn on_volume_updated(&mut self) {
        let message = self.get_volume_message();
        self.broadcast(message);
    }

    pub fn set_volume(&mut self, volume: Gain) {
        self.mut_current_output().gain.db = limit(VOLUME_MIN, VOLUME_MAX, volume.db);
        self.on_volume_updated();
    }

    pub fn move_volume(&mut self, volume_change: Gain) {
        let new_volume = Gain {
            db: self.volume().db + volume_change.db,
        };
        self.set_volume(new_volume);
    }

    pub fn select_output(&mut self, output: usize) {
        self.state.output = output;
        let message = StateChange::SelectOutput {
            output: self.state.output,
        };
        self.broadcast(message);
        self.on_volume_updated();
    }

    pub fn toggle_output(&mut self) {
        let next_output = (self.state.output + 1) % self.state.outputs.len();
        self.select_output(next_output);
    }

    pub fn set_mux_mode(&mut self, mux_mode: MuxMode) {
        self.state.mux_mode = mux_mode;
        self.broadcast(StateChange::SetMuxMode { mux_mode: mux_mode });
    }

    pub fn set_enable_drc(&mut self, enable: bool) {
        if self.state.enable_drc.is_some() {
            self.state.enable_drc = Some(enable);
            self.broadcast(StateChange::SetEnableDrc { enable });
        }
    }

    pub fn set_loudness(&mut self, loudness: LoudnessConfig) {
        self.state.loudness = loudness;
        self.on_volume_updated();
    }

    pub fn set_input_gain(&mut self, input_id: usize, gain: Gain) {
        self.state.inputs[input_id].gain = gain;
        self.broadcast(StateChange::SetInputGain {
            device: input_id,
            gain: gain,
        });
    }

    pub fn set_enable_subwoofer(&mut self, enable: bool) {
        if self.state.enable_subwoofer.is_some() {
            self.state.enable_subwoofer = Some(enable);
            self.broadcast(StateChange::SetEnableSubwoofer { enable });
        }
    }

    pub fn set_crossfeed(&mut self, crossfeed: CrossfeedConfig) {
        self.state.crossfeed = Some(crossfeed);
        self.broadcast(StateChange::SetCrossfeed { config: crossfeed });
    }

    pub fn set_stream_state(&mut self, stream_state: StreamState) {
        self.state.stream_state = stream_state;
        self.broadcast(StateChange::SetStreamState { stream_state });
    }
}

#[derive(Clone)]
pub struct SharedState {
    state: Arc<Mutex<StateController>>,
}

fn limit(min: f32, max: f32, v: f32) -> f32 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

impl SharedState {
    pub fn new() -> SharedState {
        SharedState {
            state: Arc::new(Mutex::new(StateController::new())),
        }
    }

    pub fn lock(&self) -> MutexGuard<StateController> {
        self.state.lock().unwrap()
    }
}
