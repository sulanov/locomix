use base::Gain;
use serde;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};

pub const GAIN_MIN: f32 = -90.0;
pub const GAIN_MAX: f32 = 30.0;

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
pub struct SpeakersConfig {
    pub name: String,

    // SPL level at FS calculated based on speaker sensitivity.
    pub full_scale_spl: f32,
}

#[derive(Serialize)]
pub struct OutputState {
    pub name: String,
    pub gain: Gain,
    pub subwoofer: Option<SubwooferConfig>,
    pub drc_supported: bool,

    pub speakers: Vec<SpeakersConfig>,
    pub current_speakers: Option<usize>,
}

#[derive(Serialize, Copy, Clone)]
pub struct LoudnessConfig {
    pub enabled: bool,
    pub auto: bool,
    pub base_level_spl: f32,

    // [0, 1.0]
    pub level: f32,
}

impl LoudnessConfig {
    pub fn default() -> LoudnessConfig {
        LoudnessConfig {
            enabled: true,
            auto: true,
            base_level_spl: 90.0,
            level: 0.5,
        }
    }

    pub fn get_level(&self, volume_spl: f32) -> Gain {
        match (self.enabled, self.auto) {
            (false, _) => Gain { db: 0.0 },
            (true, false) => Gain {
                db: 20.0 * self.level,
            },
            (true, true) => Gain {
                db: (self.base_level_spl - volume_spl) * self.level,
            },
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
    pub enable_crossfeed: Option<bool>,
    pub stream_state: StreamState,
}

#[derive(Copy, Clone)]
pub enum StateChange {
    SelectOutput {
        output: usize,
    },
    SetMasterVolume {
        gain: Gain,
        volume_spl: f32,
        loudness_gain: Gain,
    },
    SetInputGain {
        device: DeviceId,
        gain: Gain,
    },
    SetMuxMode {
        mux_mode: MuxMode,
    },
    SetEnableDrc {
        enable: bool,
    },
    SetEnableSubwoofer {
        enable: bool,
    },
    SetCrossfeed {
        enable: bool,
    },
    SetStreamState {
        stream_state: StreamState,
    },
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
                enable_crossfeed: None,
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

    pub fn current_gain(&self) -> Gain {
        self.current_output().gain
    }

    fn get_full_scale_spl(&self) -> f32 {
        let out = self.current_output();
        match out.current_speakers {
            Some(i) => out.speakers[i].full_scale_spl,

            // Assume 110 full-scale SPL.
            None => 110.0,
        }
    }

    fn get_volume_message(&self) -> StateChange {
        let gain = self.current_gain();
        let volume_spl = self.get_full_scale_spl() + gain.db;
        let loudness_gain = self.state.loudness.get_level(volume_spl);
        StateChange::SetMasterVolume {
            gain,
            volume_spl,
            loudness_gain,
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

    pub fn set_gain(&mut self, gain: Gain) {
        self.mut_current_output().gain.db = limit(GAIN_MIN, GAIN_MAX, gain.db);
        self.on_volume_updated();
    }

    pub fn set_volume(&mut self, volume_spl: f32) {
        let new_gain = Gain {
            db: limit(GAIN_MIN, GAIN_MAX, volume_spl - self.get_full_scale_spl()),
        };
        self.set_gain(new_gain)
    }

    pub fn move_volume(&mut self, volume_change: Gain) {
        let new_gain = Gain {
            db: limit(
                GAIN_MIN,
                GAIN_MAX,
                self.current_gain().db + volume_change.db,
            ),
        };
        self.set_gain(new_gain)
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

    pub fn set_enable_crossfeed(&mut self, enable: bool) {
        self.state.enable_crossfeed = Some(enable);
        self.broadcast(StateChange::SetCrossfeed { enable });
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
