use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};
use rustc_serialize;
use std::sync::mpsc;

pub const VOLUME_MIN: f32 = -50.0;
pub const VOLUME_MAX: f32 = 0.0;

pub type DeviceId = usize;

#[derive(Copy, Clone)]
pub struct Gain {
    pub db: f32,
}

impl rustc_serialize::Encodable for Gain {
    fn encode<S: rustc_serialize::Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_f32(self.db)
    }
}

#[derive(RustcEncodable)]
pub struct InputState {
    pub name: String,
    pub gain: Gain,
}

impl InputState {
    pub fn new(name: &str) -> InputState {
        InputState {
            name: String::from(name),
            gain: Gain { db: 0.0 },
        }
    }
}

#[derive(RustcEncodable)]
pub struct OutputState {
    pub name: String,
    pub gain: Gain,
}

impl OutputState {
    pub fn new(name: &str) -> OutputState {
        OutputState {
            name: String::from(name),
            gain: Gain { db: (VOLUME_MIN + VOLUME_MAX) / 2.0 },
        }
    }
}

#[derive(RustcEncodable, Copy, Clone)]
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
            (true, false) => Gain { db: 20.0 * self.level },

            // No loudness compensation when volume = -5dB.
            (true, true) => Gain { db: (-5.0 - volume.db) * self.level },
        }
    }
}

#[derive(RustcEncodable, Copy, Clone)]
pub struct CrossfeedConfig {
    pub enabled: bool,
    pub level: f32,
    pub delay_ms: f32,
}

impl CrossfeedConfig {
    pub fn default() -> CrossfeedConfig {
        CrossfeedConfig {
            enabled: false,
            level: 0.5,
            delay_ms: 0.5,
        }
    }

    pub fn get_level(&self) -> f32 {
        if self.enabled { self.level } else { 0.0 }
    }
}

#[derive(RustcEncodable, Copy, Clone)]
pub struct FeatureConfig {
    pub enabled: bool,
    pub level: f32,
}

impl FeatureConfig {
    pub fn default() -> FeatureConfig {
        FeatureConfig {
            enabled: false,
            level: 0.5,
        }
    }

    pub fn get_level(&self) -> f32 {
        if self.enabled { self.level } else { 0.0 }
    }
}

#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum MuxMode {
    Exclusive,
    Mixer,
}

#[derive(RustcEncodable)]
pub struct State {
    pub inputs: Vec<InputState>,
    pub outputs: Vec<OutputState>,
    pub output: usize,
    pub mux_mode: MuxMode,
    pub loudness: LoudnessConfig,
    pub enable_drc: bool,
    pub crossfeed: CrossfeedConfig,
    pub voice_boost: FeatureConfig,
}

#[derive(Copy, Clone)]
pub enum UiMessage {
    SelectOutput { output: usize },
    SetMasterVolume { volume: Gain, loudness: Gain },
    SetInputGain { device: DeviceId, gain: Gain },
    SetMuxMode { mux_mode: MuxMode },
    SetEnableDrc { enable: bool },
    SetVoiceBoost { boost: Gain },
    SetCrossfeed { level: f32, delay_ms: f32 },
}

pub type UiMessageReceiver = mpsc::Receiver<UiMessage>;

#[derive(Eq, PartialEq, Clone, Copy)]
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

pub type StreamStateReceiver = mpsc::Receiver<StreamState>;

pub struct StateController {
    state: State,
    observers: Vec<mpsc::Sender<UiMessage>>,
    stream_observers: Vec<mpsc::Sender<StreamState>>,
}

impl StateController {
    pub fn new(inputs: Vec<InputState>, outputs: Vec<OutputState>) -> StateController {
        StateController {
            state: State {
                inputs: inputs,
                outputs: outputs,
                output: 0,
                mux_mode: MuxMode::Exclusive,
                enable_drc: true,
                loudness: LoudnessConfig::default(),
                crossfeed: CrossfeedConfig::default(),
                voice_boost: FeatureConfig::default(),
            },
            stream_observers: Vec::new(),
            observers: Vec::new(),
        }
    }

    pub fn state(&self) -> &State {
        &self.state
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

    fn get_volume_message(&self) -> UiMessage {
        UiMessage::SetMasterVolume {
            volume: self.volume(),
            loudness: self.state.loudness.get_level(self.volume()),
        }
    }

    pub fn add_observer(&mut self) -> UiMessageReceiver {
        let (sender, receiver) = mpsc::channel();
        sender.send(self.get_volume_message()).unwrap();
        self.observers.push(sender);
        receiver
    }

    pub fn add_stream_observer(&mut self) -> StreamStateReceiver {
        let (sender, receiver) = mpsc::channel();
        self.stream_observers.push(sender);
        receiver
    }

    fn broadcast(&mut self, msg: UiMessage) {
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
        let new_volume = Gain { db: self.volume().db + volume_change.db };
        self.set_volume(new_volume);
    }

    pub fn select_output(&mut self, output: usize) {
        self.state.output = output;
        let message = UiMessage::SelectOutput { output: self.state.output };
        self.broadcast(message);
        self.on_volume_updated();
    }

    pub fn toggle_output(&mut self) {
        let next_output = (self.state.output + 1) % self.state.outputs.len();
        self.select_output(next_output);
    }

    pub fn set_mux_mode(&mut self, mux_mode: MuxMode) {
        self.state.mux_mode = mux_mode;
        self.broadcast(UiMessage::SetMuxMode { mux_mode: mux_mode });
    }

    pub fn set_enable_drc(&mut self, enable_drc: bool) {
        self.state.enable_drc = enable_drc;
        self.broadcast(UiMessage::SetEnableDrc { enable: enable_drc });
    }

    pub fn set_loudness(&mut self, loudness: LoudnessConfig) {
        self.state.loudness = loudness;
        self.on_volume_updated();
    }

    pub fn set_input_gain(&mut self, input_id: usize, gain: Gain) {
        self.state.inputs[input_id].gain = gain;
        self.broadcast(UiMessage::SetInputGain {
                           device: input_id,
                           gain: gain,
                       });
    }

    pub fn set_voice_boost(&mut self, voice_boost: FeatureConfig) {
        self.state.voice_boost = voice_boost;
        let msg = UiMessage::SetVoiceBoost {
            boost: Gain { db: self.state.voice_boost.get_level() * 20.0 },
        };
        self.broadcast(msg);
    }

    pub fn set_crossfeed(&mut self, crossfeed: CrossfeedConfig) {
        self.state.crossfeed = crossfeed;
        let msg = UiMessage::SetCrossfeed {
            level: self.state.crossfeed.get_level(),
            delay_ms: self.state.crossfeed.delay_ms,
        };
        self.broadcast(msg);
    }

    pub fn on_stream_state(&self, state: StreamState) {
        for o in self.stream_observers.iter() {
            o.send(state).unwrap();
        }
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
    pub fn new(inputs: Vec<InputState>, outputs: Vec<OutputState>) -> SharedState {
        SharedState { state: Arc::new(Mutex::new(StateController::new(inputs, outputs))) }
    }

    pub fn lock(&self) -> MutexGuard<StateController> {
        self.state.lock().unwrap()
    }
}
