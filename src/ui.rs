extern crate mono_display;
use self::mono_display::gfx;
use self::mono_display::DisplayDriver;

use base;
use base::{Error, Result};
use state;
use std;
use std::boxed::Box;
use std::sync::mpsc;
use std::sync::Arc;
use time::{Time, TimeDelta};

impl From<mono_display::Error> for Error {
    fn from(e: mono_display::Error) -> Error {
        Error::from_string(format!("Display error: {:?}", e))
    }
}

pub enum Key {
    Rotary,
    VolumeUp,
    VolumeDown,
    Mute,
    Menu,
}

pub enum InputEvent {
    Rotate(i32),
    Pressed(Key),
    Released(Key),
}

pub type EventSink = mpsc::Sender<InputEvent>;
pub type EventSource = mpsc::Receiver<InputEvent>;

pub struct EventPipe {
    event_source: EventSource,
    event_sink: mpsc::Sender<InputEvent>,
}

impl EventPipe {
    pub fn new() -> EventPipe {
        let (event_sink, event_source) = mpsc::channel();
        EventPipe {
            event_source,
            event_sink,
        }
    }

    pub fn get_event_sink(&self) -> EventSink {
        self.event_sink.clone()
    }

    pub fn start_ui_thread(
        self,
        mut user_interface: Box<UserInterface>,
        shared_state: state::SharedState,
    ) {
        std::thread::spawn(move || {
            user_interface.run_loop(self.event_source, shared_state);
        });
    }
}

pub trait UserInterface: Send {
    fn run_loop(&mut self, event_source: EventSource, shared_state: state::SharedState);
}

pub struct HeadlessUi {}

impl HeadlessUi {
    pub fn new() -> Box<UserInterface> {
        Box::new(HeadlessUi {})
    }
}

impl UserInterface for HeadlessUi {
    fn run_loop(&mut self, event_source: EventSource, shared_state: state::SharedState) {
        loop {
            match event_source.recv() {
                Err(_) => break,
                Ok(InputEvent::Rotate(d)) => {
                    shared_state
                        .lock()
                        .move_volume(base::Gain { db: 0.5 * d as f32 });
                }
                Ok(_) => (),
            }
        }
    }
}

#[cfg(feature = "sdl2")]
fn create_display_driver() -> mono_display::Result<mono_display::sdl_driver::SdlDriver> {
    mono_display::sdl_driver::SdlDriver::new(gfx::Size::wh(128, 32))
}

#[cfg(not(feature = "sdl2"))]
fn create_display_driver() -> mono_display::Result<mono_display::ssd1306::Ssd1306> {
    mono_display::ssd1306::Ssd1306::new(mono_display::ssd1306::Ssd1306Type::S128x32, false)
}

enum EventResult {
    NoChange,
    GoBack,
    NextScreen,
}

struct Resources {
    font_bold: gfx::Font,
    font: gfx::Font,
}

trait Screen: Send {
    fn reset(&mut self, shared_state: &mut state::SharedState);
    fn render_frame(&mut self, shared_state: &mut state::SharedState, canvas: &mut gfx::Canvas);
    fn process_event(
        &mut self,
        shared_state: &mut state::SharedState,
        event: InputEvent,
    ) -> EventResult;
}

struct ScreenBase {
    resources: Arc<Resources>,
}

impl ScreenBase {
    fn render_header(&self, canvas: &mut gfx::Canvas, header: &str) {
        canvas.draw_text(
            gfx::Vector::xy(0, 10),
            &self.resources.font_bold,
            header,
            gfx::Color::Light,
        );
    }
}

#[derive(Eq, PartialEq, Clone, Copy)]
enum MainScreenMode {
    Empty,
    Levels,
}

impl MainScreenMode {
    fn next(self) -> MainScreenMode {
        match self {
            MainScreenMode::Empty => MainScreenMode::Levels,
            MainScreenMode::Levels => MainScreenMode::Empty,
        }
    }
}

struct MainScreen {
    base: ScreenBase,
    mode: MainScreenMode,
    last_stream_updated: Time,
    shared_stream_state: state::SharedStreamInfo,
    left_peak: f32,
    right_peak: f32,
}

impl MainScreen {
    fn new(resources: Arc<Resources>, shared_stream_state: state::SharedStreamInfo) -> MainScreen {
        MainScreen {
            base: ScreenBase { resources },
            mode: MainScreenMode::Empty,
            last_stream_updated: Time::now(),
            shared_stream_state,
            left_peak: 0.0,
            right_peak: 0.0,
        }
    }
}

const PEAK_DECAY_PER_SECOND: f32 = 1.0;

impl Screen for MainScreen {
    fn reset(&mut self, _shared_state: &mut state::SharedState) {}
    fn render_frame(&mut self, shared_state: &mut state::SharedState, canvas: &mut gfx::Canvas) {
        if self.mode == MainScreenMode::Empty {
            return;
        }

        let mut volume;
        {
            let state = shared_state.lock();
            if state.state().stream_state != state::StreamState::Active {
                return;
            }
            volume = state.volume();
        }

        let mut rms = 0.0;
        {
            let stream_info = self.shared_stream_state.lock();
            if !stream_info.packets.is_empty() {
                let now = stream_info.packets.back().unwrap().time;
                let time_elapse = now - self.last_stream_updated;

                self.left_peak -= time_elapse.in_seconds_f() as f32 * PEAK_DECAY_PER_SECOND;
                self.right_peak -= time_elapse.in_seconds_f() as f32 * PEAK_DECAY_PER_SECOND;

                let mut rms_sum = 0.0;
                for packet in stream_info.packets.iter() {
                    if packet.time > self.last_stream_updated {
                        self.left_peak = self.left_peak.max(packet.left.peak);
                        self.right_peak = self.right_peak.max(packet.right.peak);
                    }
                    rms_sum += packet.left.rms + packet.right.rms;
                }

                self.last_stream_updated = now;
                rms = rms_sum / (stream_info.packets.len() * 2) as f32;
            }
        }

        self.left_peak = self.left_peak.max(0.0).min(1.0);
        self.right_peak = self.right_peak.max(0.0).min(1.0);

        volume = (volume + rms.log(10.0) * 20.0).max(0.0);

        volume = (volume * 2.0).round() / 2.0;
        canvas.draw_text(
            gfx::Vector::xy(40, 10),
            &self.base.resources.font,
            format!("{:.1} db", volume).as_str(),
            gfx::Color::Light,
        );

        let left_w = (60.0 * self.left_peak).round() as i16;
        canvas.draw_rect(gfx::Rect::ltrb(63 - left_w, 19, 63, 24), gfx::Color::Light);

        let right_w = (60.0 * self.right_peak).round() as i16;
        canvas.draw_rect(gfx::Rect::ltrb(65, 19, 65 + right_w, 24), gfx::Color::Light);
    }

    fn process_event(
        &mut self,
        shared_state: &mut state::SharedState,
        event: InputEvent,
    ) -> EventResult {
        if shared_state.lock().state().stream_state != state::StreamState::Active {
            return EventResult::NoChange;
        }
        match event {
            InputEvent::Pressed(Key::Rotary) => {
                self.mode = self.mode.next();
                EventResult::NoChange
            }
            InputEvent::Released(_) => EventResult::NoChange,
            _ => EventResult::NextScreen,
        }
    }
}

trait SliderDelegate<T>: Send {
    fn new() -> T;
    fn title<'a>(&'a self) -> &'a str;
    fn min(&self) -> f32;
    fn max(&self) -> f32;
    fn step(&self) -> f32;
    fn get_value(&self, shared_state: &mut state::SharedState) -> f32;
    fn set_value(&self, shared_state: &mut state::SharedState, value: f32);
}

struct Volume {}

impl SliderDelegate<Volume> for Volume {
    fn new() -> Volume {
        Volume {}
    }

    fn title<'a>(&'a self) -> &'a str {
        "Volume"
    }
    fn min(&self) -> f32 {
        30.0
    }
    fn max(&self) -> f32 {
        110.0
    }
    fn step(&self) -> f32 {
        0.5
    }
    fn get_value(&self, shared_state: &mut state::SharedState) -> f32 {
        shared_state.lock().volume()
    }
    fn set_value(&self, shared_state: &mut state::SharedState, value: f32) {
        shared_state.lock().set_volume(value)
    }
}

struct LoudnessBase {}

impl SliderDelegate<LoudnessBase> for LoudnessBase {
    fn new() -> LoudnessBase {
        LoudnessBase {}
    }

    fn title<'a>(&'a self) -> &'a str {
        "Loudness base"
    }
    fn min(&self) -> f32 {
        80.0
    }
    fn max(&self) -> f32 {
        110.0
    }
    fn step(&self) -> f32 {
        1.0
    }
    fn get_value(&self, shared_state: &mut state::SharedState) -> f32 {
        shared_state.lock().state().loudness.base_level_spl
    }
    fn set_value(&self, shared_state: &mut state::SharedState, value: f32) {
        let mut s = shared_state.lock();
        let mut cfg = s.state().loudness;
        cfg.base_level_spl = value;
        s.set_loudness(cfg)
    }
}

struct LoudnessLevel {}

impl SliderDelegate<LoudnessLevel> for LoudnessLevel {
    fn new() -> LoudnessLevel {
        LoudnessLevel {}
    }

    fn title<'a>(&'a self) -> &'a str {
        "Loudness level"
    }
    fn min(&self) -> f32 {
        0.0
    }
    fn max(&self) -> f32 {
        1.0
    }
    fn step(&self) -> f32 {
        0.1
    }
    fn get_value(&self, shared_state: &mut state::SharedState) -> f32 {
        shared_state.lock().state().loudness.level
    }
    fn set_value(&self, shared_state: &mut state::SharedState, value: f32) {
        let mut s = shared_state.lock();
        let mut cfg = s.state().loudness;
        cfg.level = value;
        s.set_loudness(cfg)
    }
}

struct SliderScreen<T>
where
    T: SliderDelegate<T>,
{
    base: ScreenBase,
    delegate: T,
}

impl<T> SliderScreen<T>
where
    T: 'static + SliderDelegate<T>,
{
    fn new(resources: Arc<Resources>) -> Box<Screen> {
        Box::new(SliderScreen::<T> {
            base: ScreenBase { resources },
            delegate: T::new(),
        })
    }
}

impl<T> Screen for SliderScreen<T>
where
    T: SliderDelegate<T>,
{
    fn reset(&mut self, _shared_state: &mut state::SharedState) {}
    fn render_frame(&mut self, shared_state: &mut state::SharedState, canvas: &mut gfx::Canvas) {
        self.base.render_header(canvas, self.delegate.title());
        let value = self.delegate.get_value(shared_state);

        let width = 95;
        let pos = ((value - self.delegate.min()) / (self.delegate.max() - self.delegate.min())
            * width as f32) as i16;
        canvas.draw_rect(gfx::Rect::ltrb(0, 18, width + 2, 32), gfx::Color::Light);
        canvas.draw_rect(
            gfx::Rect::ltrb(pos + 1, 19, width + 1, 31),
            gfx::Color::Dark,
        );

        let value_string = format!("{:.1}", value);
        let value_str = value_string.as_str();
        let text_rect =
            canvas.get_text_rect(gfx::Vector::xy(0, 0), &self.base.resources.font, value_str);
        canvas.draw_text(
            gfx::Vector::xy(128 - text_rect.size.width as i16, 30),
            &self.base.resources.font,
            value_str,
            gfx::Color::Light,
        );
    }

    fn process_event(
        &mut self,
        shared_state: &mut state::SharedState,
        event: InputEvent,
    ) -> EventResult {
        match event {
            InputEvent::Rotate(change) => {
                let v = self.delegate.get_value(shared_state);
                let new_v = state::limit(
                    self.delegate.min(),
                    self.delegate.max(),
                    v + change as f32 * self.delegate.step(),
                );
                if v != new_v {
                    self.delegate.set_value(shared_state, new_v);
                }
            }
            InputEvent::Pressed(Key::Rotary) => return EventResult::NextScreen,
            _ => {}
        }
        EventResult::NoChange
    }
}

trait SelectScreenDelegate<T>
where
    T: Send,
{
    fn new() -> T;
    fn reset(&mut self, shared_state: &mut state::SharedState) -> usize;
    fn title<'a>(&'a self) -> &'a str;
    fn get_num_items(&self) -> usize;
    fn get_item<'a>(&'a self, index: usize) -> &'a str;
    fn on_selected(&mut self, shared_state: &mut state::SharedState, index: usize);
    fn on_commited(&mut self, shared_state: &mut state::SharedState, index: usize);
}

struct SelectScreen<T>
where
    T: SelectScreenDelegate<T> + Send,
{
    base: ScreenBase,
    delegate: T,
    changed: bool,
    current: usize,
}

impl<T> SelectScreen<T>
where
    T: SelectScreenDelegate<T> + Send + 'static,
{
    fn new(resources: Arc<Resources>) -> Box<Screen> {
        Box::new(SelectScreen::<T> {
            base: ScreenBase { resources },
            delegate: T::new(),
            changed: false,
            current: 0,
        })
    }
}

impl<T> Screen for SelectScreen<T>
where
    T: SelectScreenDelegate<T> + Send,
{
    fn reset(&mut self, shared_state: &mut state::SharedState) {
        self.changed = false;
        self.current = self.delegate.reset(shared_state);
    }
    fn render_frame(&mut self, _shared_state: &mut state::SharedState, canvas: &mut gfx::Canvas) {
        self.base.render_header(canvas, self.delegate.title());
        canvas.draw_text(
            gfx::Vector::xy(0, 27),
            &self.base.resources.font,
            self.delegate.get_item(self.current),
            gfx::Color::Light,
        );
    }
    fn process_event(
        &mut self,
        shared_state: &mut state::SharedState,
        event: InputEvent,
    ) -> EventResult {
        match event {
            InputEvent::Rotate(change) => {
                let inc = if change > 0 {
                    1
                } else {
                    self.delegate.get_num_items() - 1
                };
                self.current = (self.current + inc) % self.delegate.get_num_items();
                self.changed = true;
                self.delegate.on_selected(shared_state, self.current);
            }
            InputEvent::Pressed(Key::Rotary) => {
                if self.changed {
                    self.delegate.on_commited(shared_state, self.current);
                    return EventResult::GoBack;
                } else {
                    return EventResult::NextScreen;
                }
            }
            _ => {}
        }
        EventResult::NoChange
    }
}

struct OutputSelectorEntry {
    title: String,
    output: usize,
    speakers: Option<usize>,
}

struct OutputSelector {
    choices: Vec<OutputSelectorEntry>,
}

impl SelectScreenDelegate<OutputSelector> for OutputSelector {
    fn new() -> OutputSelector {
        OutputSelector {
            choices: Vec::new(),
        }
    }
    fn reset(&mut self, shared_state: &mut state::SharedState) -> usize {
        let mut selected = 0;
        self.choices.clear();
        let lock = shared_state.lock();
        let state = lock.state();
        for (out_index, out) in state.outputs.iter().enumerate() {
            if out.speakers.is_empty() {
                if out_index == state.current_output {
                    selected = self.choices.len()
                }
                self.choices.push(OutputSelectorEntry {
                    title: out.name.clone(),
                    output: out_index,
                    speakers: None,
                })
            } else {
                for (speakers_index, speakers) in out.speakers.iter().enumerate() {
                    if out_index == state.current_output
                        && Some(speakers_index) == out.current_speakers
                    {
                        selected = self.choices.len()
                    }
                    self.choices.push(OutputSelectorEntry {
                        title: speakers.name.clone(),
                        output: out_index,
                        speakers: Some(speakers_index),
                    })
                }
            }
        }
        selected
    }
    fn title<'a>(&'a self) -> &'a str {
        "Output"
    }
    fn get_num_items(&self) -> usize {
        self.choices.len()
    }
    fn get_item<'a>(&'a self, index: usize) -> &'a str {
        self.choices[index].title.as_str()
    }
    fn on_selected(&mut self, _shared_state: &mut state::SharedState, _index: usize) {}
    fn on_commited(&mut self, shared_state: &mut state::SharedState, index: usize) {
        let mut state = shared_state.lock();
        state.select_output(self.choices[index].output, self.choices[index].speakers);
    }
}

struct Crossfeed {}

impl SelectScreenDelegate<Crossfeed> for Crossfeed {
    fn new() -> Crossfeed {
        Crossfeed {}
    }
    fn reset(&mut self, shared_state: &mut state::SharedState) -> usize {
        if shared_state
            .lock()
            .state()
            .enable_crossfeed
            .unwrap_or(false)
        {
            1
        } else {
            0
        }
    }
    fn title<'a>(&'a self) -> &'a str {
        "Cressfeed"
    }
    fn get_num_items(&self) -> usize {
        2
    }
    fn get_item<'a>(&'a self, index: usize) -> &'a str {
        ["NO", "YES"][index]
    }
    fn on_selected(&mut self, shared_state: &mut state::SharedState, index: usize) {
        shared_state.lock().set_enable_crossfeed(index > 0);
    }
    fn on_commited(&mut self, _shared_state: &mut state::SharedState, _index: usize) {}
}

struct LoudnessCorrection {}

impl SelectScreenDelegate<LoudnessCorrection> for LoudnessCorrection {
    fn new() -> LoudnessCorrection {
        LoudnessCorrection {}
    }
    fn reset(&mut self, shared_state: &mut state::SharedState) -> usize {
        if shared_state.lock().state().loudness.auto {
            1
        } else {
            0
        }
    }
    fn title<'a>(&'a self) -> &'a str {
        "Loudness Correction"
    }
    fn get_num_items(&self) -> usize {
        2
    }
    fn get_item<'a>(&'a self, index: usize) -> &'a str {
        ["NO", "YES"][index]
    }
    fn on_selected(&mut self, shared_state: &mut state::SharedState, index: usize) {
        let mut s = shared_state.lock();
        let mut cfg = s.state().loudness;
        cfg.enabled = index > 0;
        s.set_loudness(cfg)
    }
    fn on_commited(&mut self, _shared_state: &mut state::SharedState, _index: usize) {}
}

struct AutoLoudness {}

impl SelectScreenDelegate<AutoLoudness> for AutoLoudness {
    fn new() -> AutoLoudness {
        AutoLoudness {}
    }
    fn reset(&mut self, shared_state: &mut state::SharedState) -> usize {
        if shared_state.lock().state().loudness.auto {
            1
        } else {
            0
        }
    }
    fn title<'a>(&'a self) -> &'a str {
        "Auto Loudness"
    }
    fn get_num_items(&self) -> usize {
        2
    }
    fn get_item<'a>(&'a self, index: usize) -> &'a str {
        ["NO", "YES"][index]
    }
    fn on_selected(&mut self, shared_state: &mut state::SharedState, index: usize) {
        let mut s = shared_state.lock();
        let mut cfg = s.state().loudness;
        cfg.auto = index > 0;
        s.set_loudness(cfg)
    }
    fn on_commited(&mut self, _shared_state: &mut state::SharedState, _index: usize) {}
}

#[derive(Copy, Clone)]
enum DisplayState {
    Main,
    Menu(usize),
}

pub struct DisplayUi {
    main_screen: MainScreen,
    menu: Vec<Box<Screen>>,
    state: DisplayState,
}

impl DisplayUi {
    pub fn new(shared_stream_state: state::SharedStreamInfo) -> Result<Box<DisplayUi>> {
        let font_bold =
            gfx::Font::from_data_bdf(include_bytes!("../resources/gohu-14-bold-narrow.bdf"))?;
        let font = gfx::Font::from_data_bdf(include_bytes!("../resources/gohu-14-narrow.bdf"))?;
        let resources = Arc::new(Resources { font_bold, font });

        let mut menu = Vec::new();
        menu.push(SliderScreen::<Volume>::new(resources.clone()));
        menu.push(SelectScreen::<OutputSelector>::new(resources.clone()));
        menu.push(SelectScreen::<Crossfeed>::new(resources.clone()));
        menu.push(SelectScreen::<LoudnessCorrection>::new(resources.clone()));
        menu.push(SelectScreen::<AutoLoudness>::new(resources.clone()));
        menu.push(SliderScreen::<LoudnessBase>::new(resources.clone()));
        menu.push(SliderScreen::<LoudnessLevel>::new(resources.clone()));

        Ok(Box::new(DisplayUi {
            main_screen: MainScreen::new(resources.clone(), shared_stream_state),
            menu,
            state: DisplayState::Main,
        }))
    }

    fn render_frame(&mut self, shared_state: &mut state::SharedState, canvas: &mut gfx::Canvas) {
        match self.state {
            DisplayState::Main => self.main_screen.render_frame(shared_state, canvas),
            DisplayState::Menu(item) => self.menu[item].render_frame(shared_state, canvas),
        }
    }

    fn process_event(&mut self, shared_state: &mut state::SharedState, event: InputEvent) {
        let result = match self.state {
            DisplayState::Main => self.main_screen.process_event(shared_state, event),
            DisplayState::Menu(item) => self.menu[item].process_event(shared_state, event),
        };
        self.state = match (result, self.state) {
            (EventResult::NoChange, _) => self.state,
            (EventResult::GoBack, _) => DisplayState::Main,
            (EventResult::NextScreen, DisplayState::Menu(item)) => {
                let item = (item + 1) % self.menu.len();
                self.menu[item].reset(shared_state);
                DisplayState::Menu(item)
            }
            (EventResult::NextScreen, DisplayState::Main) => DisplayState::Menu(0),
        }
    }
}

const INPUT_TIMEOUT_SECONDS: i64 = 5;

impl UserInterface for DisplayUi {
    fn run_loop(&mut self, event_source: EventSource, mut shared_state: state::SharedState) {
        let mut display = match create_display_driver() {
            Ok(d) => d,
            Err(e) => {
                println!("ERROR: Failed to initialize display driver {:?}", e);
                return;
            }
        };

        let mut last_event_time = Time::now();
        loop {
            let start = Time::now();
            if start - last_event_time > TimeDelta::seconds(INPUT_TIMEOUT_SECONDS) {
                self.state = DisplayState::Main;
            }

            // Render frame.
            let mut canvas = gfx::Canvas::new(display.get_frame());
            self.render_frame(&mut shared_state, &mut canvas);
            display.show_frame(canvas.take_frame());

            // Process events.
            match event_source
                .recv_timeout((start + TimeDelta::milliseconds(50) - Time::now()).as_duration())
            {
                Err(mpsc::RecvTimeoutError::Timeout) => (),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Ok(event) => {
                    last_event_time = Time::now();
                    self.process_event(&mut shared_state, event);
                }
            }
        }
    }
}
