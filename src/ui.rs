use base;
use state;
use std;
use std::sync::mpsc;

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

pub struct UserInterface {
    shared_state: state::SharedState,
    event_receiver: mpsc::Receiver<InputEvent>,
    event_sender: mpsc::Sender<InputEvent>,
}

impl UserInterface {
    pub fn new(shared_state: state::SharedState) -> UserInterface {
        let (event_sender, event_receiver) = mpsc::channel();
        UserInterface {
            shared_state,
            event_receiver,
            event_sender,
        }
    }

    pub fn get_event_sink(&self) -> EventSink {
        self.event_sender.clone()
    }

    fn run_loop(&mut self) {
        loop {
            match self.event_receiver.recv() {
                Err(_) => break,
                Ok(InputEvent::Rotate(d)) => self
                    .shared_state
                    .lock()
                    .move_volume(base::Gain { db: 0.5 * d as f32 }),
                Ok(_) => (),
            }
        }
    }

    pub fn start_loop(mut self) {
        std::thread::spawn(move || {
            self.run_loop();
        });
    }
}
