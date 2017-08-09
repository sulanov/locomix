use input_device::*;
use std;
use time::{Time, TimeDelta};
use ui::*;

const REOPEN_PERIOD_SECONDS: i64 = 3;

struct LightController {
    device_path: String,
    dev: Option<InputDevice>,
    state: SharedState,
    ui_receiver: UiMessageReceiver,
    last_open_attempt: Time,
}

impl LightController {
    fn new(device_path: &str, state: SharedState) -> LightController {
        let receiver = state.lock().add_observer();
        LightController {
            device_path: String::from(device_path),
            dev: None,
            state: state,
            ui_receiver: receiver,
            last_open_attempt: Time::now() - TimeDelta::seconds(REOPEN_PERIOD_SECONDS),
        }
    }

    fn try_open(&mut self, log_error: bool) {
        if self.dev.is_some() ||
            Time::now() - self.last_open_attempt < TimeDelta::seconds(REOPEN_PERIOD_SECONDS)
        {
            return;
        }
        match InputDevice::new(&self.device_path) {
            Ok(d) => {
                println!("INFO: Controlling light via {}", self.device_path);
                self.dev = Some(d)
            }
            Err(e) => if log_error {
                println!("ERROR: Failed to open {}: {}", self.device_path, e);
            },
        }
    }

    fn set_light(&mut self, level: f32) {
        self.try_open(false);

        let mut reset = false;
        match self.dev.as_mut() {
            Some(ref mut dev) => {
                let mut e = InputEvent::zero();
                e.type_ = EV_MSC;
                e.code = MSC_PULSELED;
                e.value = (level * 32.0) as u32;
                match dev.write(e) {
                    Ok(_) => (),
                    Err(e) => {
                        println!(
                            "WARNING: failed to write input event to {}: {}",
                            self.device_path,
                            e
                        );
                        reset = true;
                    }
                }
            }
            None => {
                reset = true;
            }
        }
        if reset {
            self.dev = None;
        }

    }

    fn run(&mut self) {
        self.try_open(true);

        loop {
            match self.ui_receiver.recv() {
                Ok(_) => (),
                Err(_) => return,
            }

            // Skip all pending messages if there were more than one.
            for _ in self.ui_receiver.try_iter() {}

            let volume = self.state.lock().volume().db;

            self.set_light(0.0);
            std::thread::sleep(TimeDelta::milliseconds(50).as_duration());
            self.set_light((volume - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN));
            std::thread::sleep(TimeDelta::milliseconds(50).as_duration());
        }
    }
}

pub fn start_light_controller(device_path: &str, shared_state: SharedState) {
    let mut c = LightController::new(device_path, shared_state);
    std::thread::spawn(move || { c.run(); });
}
