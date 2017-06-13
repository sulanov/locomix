use input_device::*;
use ui::*;
use std;

fn try_open(device_path: &str, log_error: bool) -> Option<InputDevice> {
    match InputDevice::new(device_path) {
        Ok(d) => {
            println!("INFO: Opened {}", device_path);
            match d.set_repeat_rate(200, 100) {
                Err(e) => {
                    println!("WARNING: Failed to set repeat rate for {}: {}",
                             device_path,
                             e)
                }
                Ok(_) => (),
            }
            Some(d)
        }
        Err(e) => {
            if log_error {
                println!("ERROR: Failed to open {}: {}", device_path, e);
            }
            None
        }
    }
}

fn process_event(shared_state: &mut SharedState, event: InputEvent) {
    match event {
        InputEvent {
            time: _,
            type_: EV_KEY,
            code: KEY_VOLUMEUP,
            value: EV_PRESSED...EV_REPEATED,
        } => {
            shared_state.lock().move_volume(Gain { db: 1.0 });
        }

        InputEvent {
            time: _,
            type_: EV_KEY,
            code: KEY_VOLUMEDOWN,
            value: EV_PRESSED...EV_REPEATED,
        } => {
            shared_state.lock().move_volume(Gain { db: -1.0 });
        }

        InputEvent {
            time: _,
            type_: EV_KEY,
            code: KEY_MUTE,
            value: EV_PRESSED,
        } => {
            shared_state.lock().toggle_output();
        }

        InputEvent {
            time: _,
            type_: EV_KEY,
            code: KEY_MENU,
            value: EV_PRESSED,
        } => {
            let mut state = shared_state.lock();
            let mut loudness_config = state.state().loudness.clone();
            loudness_config.enabled = !loudness_config.enabled;
            state.set_loudness(loudness_config);
        }

        InputEvent {
            time: _,
            type_: EV_KEY,
            code: KEY_HOMEPAGE,
            value: EV_PRESSED,
        } => {
            let mut state = shared_state.lock();
            let mut voice_boost_config = state.state().voice_boost.clone();
            voice_boost_config.enabled = !voice_boost_config.enabled;
            state.set_voice_boost(voice_boost_config);
        }

        InputEvent {
            time: _,
            type_: EV_REL,
            code: REL_DIAL,
            value: change,
        } => {
            shared_state
                .lock()
                .move_volume(Gain { db: (change as i32) as f32 / 2.0 });
        }

        InputEvent {
            time: _,
            type_: EV_KEY,
            code: BTN_MISC,
            value: EV_PRESSED,
        } => {
            println!("Griffin PowerMate pushed");
        }

        _ => (),
    }
}

fn handle_input_device(device_path: &str, mut shared_state: SharedState) {
    let mut dev: Option<InputDevice> = try_open(device_path, true);
    loop {
        let mut reopen = false;
        match dev.as_mut() {
            Some(ref mut dev) => {
                match dev.read() {
                    Ok(event) => process_event(&mut shared_state, event),
                    Err(e) => {
                        println!("WARNING: failed to read input event from {}: {}",
                                 device_path,
                                 e);
                        reopen = true;
                    }
                }
            }
            None => {
                reopen = true;
            }
        }
        if reopen {
            dev = try_open(device_path, false);
            if dev.is_none() {
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        }
    }

}

pub fn start_input_handler(device_path: &str, shared_state: SharedState) {
    let device_path_copy = String::from(device_path);
    std::thread::spawn(move || { handle_input_device(&device_path_copy, shared_state); });
}
