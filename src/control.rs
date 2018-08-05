extern crate toml;

use base::*;
use input_device::*;
use std;
use ui;

fn try_open(device_path: &str, log_error: bool) -> Option<InputDevice> {
    match InputDevice::new(device_path) {
        Ok(d) => {
            println!("INFO: Opened {}", device_path);
            match d.set_repeat_rate(200, 100) {
                Err(e) => println!(
                    "WARNING: Failed to set repeat rate for {}: {}",
                    device_path, e
                ),
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

fn process_event(event: InputEvent) -> Option<ui::InputEvent> {
    match (event.type_, event.code, event.value) {
        (EV_KEY, KEY_VOLUMEUP, EV_PRESSED...EV_REPEATED) => {
            Some(ui::InputEvent::Pressed(ui::Key::VolumeUp))
        }
        (EV_KEY, KEY_VOLUMEUP, EV_RELEASED) => Some(ui::InputEvent::Released(ui::Key::VolumeUp)),
        (EV_KEY, KEY_VOLUMEDOWN, EV_PRESSED...EV_REPEATED) => {
            Some(ui::InputEvent::Pressed(ui::Key::VolumeDown))
        }
        (EV_KEY, KEY_VOLUMEDOWN, EV_RELEASED) => {
            Some(ui::InputEvent::Released(ui::Key::VolumeDown))
        }
        (EV_KEY, KEY_MUTE, EV_PRESSED) => Some(ui::InputEvent::Pressed(ui::Key::Mute)),
        (EV_KEY, KEY_MUTE, EV_RELEASED) => Some(ui::InputEvent::Released(ui::Key::Mute)),
        (EV_KEY, KEY_MENU, EV_PRESSED) => Some(ui::InputEvent::Pressed(ui::Key::Menu)),
        (EV_KEY, KEY_MENU, EV_RELEASED) => Some(ui::InputEvent::Released(ui::Key::Menu)),
        (EV_REL, REL_DIAL, change) => Some(ui::InputEvent::Rotate(change as i32)),
        (EV_KEY, BTN_MISC, EV_PRESSED) => Some(ui::InputEvent::Pressed(ui::Key::Rotary)),
        (EV_KEY, BTN_MISC, EV_RELEASED) => Some(ui::InputEvent::Released(ui::Key::Rotary)),
        _ => None,
    }
}

fn handle_input_device(device_path: &str, event_sink: ui::EventSink) {
    let mut dev: Option<InputDevice> = try_open(device_path, true);
    loop {
        let mut reopen = false;
        match dev.as_mut() {
            Some(ref mut dev) => match dev.read() {
                Ok(event) => {
                    if let Some(event) = process_event(event) {
                        if let Err(e) = event_sink.send(event) {
                            println!("ERROR: Failed to send event: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    println!(
                        "WARNING: failed to read input event from {}: {}",
                        device_path, e
                    );
                    reopen = true;
                }
            },
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

pub fn start_input_handler(config: &toml::value::Table, event_sink: ui::EventSink) -> Result<()> {
    let device_path = config
        .get("device")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::new("device field not specified for input device"))?
        .to_string();
    std::thread::spawn(move || {
        handle_input_device(device_path.as_str(), event_sink);
    });
    Ok(())
}
