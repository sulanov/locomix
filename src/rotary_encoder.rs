extern crate rppal;
extern crate toml;

use base::*;
use std;
use time::{Time, TimeDelta};
use ui;

fn parse_pin_param(config: &toml::value::Table, name: &str) -> Result<u8> {
    match config.get(name).and_then(|v| v.as_integer()) {
        Some(v) if v > 0 && v < 30 => Ok(v as u8),
        _ => Err(Error::from_string(format!(
            "rotary_encoder: {} not specified.",
            name
        ))),
    }
}

struct RotaryEncoder {
    gpio: rppal::gpio::Gpio,

    pin_a: u8,
    pin_a_state: bool,
    pin_b: u8,
    pin_b_state: bool,

    pin_c: u8,
    pin_c_state: bool,
    reported_c_state: bool,
    last_c_change: Option<Time>,

    event_sink: ui::EventSink,
}

const BUTTON_DELAY_MS: i64 = 200;

impl RotaryEncoder {
    fn new(config: &toml::value::Table, event_sink: ui::EventSink) -> Result<RotaryEncoder> {
        let pin_a = parse_pin_param(config, "pin_a")?;
        let pin_b = parse_pin_param(config, "pin_b")?;
        let pin_c = parse_pin_param(config, "pin_c")?;

        let mut gpio = rppal::gpio::Gpio::new()?;
        gpio.set_mode(pin_a, rppal::gpio::Mode::Input);
        gpio.set_pullupdown(pin_a, rppal::gpio::PullUpDown::PullUp);
        gpio.set_mode(pin_b, rppal::gpio::Mode::Input);
        gpio.set_pullupdown(pin_b, rppal::gpio::PullUpDown::PullUp);
        gpio.set_mode(pin_c, rppal::gpio::Mode::Input);
        gpio.set_pullupdown(pin_c, rppal::gpio::PullUpDown::PullUp);

        gpio.set_interrupt(pin_a, rppal::gpio::Trigger::Both)?;
        gpio.set_interrupt(pin_b, rppal::gpio::Trigger::Both)?;
        gpio.set_interrupt(pin_c, rppal::gpio::Trigger::Both)?;

        Ok(RotaryEncoder {
            gpio,
            pin_a,
            pin_a_state: false,
            pin_b,
            pin_b_state: false,
            pin_c,
            pin_c_state: false,
            reported_c_state: false,
            last_c_change: None,
            event_sink,
        })
    }

    fn get_c_event(&mut self) -> Option<ui::InputEvent> {
        if self.reported_c_state != self.pin_c_state {
            self.reported_c_state = self.pin_c_state;
            if self.pin_c_state {
                Some(ui::InputEvent::Pressed(ui::Key::Rotary))
            } else {
                Some(ui::InputEvent::Released(ui::Key::Rotary))
            }
        } else {
            None
        }
    }

    fn poll_event(&mut self) -> Result<Option<ui::InputEvent>> {
        let timeout = self.last_c_change.map(|time| {
            let retrigger_c_event_time = time + TimeDelta::milliseconds(BUTTON_DELAY_MS);
            (retrigger_c_event_time - Time::now()).as_duration()
        });

        let event = match self.gpio.poll_interrupts(&[self.pin_a, self.pin_b, self.pin_c],
                                                /*reset=*/false, timeout)? {
            None => {
                self.last_c_change = None;
                return Ok(self.get_c_event());
            }
            Some(event) => event,
        };

        match event {
            (pin, rppal::gpio::Level::Low) if pin == self.pin_a => {
                self.pin_a_state = true;
            }
            (pin, rppal::gpio::Level::High) if pin == self.pin_a => {
                if self.pin_a_state {
                    self.pin_a_state = false;
                    if self.pin_b_state {
                        return Ok(Some(ui::InputEvent::Rotate(1)));
                    }
                }
            }
            (pin, rppal::gpio::Level::Low) if pin == self.pin_b => {
                self.pin_b_state = true;
            }
            (pin, rppal::gpio::Level::High) if pin == self.pin_b => {
                if self.pin_b_state {
                    self.pin_b_state = false;
                    if self.pin_a_state {
                        return Ok(Some(ui::InputEvent::Rotate(-1)));
                    }
                }
            }
            (pin, state) if pin == self.pin_c => {
                self.pin_c_state = state == rppal::gpio::Level::Low;
                let now = Time::now();
                let report_event = match self.last_c_change.take() {
                    None => true,
                    Some(last_time) => now - last_time >= TimeDelta::milliseconds(BUTTON_DELAY_MS),
                };
                self.last_c_change = Some(now);
                if report_event {
                    return Ok(self.get_c_event());
                }
            }
            _ => ()
        }
        Ok(None)
    }

    fn run_loop(&mut self) {
        loop {
            match self.poll_event() {
                Err(e) => println!("ERROR: Failed to read GPIO: {:?}", e),
                Ok(Some(event)) => {
                    if let Err(e) = self.event_sink.send(event) {
                        println!("ERROR: Failed to send input event: {:?}", e);
                    }
                }
                _ => (),
            }
        }
    }
}

pub fn start_rotary_encoder_handler(
    config: &toml::value::Table,
    event_sink: ui::EventSink,
) -> Result<()> {
    let mut encoder = RotaryEncoder::new(config, event_sink)?;
    std::thread::spawn(move || {
        encoder.run_loop();
    });
    Ok(())
}
