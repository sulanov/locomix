extern crate rppal;
extern crate toml;

use base::*;
use gpio;
use std;
use std::sync::mpsc;
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
    pin_a_state: bool,
    pin_b_state: bool,

    pin_c_state: bool,
    reported_c_state: bool,
    last_c_change: Option<Time>,

    gpio_channel: mpsc::Receiver<(Pin, bool, Time)>,
    event_sink: ui::EventSink,
}

#[derive(Debug)]
enum Pin {
    A,
    B,
    C,
}

fn set_pin_level(sender: &mpsc::Sender<(Pin, bool, Time)>, pin: Pin, l: rppal::gpio::Level) {
    let value = if l == rppal::gpio::Level::Low {
        true
    } else {
        false
    };
    sender
        .send((pin, value, Time::now()))
        .unwrap_or_else(|e| println!("ERROR: Failed to send gpio message: {}.", e));
}

const BUTTON_DELAY_MS: i64 = 200;

impl RotaryEncoder {
    fn new(config: &toml::value::Table, event_sink: ui::EventSink) -> Result<RotaryEncoder> {
        let pin_a = parse_pin_param(config, "pin_a")?;
        let pin_b = parse_pin_param(config, "pin_b")?;
        let pin_c = parse_pin_param(config, "pin_c")?;

        let mut gpio_locked_o = gpio::get_gpio();
        let gpio_locked = gpio_locked_o.as_mut().unwrap();
        gpio_locked.set_mode(pin_a, rppal::gpio::Mode::Input);
        gpio_locked.set_pullupdown(pin_a, rppal::gpio::PullUpDown::PullUp);
        gpio_locked.set_mode(pin_b, rppal::gpio::Mode::Input);
        gpio_locked.set_pullupdown(pin_b, rppal::gpio::PullUpDown::PullUp);
        gpio_locked.set_mode(pin_c, rppal::gpio::Mode::Input);
        gpio_locked.set_pullupdown(pin_c, rppal::gpio::PullUpDown::PullUp);

        let (sender, receiver) = mpsc::channel();

        let sender_a = sender.clone();
        gpio_locked.set_async_interrupt(pin_a, rppal::gpio::Trigger::Both, move |l| {
            set_pin_level(&sender_a, Pin::A, l);
        })?;

        let sender_b = sender.clone();
        gpio_locked.set_async_interrupt(pin_b, rppal::gpio::Trigger::Both, move |l| {
            set_pin_level(&sender_b, Pin::B, l);
        })?;

        gpio_locked.set_async_interrupt(pin_c, rppal::gpio::Trigger::Both, move |l| {
            set_pin_level(&sender, Pin::C, l);
        })?;

        Ok(RotaryEncoder {
            pin_a_state: false,
            pin_b_state: false,
            pin_c_state: false,
            reported_c_state: false,
            last_c_change: None,
            gpio_channel: receiver,
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
        let event = match self.last_c_change.clone() {
            Some(time) => {
                let retrigger_c_event_time = time + TimeDelta::milliseconds(BUTTON_DELAY_MS);
                let timeout = retrigger_c_event_time - Time::now();
                match self.gpio_channel.recv_timeout(timeout.as_duration()) {
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        self.last_c_change = None;
                        return Ok(self.get_c_event());
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(Error::new("Listeners were dropped"))
                    }
                    Ok(event) => event,
                }
            }
            None => self.gpio_channel.recv().unwrap(),
        };

        match event {
            (Pin::A, true, _) => {
                self.pin_a_state = true;
                if !self.pin_b_state {
                    return Ok(Some(ui::InputEvent::Rotate(1)));
                }
            }
            (Pin::A, false, _) => {
                self.pin_a_state = false;
            }
            (Pin::B, true, _) => {
                self.pin_b_state = true;
                if !self.pin_a_state {
                    return Ok(Some(ui::InputEvent::Rotate(-1)));
                }
            }
            (Pin::B, false, _) => {
                self.pin_b_state = false;
            }
            (Pin::C, state, time) => {
                self.pin_c_state = state;
                let report_event = match self.last_c_change.take() {
                    None => true,
                    Some(last_time) => time - last_time >= TimeDelta::milliseconds(BUTTON_DELAY_MS),
                };
                self.last_c_change = Some(time);
                if report_event {
                    return Ok(self.get_c_event());
                }
            }
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
    gpio::initialize()?;
    let mut encoder = RotaryEncoder::new(config, event_sink)?;
    std::thread::spawn(move || {
        encoder.run_loop();
    });
    Ok(())
}
