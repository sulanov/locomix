extern crate rppal;
extern crate toml;

use base::*;
use gpio;
use std;
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
    gpio_channel: std::sync::mpsc::Receiver<(Pin, bool)>,
    event_sink: ui::EventSink,
}

#[derive(Debug)]
enum Pin {
    A,
    B,
    C,
}

fn set_pin_level(sender: &std::sync::mpsc::Sender<(Pin, bool)>, pin: Pin, l: rppal::gpio::Level) {
    let value = if l == rppal::gpio::Level::Low {
        true
    } else {
        false
    };
    sender
        .send((pin, value))
        .unwrap_or_else(|e| println!("ERROR: Failed to send gpio message: {}.", e));
}

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

        let (sender, receiver) = std::sync::mpsc::channel();

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
            gpio_channel: receiver,
            event_sink,
        })
    }

    fn poll_event(&mut self) -> Result<Option<ui::InputEvent>> {
        match self.gpio_channel.recv().unwrap() {
            (Pin::A, true) => {
                self.pin_a_state = true;
                if !self.pin_b_state {
                    return Ok(Some(ui::InputEvent::Rotate(1)));
                }
            }
            (Pin::A, false) => {
                self.pin_a_state = false;
            }
            (Pin::B, true) => {
                self.pin_b_state = true;
                if !self.pin_a_state {
                    return Ok(Some(ui::InputEvent::Rotate(-1)));
                }
            }
            (Pin::B, false) => {
                self.pin_b_state = false;
            }
            (Pin::C, true) => {
                return Ok(Some(ui::InputEvent::Pressed(ui::Key::Rotary)));
            }
            (Pin::C, false) => {
                return Ok(Some(ui::InputEvent::Released(ui::Key::Rotary)));
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
