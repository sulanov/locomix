use base;
use input;
use std::sync::mpsc;
use std::thread;
use time::TimeDelta;

enum PipeMessage {
    Frame(base::Frame, TimeDelta),
    Error(base::Error),
}

pub struct AsyncInput {
    receiver: mpsc::Receiver<PipeMessage>,
    min_delay: TimeDelta,
}

impl AsyncInput {
    pub fn new(mut input: Box<dyn input::Input>) -> AsyncInput {
        let (sender, receiver) = mpsc::channel();

        let min_delay = input.min_delay();

        thread::spawn(move || loop {
            match input.read() {
                Err(e) => {
                    sender.send(PipeMessage::Error(e)).unwrap();
                    break;
                }
                Ok(None) => (),
                Ok(Some(frame)) => {
                    sender
                        .send(PipeMessage::Frame(frame, input.min_delay()))
                        .unwrap();
                }
            }
        });

        AsyncInput {
            receiver: receiver,
            min_delay: min_delay,
        }
    }

    pub fn read(&mut self, timeout: TimeDelta) -> base::Result<Option<base::Frame>> {
        match self.receiver.recv_timeout(timeout.as_duration()) {
            Ok(PipeMessage::Frame(frame, min_delay)) => {
                self.min_delay = min_delay;
                Ok(Some(frame))
            }
            Ok(PipeMessage::Error(e)) => Err(e),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(base::Error::new("Channel closed")),
        }
    }

    pub fn min_delay(&self) -> TimeDelta {
        self.min_delay
    }
}
