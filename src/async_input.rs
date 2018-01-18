use std::thread;
use time::TimeDelta;
use std::sync::mpsc;
use input;
use base;

enum PipeMessage {
    Frame(base::Frame),
    Error(base::Error),
}

pub struct AsyncInput {
    receiver: mpsc::Receiver<PipeMessage>,
    delay: TimeDelta,
}

impl AsyncInput {
    pub fn new(mut input: Box<input::Input>) -> AsyncInput {
        let (sender, receiver) = mpsc::channel();

        let delay = input.min_delay();

        thread::spawn(move || loop {
            match input.read() {
                Err(e) => {
                    sender.send(PipeMessage::Error(e)).unwrap();
                    break;
                }
                Ok(None) => (),
                Ok(Some(frame)) => {
                    sender.send(PipeMessage::Frame(frame)).unwrap();
                }
            }
        });

        AsyncInput {
            receiver: receiver,
            delay: delay,
        }
    }

    pub fn read(&mut self, timeout: TimeDelta) -> base::Result<Option<base::Frame>> {
        match self.receiver.recv_timeout(timeout.as_duration()) {
            Ok(PipeMessage::Frame(frame)) => Ok(Some(frame)),
            Ok(PipeMessage::Error(e)) => Err(e),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(base::Error::new("Channel closed")),
        }
    }

    pub fn min_delay(&self) -> TimeDelta {
        self.delay
    }
}
