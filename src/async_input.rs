use std::thread;
use time::TimeDelta;
use std::sync::mpsc;
use input;
use base;
use scheduler;

enum PipeMessage {
    Frame(base::Frame),
    Error(base::Error),
}

pub struct AsyncInput {
    receiver: mpsc::Receiver<PipeMessage>,
}

impl AsyncInput {
    pub fn new(mut input: Box<input::Input>, affinity: Option<scheduler::CpuSet>) -> AsyncInput {
        let (sender, receiver) = mpsc::channel();

        thread::spawn(move || {
            match affinity {
                Some(cpu_set) => {
                    scheduler::set_self_affinity(cpu_set).expect("Failed to set affinity");
                }
                None => (),
            }

            loop {
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
            }
        });

        AsyncInput { receiver: receiver }
    }

    pub fn read(&mut self, timeout: TimeDelta) -> base::Result<Option<base::Frame>> {
        match self.receiver.recv_timeout(timeout.as_duration()) {
            Ok(PipeMessage::Frame(frame)) => Ok(Some(frame)),
            Ok(PipeMessage::Error(e)) => Err(e),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(base::Error::new("Channel closed")),
        }
    }
}
