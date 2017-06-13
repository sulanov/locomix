use std;
use std::thread;
use std::sync::mpsc;
use std::collections::VecDeque;
use input;
use base;

enum PipeMessage {
    Frame(base::Frame),
    Error(base::Error),
}

pub struct AsyncInput {
    receiver: mpsc::Receiver<PipeMessage>,
    frames_queue: VecDeque<base::Frame>,
    samples_queued: usize,
    synchronized: bool,
}

trait Sender: Send {
    fn send_message(&self,
                    t: PipeMessage)
                    -> std::result::Result<(), mpsc::SendError<PipeMessage>>;
    fn frame(&self, f: base::Frame) {
        self.send_message(PipeMessage::Frame(f)).unwrap()
    }
    fn error(&self, e: base::Error) {
        self.send_message(PipeMessage::Error(e)).unwrap()
    }
}

impl Sender for mpsc::Sender<PipeMessage> {
    fn send_message(&self,
                    m: PipeMessage)
                    -> std::result::Result<(), mpsc::SendError<PipeMessage>> {
        self.send(m)
    }
}

impl Sender for mpsc::SyncSender<PipeMessage> {
    fn send_message(&self,
                    m: PipeMessage)
                    -> std::result::Result<(), mpsc::SendError<PipeMessage>> {
        self.send(m)
    }
}


impl AsyncInput {
    pub fn new(mut input: Box<input::Input>) -> AsyncInput {

        let synchronized = input.is_synchronized();

        let (sender, receiver) = if synchronized {
            let (s, r) = mpsc::channel();
            (Box::new(s) as Box<Sender>, r)
        } else {
            let (s, r) = mpsc::sync_channel(3);
            (Box::new(s) as Box<Sender>, r)
        };

        thread::spawn(move || loop {
                          match input.read() {
                              Err(e) => {
                sender.error(e);
                break;
            }
                              Ok(Some(frame)) => sender.frame(frame),
                              Ok(None) => (),
                          }
                      });

        AsyncInput {
            receiver: receiver,
            frames_queue: VecDeque::new(),
            samples_queued: 0,
            synchronized: synchronized,
        }
    }

    fn pump_messages(&mut self) -> base::Result<()> {
        loop {
            if !self.synchronized && self.frames_queue.len() > 0 {
                return Ok(());
            }
            match self.receiver.try_recv() {
                Ok(PipeMessage::Frame(frame)) => {
                    self.samples_queued += frame.len();
                    self.frames_queue.push_back(frame);
                }
                Ok(PipeMessage::Error(e)) => return Err(e),
                Err(mpsc::TryRecvError::Empty) => return Ok(()),
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Err(base::Error::new("Input thread stopped."));
                }
            }
        }
    }
}

impl input::Input for AsyncInput {
    fn read(&mut self) -> base::Result<Option<base::Frame>> {
        try!(self.pump_messages());

        match self.frames_queue.pop_front() {
            Some(f) => {
                self.samples_queued -= f.len();
                Ok(Some(f))
            }
            None => Ok(None),
        }
    }

    fn samples_buffered(&mut self) -> base::Result<usize> {
        try!(self.pump_messages());
        Ok(self.samples_queued)
    }

    fn is_synchronized(&self) -> bool {
        self.synchronized
    }
}
