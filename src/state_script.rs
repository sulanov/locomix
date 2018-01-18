use std;
use std::process::Command;
use ui::*;

struct StateScript {
    script_path: String,
    state_receiver: StreamStateReceiver,
}

impl StateScript {
    fn new(script_path: &str, state: SharedState) -> StateScript {
        let receiver = state.lock().add_stream_observer();
        StateScript {
            script_path: String::from(script_path),
            state_receiver: receiver,
        }
    }

    fn run(&mut self) {
        loop {
            let state = match self.state_receiver.recv() {
                Ok(s) => s,
                Err(_) => return,
            };

            let result = Command::new(&self.script_path).arg(state.as_str()).status();
            match result {
                Ok(status) => if !status.success() {
                    println!(
                        "ERROR: {} {} failed with error code {}",
                        self.script_path,
                        state.as_str(),
                        status.code().unwrap_or(0)
                    );
                },
                Err(e) => println!("ERROR: Failed to run {}: {}", self.script_path, e),
            }
        }
    }
}

pub fn start_state_script_contoller(script_path: &str, shared_state: SharedState) {
    let mut c = StateScript::new(script_path, shared_state);
    std::thread::spawn(move || {
        c.run();
    });
}
