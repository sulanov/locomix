use crate::state::*;
use std;
use std::process::Command;

struct StateScript {
    script_path: String,
    shared_state: SharedState,
    state_observer: StateObserver,
}

impl StateScript {
    fn new(script_path: &str, shared_state: SharedState) -> StateScript {
        let state_observer = shared_state.lock().add_observer();
        StateScript {
            script_path: String::from(script_path),
            shared_state,
            state_observer,
        }
    }

    fn run_script(&self, state: StreamState, output: &str) {
        let result = Command::new(&self.script_path)
            .arg(state.as_str())
            .arg(output)
            .status();
        match result {
            Ok(status) => {
                if !status.success() {
                    println!(
                        "ERROR: {} {} failed with error code {}",
                        self.script_path,
                        state.as_str(),
                        status.code().unwrap_or(0)
                    );
                }
            }
            Err(e) => println!("ERROR: Failed to run {}: {}", self.script_path, e),
        }
    }

    fn run(&mut self) {
        let mut stream_state;
        let mut output_name: String;
        {
            let state = self.shared_state.lock();
            output_name = state.current_output().name.clone();
            stream_state = state.state().stream_state;
        };
        self.run_script(stream_state, output_name.as_str());

        loop {
            match self.state_observer.recv() {
                Ok(StateChange::SelectOutput { output }) => {
                    output_name = self.shared_state.lock().state().outputs[output]
                        .name
                        .clone();
                }
                Ok(StateChange::SetStreamState {
                    stream_state: new_stream_state,
                }) => {
                    stream_state = new_stream_state;
                }
                Ok(_) => continue,
                Err(_) => return,
            };

            self.run_script(stream_state, output_name.as_str());
        }
    }
}

pub fn start_state_script_contoller(script_path: &str, shared_state: SharedState) {
    let mut c = StateScript::new(script_path, shared_state);
    std::thread::spawn(move || {
        c.run();
    });
}
