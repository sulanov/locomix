use std;
use std::process::Command;
use ui::*;

struct StateScript {
    script_path: String,
    state: SharedState,
    state_observer: StateObserver,
}

impl StateScript {
    fn new(script_path: &str, state: SharedState) -> StateScript {
        let receiver = state.lock().add_observer();
        StateScript {
            script_path: String::from(script_path),
            state: state,
            state_observer: receiver,
        }
    }

    fn run_script(&self, state: StreamState, output: &str) {
        let result = Command::new(&self.script_path)
            .arg(state.as_str())
            .arg(output)
            .status();
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

    fn run(&mut self) {
        let mut state;
        let mut output: String;
        {
            let l = self.state.lock();
            let s = l.state();
            output = s.outputs[s.output].name.clone();
            state = s.stream_state;
        };
        self.run_script(state, output.as_str());

        loop {
            match self.state_observer.recv() {
                Ok(StateChange::SelectOutput { output: o }) => {
                    output = self.state.lock().state().outputs[o].name.clone();
                }
                Ok(StateChange::SetStreamState { stream_state }) => {
                    state = stream_state;
                }
                Ok(_) => continue,
                Err(_) => return,
            };

            self.run_script(state, output.as_str());
        }
    }
}

pub fn start_state_script_contoller(script_path: &str, shared_state: SharedState) {
    let mut c = StateScript::new(script_path, shared_state);
    std::thread::spawn(move || {
        c.run();
    });
}
