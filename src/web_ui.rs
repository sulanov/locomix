use base::*;
use rouille;
use rouille::Response;
use state::*;
use std::fs::File;
use std::io::Read;
use std::thread;

fn load_file(filename: &str) -> Result<String> {
    let mut file = try!(File::open(filename));
    let mut content = String::new();
    try!(file.read_to_string(&mut content));
    Ok(content)
}

#[derive(Serialize)]
struct EmptyResponse {}

fn serve_web(address: &str, shared_state: SharedState) {
    rouille::start_server(address, move |request| {
        router!(request,
            (GET) (/) => {
                Response::html(include_str!("../web/index.html"))
            },
            (GET) (/api/state) => {
                let state_controller = shared_state.lock();
                Response::json(state_controller.state())
            },
            (POST) (/api/state) => {
                 #[derive(Deserialize)]
                 struct LoudnessParams {
                     enabled: Option<bool>,
                     auto: Option<bool>,
                     level: Option<f32>,
                     base_level_spl: Option<f32>,
                 }

                 #[derive(Deserialize)]
                 struct RequestParams {
                     volume: Option<f32>,
                     current_output: Option<usize>,
                     current_speakers: Option<usize>,
                     mux_mode: Option<MuxMode>,
                     enable_drc: Option<bool>,
                     enable_subwoofer: Option<bool>,
                     enable_crossfeed: Option<bool>,
                     bass_boost: Option<f32>,
                     loudness: Option<LoudnessParams>,
                 }

                let json: RequestParams = try_or_400!(rouille::input::json_input(request));

                let mut state_controller = shared_state.lock();

                json.volume.map( |volume| {
                  state_controller.set_volume(volume);
                });

                if let Some(output) = json.current_output {
                    if output > state_controller.state().outputs.len() {
                       return Response::text("Invalid output id").with_status_code(404)
                    }
                    if let Some(speakers) = json.current_speakers  {
                      if speakers >= state_controller.state().outputs[output].speakers.len() {
                       return Response::text("Invalid speakers id").with_status_code(404)
                      }
                    }

                    state_controller.select_output(output, json.current_speakers);
                }

                json.mux_mode.map( |mux_mode| {
                    state_controller.set_mux_mode(mux_mode);
                });

                json.enable_drc.map( |enable| {
                    state_controller.set_enable_drc(enable);
                });

                json.enable_subwoofer.map( |enable| {
                    state_controller.set_enable_subwoofer(enable);
                });

                json.enable_crossfeed.map( |enable| {
                    state_controller.set_enable_crossfeed(enable);
                });

                json.bass_boost.map( |bass_boost| {
                    state_controller.set_bass_boost(Gain { db: bass_boost});
                });

                json.loudness.map( |loudness| {
                    let mut loudness_config = state_controller.state().loudness.clone();
                    loudness.enabled.map( |enabled| {
                        loudness_config.enabled = enabled;
                    });
                    loudness.auto.map( |auto| {
                        loudness_config.auto = auto;
                    });
                    loudness.level.map( |level| {
                        loudness_config.level = level;
                    });
                    loudness.base_level_spl.map( |base_level_spl| {
                        loudness_config.base_level_spl = base_level_spl;
                    });
                    state_controller.set_loudness(loudness_config);
                });

                Response::json(&EmptyResponse{})
            },
            (POST) (/api/inputs/{id: usize}) => {
                 #[derive(Deserialize)]
                 struct RequestParams {
                     gain: f32,
                 }

                let json: RequestParams = try_or_400!(rouille::input::json_input(request));
                let gain = Gain { db: json.gain };

                let mut state_controller = shared_state.lock();
                if id >= state_controller.state().inputs.len() {
                    return Response::text("Invalid input id").with_status_code(404)
                }

                state_controller.set_input_gain(id, gain);

                Response::json(&EmptyResponse{})
            },
            (GET) (/status) => {
                let temp_str = match load_file("/sys/class/thermal/thermal_zone0/temp") {
                    Ok(s) => s,
                    Err(error) => {
                        println!("Failed to get temp: {}", error);
                        return Response::text("Internal Server Error").with_status_code(500)
                    }
                };
                let temp = match temp_str.trim().parse::<usize>() {
                    Ok(temp) => temp,
                    Err(error) => {
                        println!("Failed to parse temp: {}", error);
                        return Response::text("Internal Server Error").with_status_code(500)
                    }
                };
                Response::html(format!(r#"
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1">
        </meta>
        <body>
        CPU temperature: <b>{:.1}</b>°C
        </body>
        </html>"#, temp as f32 / 1000.0))
            },
            _ => Response::empty_404()
        )
    });
}

pub fn start_web(address: &str, shared_state: SharedState) {
    let address_copy = String::from(address);
    thread::spawn(move || {
        serve_web(&address_copy, shared_state);
    });
}
