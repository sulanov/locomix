use base::*;
use rouille;
use rouille::Response;
use ui::*;
use std::fs::File;
use std::io::Read;
use std::thread;

fn load_file(filename: &str) -> Result<String> {
    let mut file = try!(File::open(filename));
    let mut content = String::new();
    try!(file.read_to_string(&mut content));
    Ok(content)
}

#[derive(RustcEncodable)]
struct EmptyResponse {}

fn serve_web(address: &str, shared_state: SharedState) {
    rouille::start_server(address, move |request| {
        router!(request,
            (GET) (/) => {
                match load_file("web/index.html") {
                    Ok(html) => Response::html(html),
                    Err(error) => {
                        println!("Can't load index.html: {}", error);
                        Response::text("Internal Server Error").with_status_code(500)
                    }
                }
            },
            (GET) (/api/state) => {
                let state_controller = shared_state.lock();
                Response::json(state_controller.state())
            },
            (POST) (/api/state) => {
                 #[derive(RustcDecodable)]
                 struct LoudnessParams {
                     enabled: Option<bool>,
                     auto: Option<bool>,
                     level: Option<f32>,
                 }

                 #[derive(RustcDecodable)]
                 struct CrossfeedParams {
                     enabled: Option<bool>,
                     level: Option<f32>,
                     delay_ms: Option<f32>,
                 }

                 #[derive(RustcDecodable)]
                 struct FeatureParams {
                     enabled: Option<bool>,
                     level: Option<f32>,
                 }

                 #[derive(RustcDecodable)]
                 struct RequestParams {
                     volume: Option<f32>,
                     output: Option<usize>,
                     mux_mode: Option<MuxMode>,
                     enable_drc: Option<bool>,
                     loudness: Option<LoudnessParams>,
                     voice_boost: Option<FeatureParams>,
                     crossfeed: Option<CrossfeedParams>,
                 }

                let json: RequestParams = try_or_400!(rouille::input::json_input(request));

                let mut state_controller = shared_state.lock();

                json.volume.map( |volume| {
                  state_controller.set_volume(Gain{db: volume});
                });

                match json.output {
                    Some(output) => {
                          if output > state_controller.state().outputs.len() {
                            return Response::text("Invalid output id").with_status_code(404)
                          }
                          state_controller.select_output(output);
                        }
                    None => ()
                };

                json.mux_mode.map( |mux_mode| {
                    state_controller.set_mux_mode(mux_mode);
                });

                json.enable_drc.map( |enable_drc| {
                    state_controller.set_enable_drc(enable_drc);
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
                    state_controller.set_loudness(loudness_config);
                });

                json.voice_boost.map( |voice_boost| {
                    let mut voice_boost_config = state_controller.state().voice_boost.clone();
                    voice_boost.enabled.map( |enabled| {
                        voice_boost_config.enabled = enabled;
                    });
                    voice_boost.level.map( |level| {
                        voice_boost_config.level = level;
                    });
                    state_controller.set_voice_boost(voice_boost_config);
                });

                json.crossfeed.map( |crossfeed| {
                    let mut crossfeed_config = state_controller.state().crossfeed.clone();
                    crossfeed.enabled.map( |enabled| {
                        crossfeed_config.enabled = enabled;
                    });
                    crossfeed.level.map( |level| {
                        crossfeed_config.level = level;
                    });
                    crossfeed.delay_ms.map( |delay_ms| {
                        crossfeed_config.delay_ms = delay_ms;
                    });
                    state_controller.set_crossfeed(crossfeed_config);
                });

                Response::json(&EmptyResponse{})
            },
            (POST) (/api/inputs/{id: usize}) => {
                 #[derive(RustcDecodable)]
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
    thread::spawn(move || { serve_web(&address_copy, shared_state); });
}
