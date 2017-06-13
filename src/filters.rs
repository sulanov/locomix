use base::*;
use std::time::Instant;
use std::f32::consts::PI;

type FCoef = f32;

pub trait AudioFilter<T> {
    type Params: Copy;
    fn new(params: Self::Params) -> T;
    fn set_params(&mut self, params: Self::Params);
    fn apply_one(&mut self, sample: FCoef) -> FCoef;
    fn apply_multi(&mut self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() == output.len());
        for i in 0..input.len() {
            output[i] = self.apply_one(input[i] as FCoef) as f32;
        }
    }
}

#[derive(Copy, Clone)]
pub struct BiquadParams {
    b0: FCoef,
    b1: FCoef,
    b2: FCoef,
    a1: FCoef,
    a2: FCoef,
}

impl BiquadParams {
    pub fn new(b0: FCoef, b1: FCoef, b2: FCoef, a0: FCoef, a1: FCoef, a2: FCoef) -> BiquadParams {
        BiquadParams {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    pub fn low_shelf_filter(sample_rate: FCoef,
                            f0: FCoef,
                            slope: FCoef,
                            gain_db: FCoef)
                            -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let a = (10.0 as FCoef).powf(gain_db / 40.0);
        let a_sqrt = a.sqrt();
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let alpha_a = 2.0 * a_sqrt * alpha;

        BiquadParams::new(a * ((a + 1.0) - (a - 1.0) * w0_cos + alpha_a),
                          2.0 * a * ((a - 1.0) - (a + 1.0) * w0_cos),
                          a * ((a + 1.0) - (a - 1.0) * w0_cos - alpha_a),
                          (a + 1.0) + (a - 1.0) * w0_cos + alpha_a,
                          -2.0 * ((a - 1.0) + (a + 1.0) * w0_cos),
                          (a + 1.0) + (a - 1.0) * w0_cos - alpha_a)
    }

    pub fn high_shelf_filter(sample_rate: FCoef,
                             f0: FCoef,
                             slope: FCoef,
                             gain_db: FCoef)
                             -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let a = (10.0 as FCoef).powf(gain_db / 40.0);
        let a_sqrt = a.sqrt();
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let alpha_a = 2.0 * a_sqrt * alpha;

        BiquadParams::new(a * ((a + 1.0) + (a - 1.0) * w0_cos + alpha_a),
                          -2.0 * a * ((a - 1.0) + (a + 1.0) * w0_cos),
                          a * ((a + 1.0) + (a - 1.0) * w0_cos - alpha_a),
                          (a + 1.0) - (a - 1.0) * w0_cos + alpha_a,
                          2.0 * ((a - 1.0) - (a + 1.0) * w0_cos),
                          (a + 1.0) - (a - 1.0) * w0_cos - alpha_a)
    }
}

pub struct BiquadFilter {
    p: BiquadParams,

    x1: FCoef,
    x2: FCoef,
    z1: FCoef,
    z2: FCoef,
}

impl AudioFilter<BiquadFilter> for BiquadFilter {
    type Params = BiquadParams;

    fn new(params: BiquadParams) -> BiquadFilter {
        BiquadFilter {
            p: params,
            x1: 0.0,
            x2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn set_params(&mut self, params: BiquadParams) {
        self.p = params;
    }

    fn apply_one(&mut self, x0: FCoef) -> FCoef {
        let z0 = self.p.b0 * x0 + self.p.b1 * self.x1 + self.p.b2 * self.x2 -
                 self.p.a1 * self.z1 - self.p.a2 * self.z2;
        self.z2 = self.z1;
        self.z1 = z0;
        self.x2 = self.x1;
        self.x1 = x0;
        z0
    }
}

#[derive(Copy, Clone)]
pub struct SimpleFilterParams {
    sample_rate: FCoef,
    gain_db: FCoef,
}

impl SimpleFilterParams {
    pub fn new(sample_rate: usize, gain_db: f32) -> SimpleFilterParams {
        SimpleFilterParams {
            sample_rate: sample_rate as FCoef,
            gain_db: gain_db as FCoef,
        }
    }
}

pub trait FilterPairConfig {
    type Impl: FilterPairConfig;
    fn new(p: SimpleFilterParams) -> Self::Impl;
    fn set_params(&mut self, p: SimpleFilterParams);
    fn get_filter1_params(&self) -> BiquadParams;
    fn get_filter2_params(&self) -> BiquadParams;
}

pub struct FilterPair<C: FilterPairConfig> {
    filter1: BiquadFilter,
    filter2: BiquadFilter,
    config: C::Impl,
}

impl<C: FilterPairConfig> AudioFilter<FilterPair<C>> for FilterPair<C> {
    type Params = SimpleFilterParams;

    fn new(params: Self::Params) -> FilterPair<C> {
        let mut config = C::new(params);
        config.set_params(params);
        FilterPair::<C> {
            filter1: BiquadFilter::new(config.get_filter1_params()),
            filter2: BiquadFilter::new(config.get_filter2_params()),
            config: config,
        }
    }

    fn set_params(&mut self, params: SimpleFilterParams) {
        self.config.set_params(params);
        self.filter1.set_params(self.config.get_filter1_params());
        self.filter2.set_params(self.config.get_filter2_params());
    }

    fn apply_one(&mut self, sample: FCoef) -> FCoef {
        self.filter2.apply_one(self.filter1.apply_one(sample))
    }
}

pub struct LoudnessFilterPairConfig(SimpleFilterParams);

impl FilterPairConfig for LoudnessFilterPairConfig {
    type Impl = LoudnessFilterPairConfig;

    fn new(p: SimpleFilterParams) -> LoudnessFilterPairConfig {
        LoudnessFilterPairConfig(p)
    }
    fn set_params(&mut self, p: SimpleFilterParams) {
        self.0 = p;
    }

    fn get_filter1_params(&self) -> BiquadParams {
        const LOUDNESS_BASS_F0: FCoef = 100.0;
        const LOUDNESS_BASS_Q: FCoef = 0.25;
        BiquadParams::low_shelf_filter(self.0.sample_rate,
                                       LOUDNESS_BASS_F0,
                                       LOUDNESS_BASS_Q,
                                       self.0.gain_db)
    }
    fn get_filter2_params(&self) -> BiquadParams {
        const LOUDNESS_TREBLE_F0: FCoef = 14000.0;
        const LOUDNESS_TREBLE_Q: FCoef = 1.0;
        BiquadParams::high_shelf_filter(self.0.sample_rate,
                                        LOUDNESS_TREBLE_F0,
                                        LOUDNESS_TREBLE_Q,
                                        self.0.gain_db / 2.0)
    }
}
pub type LoudnessFilter = FilterPair<LoudnessFilterPairConfig>;

pub struct VoiceBoostFilterPairConfig(SimpleFilterParams);

impl FilterPairConfig for VoiceBoostFilterPairConfig {
    type Impl = VoiceBoostFilterPairConfig;

    fn new(p: SimpleFilterParams) -> VoiceBoostFilterPairConfig {
        VoiceBoostFilterPairConfig(p)
    }
    fn set_params(&mut self, p: SimpleFilterParams) {
        self.0 = p;
    }
    fn get_filter1_params(&self) -> BiquadParams {
        const VOICE_BOOST_BASS_F0: FCoef = 270.0;
        const VOICE_BOOST_BASS_Q: FCoef = 2.0;
        BiquadParams::low_shelf_filter(self.0.sample_rate,
                                       VOICE_BOOST_BASS_F0,
                                       VOICE_BOOST_BASS_Q,
                                       -self.0.gain_db)
    }
    fn get_filter2_params(&self) -> BiquadParams {
        const VOICE_BOOST_TREBLE_F0: FCoef = 3300.0;
        const VOICE_BOOST_TREBLE_Q: FCoef = 2.0;
        BiquadParams::high_shelf_filter(self.0.sample_rate,
                                        VOICE_BOOST_TREBLE_F0,
                                        VOICE_BOOST_TREBLE_Q,
                                        -self.0.gain_db)
    }
}
pub type VoiceBoostFilter = FilterPair<VoiceBoostFilterPairConfig>;

pub struct StereoFilter<T: AudioFilter<T>> {
    left: T,
    right: T,
}

impl<T: AudioFilter<T>> StereoFilter<T> {
    pub fn new(params: T::Params) -> StereoFilter<T> {
        StereoFilter {
            left: T::new(params),
            right: T::new(params),
        }
    }

    pub fn set_params(&mut self, params: T::Params) {
        self.left.set_params(params);
        self.right.set_params(params);
    }

    pub fn apply(&mut self, frame: &Frame) -> Frame {
        let mut out = Frame::new(frame.sample_rate, frame.len());
        self.left.apply_multi(&frame.left[..], &mut out.left[..]);
        self.right
            .apply_multi(&frame.right[..], &mut out.right[..]);
        out
    }
}

pub struct CrossfeedFilter {
    level: f32,
    delay_ms: f32,
    previous: Option<Frame>,
}

impl CrossfeedFilter {
    pub fn new() -> CrossfeedFilter {
        CrossfeedFilter {
            level: 0.0,
            delay_ms: 0.5,
            previous: None,
        }
    }

    pub fn set_params(&mut self, level: f32, delay_ms: f32) {
        self.level = level;
        self.delay_ms = delay_ms;
    }

    pub fn apply(&mut self, frame: Frame) -> Frame {
        if self.level == 0.0 {
            return frame;
        }
        let mut out = Frame::new(frame.sample_rate, frame.len());
        let b = self.level * 0.5;
        let a = 1.0 - b;

        let delay = (self.delay_ms * frame.sample_rate as f32 / 1000.0) as usize;
        assert!(delay < out.len());

        match self.previous.as_ref() {
            Some(ref p) => {
                let s = p.len() - delay;
                for i in 0..delay {
                    out.left[i] = frame.left[i] * a + p.right[s + i] * b;
                    out.right[i] = frame.right[i] * a + p.left[s + i] * b;
                }
            }
            None => {
                for i in 0..delay {
                    out.left[i] = frame.left[i] * a;
                    out.right[i] = frame.right[i] * a;
                }
            }
        }

        for i in delay..out.len() {
            out.left[i] = frame.left[i] * a + frame.right[i - delay] * b;
            out.right[i] = frame.right[i] * a + frame.left[i - delay] * b;
        }

        self.previous = Some(frame);

        out
    }
}

fn get_filter_response<T: AudioFilter<T>>(p: T::Params, sample_rate: FCoef, freq: FCoef) -> FCoef {
    let mut f = T::new(p);
    let mut test_signal = vec![0.0; 2 * sample_rate as usize];
    for i in 0..test_signal.len() {
        test_signal[i] = (i as FCoef / sample_rate * freq * 2.0 * PI).sin() as f32;
    }
    let mut result = vec![0.0; test_signal.len()];
    f.apply_multi(&test_signal, &mut result);

    let t = Instant::now();

    let mut p_sum = 0.0;
    for i in 0..result.len() {
        p_sum += (result[i] as FCoef).powi(2);
    }

    let d = t.elapsed();
    println!("{} ", d.subsec_nanos() as f32 / 1000.0);

    ((p_sum / (result.len() as FCoef)) * (2 as FCoef)).log(10.0) * 10.0
}

pub fn draw_filter_graph<T: AudioFilter<T>>(params: T::Params) {
    let sample_rate: FCoef = 88100.0;
    let mut freq: FCoef = 20.0;
    for _ in 0..82 {
        let response = get_filter_response::<T>(params.clone(), sample_rate, freq);
        println!("{} {}", freq as usize, response);
        freq = freq * (2 as FCoef).powf(0.125);
    }
}
