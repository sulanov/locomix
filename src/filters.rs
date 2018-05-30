use base::*;
use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use time;

pub type FCoef = f32;

pub trait AudioFilter<T> {
    type Params: Clone + Send;
    fn new(params: Self::Params) -> T;
    fn set_params(&mut self, params: Self::Params);
    fn apply_one(&mut self, sample: FCoef) -> FCoef;
    fn apply_multi(&mut self, buffer: &mut [f32]) {
        for i in 0..buffer.len() {
            buffer[i] = self.apply_one(buffer[i] as FCoef) as f32;
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

    pub fn low_shelf_filter(
        sample_rate: FCoef,
        f0: FCoef,
        slope: FCoef,
        gain_db: FCoef,
    ) -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let a = (10.0 as FCoef).powf(gain_db / 40.0);
        let a_sqrt = a.sqrt();
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let alpha_a = 2.0 * a_sqrt * alpha;

        BiquadParams::new(
            a * ((a + 1.0) - (a - 1.0) * w0_cos + alpha_a),
            2.0 * a * ((a - 1.0) - (a + 1.0) * w0_cos),
            a * ((a + 1.0) - (a - 1.0) * w0_cos - alpha_a),
            (a + 1.0) + (a - 1.0) * w0_cos + alpha_a,
            -2.0 * ((a - 1.0) + (a + 1.0) * w0_cos),
            (a + 1.0) + (a - 1.0) * w0_cos - alpha_a,
        )
    }

    pub fn high_shelf_filter(
        sample_rate: FCoef,
        f0: FCoef,
        slope: FCoef,
        gain_db: FCoef,
    ) -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let a = (10.0 as FCoef).powf(gain_db / 40.0);
        let a_sqrt = a.sqrt();
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let alpha_a = 2.0 * a_sqrt * alpha;

        BiquadParams::new(
            a * ((a + 1.0) + (a - 1.0) * w0_cos + alpha_a),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * w0_cos),
            a * ((a + 1.0) + (a - 1.0) * w0_cos - alpha_a),
            (a + 1.0) - (a - 1.0) * w0_cos + alpha_a,
            2.0 * ((a - 1.0) - (a + 1.0) * w0_cos),
            (a + 1.0) - (a - 1.0) * w0_cos - alpha_a,
        )
    }

    pub fn low_pass_filter(sample_rate: FCoef, f0: FCoef, slope: FCoef) -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let alpha = w0.sin() / (2.0 * slope);
        BiquadParams::new(
            (1.0 - w0_cos) / 2.0,
            1.0 - w0_cos,
            (1.0 - w0_cos) / 2.0,
            1.0 + alpha,
            -2.0 * w0_cos,
            1.0 - alpha,
        )
    }

    pub fn high_pass_filter(sample_rate: FCoef, f0: FCoef, slope: FCoef) -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let alpha = w0.sin() / (2.0 * slope);
        BiquadParams::new(
            (1.0 + w0_cos) / 2.0,
            -(1.0 + w0_cos),
            (1.0 + w0_cos) / 2.0,
            1.0 + alpha,
            -2.0 * w0_cos,
            1.0 - alpha,
        )
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
        let z0 = self.p.b0 * x0 + self.p.b1 * self.x1 + self.p.b2 * self.x2 - self.p.a1 * self.z1
            - self.p.a2 * self.z2;
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
    pub fn new(sample_rate: f64, gain_db: f32) -> SimpleFilterParams {
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
        BiquadParams::low_shelf_filter(
            self.0.sample_rate,
            LOUDNESS_BASS_F0,
            LOUDNESS_BASS_Q,
            self.0.gain_db,
        )
    }
    fn get_filter2_params(&self) -> BiquadParams {
        const LOUDNESS_TREBLE_F0: FCoef = 14000.0;
        const LOUDNESS_TREBLE_Q: FCoef = 1.0;
        BiquadParams::high_shelf_filter(
            self.0.sample_rate,
            LOUDNESS_TREBLE_F0,
            LOUDNESS_TREBLE_Q,
            self.0.gain_db / 2.0,
        )
    }
}

pub type LoudnessFilter = FilterPair<LoudnessFilterPairConfig>;

pub trait StreamFilter: Send {
    fn apply(&mut self, frame: Frame) -> Frame;
    fn reset(&mut self);
}

pub struct MultichannelFilter<T: AudioFilter<T>> {
    params: T::Params,
    filters: PerChannel<T>,
}

impl<T: AudioFilter<T>> MultichannelFilter<T> {
    pub fn new(params: T::Params) -> MultichannelFilter<T> {
        MultichannelFilter {
            params: params,
            filters: PerChannel::new(),
        }
    }

    pub fn set_params(&mut self, params: T::Params) {
        for (_, f) in self.filters.iter() {
            f.set_params(params.clone());
        }
        self.params = params
    }
}

impl<T: AudioFilter<T> + Send> StreamFilter for MultichannelFilter<T> {
    fn apply(&mut self, mut frame: Frame) -> Frame {
        for (c, pcm) in frame.iter_channels() {
            if !self.filters.have_channel(c) {
                self.filters.set(c, T::new(self.params.clone()))
            }

            self.filters.get_mut(c).unwrap().apply_multi(pcm);
        }
        frame
    }

    fn reset(&mut self) {
        self.filters.clear();
    }
}

enum FilterJob {
    Apply {
        data: Vec<f32>,
        size_multiplier: usize,
        response_channel: mpsc::Sender<Vec<f32>>,
    },
    Reset,
}

pub struct MultichannelFirFilter {
    threads: PerChannel<mpsc::Sender<FilterJob>>,
    size_multiplier: usize,
    delay1: f64,
    delay2: f64,
}

impl MultichannelFirFilter {
    pub fn new_pair(left: FirFilterParams, right: FirFilterParams) -> MultichannelFirFilter {
        let mut params = PerChannel::new();
        params.set(ChannelPos::FL, left);
        params.set(ChannelPos::FR, right);
        return MultichannelFirFilter::new(params);
    }

    pub fn new(mut params: PerChannel<FirFilterParams>) -> MultichannelFirFilter {
        let mut threads = PerChannel::new();
        for (channel, channel_params) in params.iter() {
            let (job_sender, job_receiver) = mpsc::channel::<FilterJob>();
            let params = channel_params.clone();
            threads.set(channel, job_sender);
            thread::spawn(move || {
                let mut filter = FirFilter::new(params);
                for mut job in job_receiver.iter() {
                    match job {
                        FilterJob::Apply {
                            mut data,
                            size_multiplier,
                            response_channel,
                        } => {
                            filter.window_size = filter.buffer.len() * size_multiplier / 256;
                            filter.apply_multi(&mut data[..]);
                            response_channel.send(data).expect("Failed to send");
                        }
                        FilterJob::Reset => {
                            filter.reset();
                        }
                    }
                }
            });
        }

        MultichannelFirFilter {
            threads: threads,
            size_multiplier: 256,
            delay1: 0.0,
            delay2: 0.0,
        }
    }
}

impl StreamFilter for MultichannelFirFilter {
    fn apply(&mut self, mut frame: Frame) -> Frame {
        let start = time::Time::now();

        let mut recv_channels = vec![];

        for (c, t) in self.threads.iter() {
            let (s, r) = mpsc::channel::<Vec<f32>>();
            recv_channels.push(r);
            frame.ensure_channel(c);
            t.send(FilterJob::Apply {
                data: frame.take_channel(c).unwrap(),
                response_channel: s,
                size_multiplier: self.size_multiplier,
            }).expect("Failed to send");
        }

        for (i, (c, _t)) in self.threads.iter().enumerate() {
            let mut v = recv_channels[i].recv().expect("Failed to apply filter");
            frame.set_channel(c, v);
        }

        let now = time::Time::now();
        let delay = (now - start).in_seconds_f() / frame.duration().in_seconds_f();
        let mean_delay = (self.delay1 + self.delay2 + delay) / 3.0;
        let m: f64 = if mean_delay > 0.9 && self.size_multiplier > 1 {
            0.5
        } else if mean_delay < 0.4 && self.size_multiplier < 256 {
            2.0
        } else {
            1.0
        };
        self.delay1 = self.delay2;
        self.delay2 = delay;

        if m != 1.0 {
            self.size_multiplier = self.size_multiplier * (2.0 * m) as usize / 2;
            self.delay1 *= m;
            self.delay2 *= m;
            println!(
                "Updated FIR filter size to {}, delay: {:?} ",
                self.size_multiplier as f32 / 256.0,
                mean_delay
            );
        }

        frame
    }

    fn reset(&mut self) {
        for (_c, t) in self.threads.iter() {
            t.send(FilterJob::Reset)
                .expect("Failed to send reset command");
        }
    }
}

#[derive(Clone)]
pub struct FirFilterParams {
    coefficients: Arc<Vec<FCoef>>,
}

impl FirFilterParams {
    pub fn new(fir: Vec<f32>, size: usize) -> FirFilterParams {
        FirFilterParams {
            coefficients: Arc::new(fir[0..size].to_vec()),
        }
    }
}

pub struct FirFilter {
    params: FirFilterParams,

    // Circular buffer for input samples.
    buffer: Vec<f32>,

    // Current position in the buffer.
    buffer_pos: usize,

    window_size: usize,
}

impl FirFilter {
    fn reset(&mut self) {
        for v in &mut self.buffer {
            *v = 0.0;
        }
    }
}

impl AudioFilter<FirFilter> for FirFilter {
    type Params = FirFilterParams;

    fn new(params: FirFilterParams) -> FirFilter {
        let size = params.coefficients.len();
        FirFilter {
            params: params,
            buffer: vec![0.0; size],
            buffer_pos: 0,
            window_size: size,
        }
    }

    fn set_params(&mut self, params: FirFilterParams) {
        let size = params.coefficients.len();
        self.params = params;
        self.buffer = vec![0.0; size];
        self.buffer_pos = 0;
        self.window_size = size;
    }

    fn apply_one(&mut self, x0: FCoef) -> FCoef {
        self.buffer_pos = (self.buffer_pos + self.buffer.len() - 1) % self.buffer.len();
        self.buffer[self.buffer_pos] = x0;

        let p1_size = self.buffer.len() - self.buffer_pos;
        if self.window_size <= p1_size {
            convolve(
                &self.buffer[self.buffer_pos..(self.buffer_pos + self.window_size)],
                &self.params.coefficients[0..self.window_size],
            )
        } else {
            convolve(
                &self.buffer[self.buffer_pos..],
                &self.params.coefficients[0..p1_size],
            )
                + convolve(
                    &self.buffer[0..(self.window_size - p1_size)],
                    &self.params.coefficients[p1_size..self.window_size],
                )
        }
    }
}

pub struct CrossfeedFilter {
    level: f32,
    delay_ms: f32,
    previous: Frame,
    left_cross_filter: BiquadFilter,
    left_straight_filter: BiquadFilter,
    right_cross_filter: BiquadFilter,
    right_straight_filter: BiquadFilter,
}

impl CrossfeedFilter {
    pub fn new(sample_rate: f64) -> CrossfeedFilter {
        CrossfeedFilter {
            level: 0.0,
            delay_ms: 0.3,
            previous: Frame::new(1.0, time::Time::now(), 0),
            left_straight_filter: BiquadFilter::new(BiquadParams::low_shelf_filter(
                sample_rate as FCoef,
                200.0,
                1.1,
                -2.5,
            )),
            left_cross_filter: BiquadFilter::new(BiquadParams::low_pass_filter(
                sample_rate as FCoef,
                400.0,
                0.8,
            )),
            right_straight_filter: BiquadFilter::new(BiquadParams::low_shelf_filter(
                sample_rate as FCoef,
                200.0,
                1.1,
                -2.5,
            )),
            right_cross_filter: BiquadFilter::new(BiquadParams::low_pass_filter(
                sample_rate as FCoef,
                400.0,
                0.8,
            )),
        }
    }

    pub fn set_params(&mut self, level: f32, delay_ms: f32) {
        self.level = level;
        self.delay_ms = delay_ms;
    }

    fn crossfeed(
        out: &mut [f32],
        inp: &[f32],
        other: &[f32],
        prev: &[f32],
        delay: usize,
        level: f32,
        straight_filter: &mut BiquadFilter,
        cross_filter: &mut BiquadFilter,
    ) {
        for i in 0..out.len() {
            out[i] = straight_filter.apply_one(inp[i])
        }

        let mut out_pos = 0;
        if prev.len() < delay {
            out_pos = delay - prev.len()
        }

        while out_pos < out.len() && out_pos < delay {
            out[out_pos] += cross_filter.apply_one(prev[prev.len() + out_pos - delay]) * level;
            out_pos += 1;
        }

        while out_pos < out.len() {
            out[out_pos] += cross_filter.apply_one(other[out_pos - delay]) * level;
            out_pos += 1;
        }
    }

    pub fn apply(&mut self, mut frame: Frame) -> Frame {
        if self.level == 0.0 {
            return frame;
        }
        let mut out = Frame::new(frame.sample_rate, frame.timestamp, frame.len());

        let delay = (self.delay_ms * frame.sample_rate as f32 / 1000.0) as usize;

        frame.ensure_channel(ChannelPos::FL);
        frame.ensure_channel(ChannelPos::FR);

        CrossfeedFilter::crossfeed(
            out.ensure_channel(ChannelPos::FL),
            frame.get_channel(ChannelPos::FL).unwrap(),
            frame.get_channel(ChannelPos::FR).unwrap(),
            self.previous.ensure_channel(ChannelPos::FR),
            delay,
            self.level,
            &mut self.left_straight_filter,
            &mut self.left_cross_filter,
        );

        CrossfeedFilter::crossfeed(
            out.ensure_channel(ChannelPos::FR),
            frame.get_channel(ChannelPos::FR).unwrap(),
            frame.get_channel(ChannelPos::FL).unwrap(),
            self.previous.ensure_channel(ChannelPos::FL),
            delay,
            self.level,
            &mut self.right_straight_filter,
            &mut self.right_cross_filter,
        );

        self.previous = frame;

        out
    }
}

pub struct CascadingFilter {
    filter_1: BiquadFilter,
    filter_2: BiquadFilter,
}

impl AudioFilter<CascadingFilter> for CascadingFilter {
    type Params = BiquadParams;

    fn new(params: BiquadParams) -> CascadingFilter {
        CascadingFilter {
            filter_1: BiquadFilter::new(params.clone()),
            filter_2: BiquadFilter::new(params),
        }
    }

    fn set_params(&mut self, params: Self::Params) {
        self.filter_1.set_params(params.clone());
        self.filter_2.set_params(params.clone());
    }

    fn apply_one(&mut self, sample: FCoef) -> FCoef {
        self.filter_2.apply_one(self.filter_1.apply_one(sample))
    }
}
