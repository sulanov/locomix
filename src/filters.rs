use crate::base::*;
use crate::time;
use std;
use std::f64::consts::PI;
use std::fs;
use std::sync::Arc;

pub type BqCoef = f64;
pub type FirCoef = f32;

pub trait AudioFilter: Send {
    fn apply_one(&mut self, sample: f32) -> f32;
    fn apply_multi(&mut self, buffer: &mut [f32]) {
        for i in 0..buffer.len() {
            buffer[i] = self.apply_one(buffer[i]);
        }
    }
    fn reset(&mut self);
}

pub trait AudioFilterWithParams<T> {
    type Params: Clone + Send;
    fn new(params: &Self::Params) -> T;
    fn set_params(&mut self, params: &Self::Params);
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BiquadParams {
    b0: BqCoef,
    b1: BqCoef,
    b2: BqCoef,
    a1: BqCoef,
    a2: BqCoef,
}

impl BiquadParams {
    pub fn new(
        b0: BqCoef,
        b1: BqCoef,
        b2: BqCoef,
        a0: BqCoef,
        a1: BqCoef,
        a2: BqCoef,
    ) -> BiquadParams {
        BiquadParams {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    pub fn low_shelf_filter(
        sample_rate: BqCoef,
        f0: BqCoef,
        slope: BqCoef,
        gain_db: BqCoef,
    ) -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let a = (10.0 as BqCoef).powf(gain_db / 40.0);
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
        sample_rate: BqCoef,
        f0: BqCoef,
        slope: BqCoef,
        gain_db: BqCoef,
    ) -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let a = (10.0 as BqCoef).powf(gain_db / 40.0);
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

    pub fn low_pass_filter(sample_rate: BqCoef, f0: BqCoef, slope: BqCoef) -> BiquadParams {
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

    pub fn high_pass_filter(sample_rate: BqCoef, f0: BqCoef, slope: BqCoef) -> BiquadParams {
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

    pub fn peaking_filter(
        sample_rate: BqCoef,
        f0: BqCoef,
        slope: BqCoef,
        gain_db: BqCoef,
    ) -> BiquadParams {
        let w0 = 2.0 * PI * f0 / sample_rate;
        let w0_cos = w0.cos();
        let a = (10.0 as BqCoef).powf(gain_db / 40.0);
        let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();

        BiquadParams::new(
            1.0 + alpha * a,
            -2.0 * w0_cos,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * w0_cos,
            1.0 - alpha / a,
        )
    }

    pub fn identity_filter() -> BiquadParams {
        BiquadParams::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    }

    pub fn is_identity(&self) -> bool {
        self.b0 == 1.0 && self.b1 == 0.0 && self.b2 == 0.0 && self.a1 == 0.0 && self.a2 == 0.0
    }
}

#[derive(Debug)]
pub struct BiquadFilter {
    p: BiquadParams,

    x1: BqCoef,
    x2: BqCoef,
    z1: BqCoef,
    z2: BqCoef,
}

impl AudioFilterWithParams<BiquadFilter> for BiquadFilter {
    type Params = BiquadParams;

    fn new(params: &BiquadParams) -> BiquadFilter {
        BiquadFilter {
            p: *params,
            x1: 0.0,
            x2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn set_params(&mut self, params: &BiquadParams) {
        self.p = *params;
    }
}

impl AudioFilter for BiquadFilter {
    fn apply_one(&mut self, x0: f32) -> f32 {
        let x0 = x0 as BqCoef;
        let z0 = self.p.b0 * x0 + self.p.b1 * self.x1 + self.p.b2 * self.x2
            - self.p.a1 * self.z1
            - self.p.a2 * self.z2;
        self.z2 = self.z1;
        self.z1 = z0;
        self.x2 = self.x1;
        self.x1 = x0;
        z0 as f32
    }
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

#[derive(Copy, Clone)]
pub struct SimpleFilterParams {
    sample_rate: BqCoef,
    gain_db: BqCoef,
}

impl SimpleFilterParams {
    pub fn new(sample_rate: f64, gain_db: f32) -> SimpleFilterParams {
        SimpleFilterParams {
            sample_rate: sample_rate as BqCoef,
            gain_db: gain_db as BqCoef,
        }
    }
}

pub trait FilterPairConfig: Send {
    type Impl: FilterPairConfig;
    fn new(p: &SimpleFilterParams) -> Self::Impl;
    fn set_params(&mut self, p: &SimpleFilterParams);
    fn get_filter1_params(&self) -> BiquadParams;
    fn get_filter2_params(&self) -> BiquadParams;
}

pub struct FilterPair<C: FilterPairConfig> {
    filter1: BiquadFilter,
    filter2: BiquadFilter,
    config: C::Impl,
}

impl<C: FilterPairConfig> AudioFilterWithParams<FilterPair<C>> for FilterPair<C> {
    type Params = SimpleFilterParams;

    fn new(params: &Self::Params) -> FilterPair<C> {
        let mut config = C::new(params);
        config.set_params(params);
        FilterPair::<C> {
            filter1: BiquadFilter::new(&config.get_filter1_params()),
            filter2: BiquadFilter::new(&config.get_filter2_params()),
            config: config,
        }
    }

    fn set_params(&mut self, params: &SimpleFilterParams) {
        self.config.set_params(params);
        self.filter1.set_params(&self.config.get_filter1_params());
        self.filter2.set_params(&self.config.get_filter2_params());
    }
}

impl<C: FilterPairConfig> AudioFilter for FilterPair<C> {
    fn apply_one(&mut self, sample: f32) -> f32 {
        self.filter2.apply_one(self.filter1.apply_one(sample))
    }
    fn reset(&mut self) {
        self.filter1.reset();
        self.filter2.reset();
    }
}

#[derive(Copy, Clone)]
pub struct LoudnessFilterPairConfig(SimpleFilterParams);

impl FilterPairConfig for LoudnessFilterPairConfig {
    type Impl = LoudnessFilterPairConfig;

    fn new(p: &SimpleFilterParams) -> LoudnessFilterPairConfig {
        LoudnessFilterPairConfig(*p)
    }
    fn set_params(&mut self, p: &SimpleFilterParams) {
        self.0 = *p;
    }

    fn get_filter1_params(&self) -> BiquadParams {
        const LOUDNESS_BASS_F0: BqCoef = 100.0;
        const LOUDNESS_BASS_Q: BqCoef = 0.25;
        BiquadParams::low_shelf_filter(
            self.0.sample_rate,
            LOUDNESS_BASS_F0,
            LOUDNESS_BASS_Q,
            self.0.gain_db,
        )
    }
    fn get_filter2_params(&self) -> BiquadParams {
        const LOUDNESS_TREBLE_F0: BqCoef = 14000.0;
        const LOUDNESS_TREBLE_Q: BqCoef = 1.0;
        BiquadParams::high_shelf_filter(
            self.0.sample_rate,
            LOUDNESS_TREBLE_F0,
            LOUDNESS_TREBLE_Q,
            self.0.gain_db / 2.0,
        )
    }
}

pub type LoudnessFilter = FilterPair<LoudnessFilterPairConfig>;

pub struct BassBoostFilter(BiquadFilter);

impl BassBoostFilter {
    fn get_biquad_params(params: &SimpleFilterParams) -> BiquadParams {
        BiquadParams::low_shelf_filter(params.sample_rate, 100.0, 0.7, params.gain_db)
    }
}

impl AudioFilterWithParams<BassBoostFilter> for BassBoostFilter {
    type Params = SimpleFilterParams;

    fn new(params: &Self::Params) -> Self {
        BassBoostFilter(BiquadFilter::new(&Self::get_biquad_params(params)))
    }
    fn set_params(&mut self, params: &Self::Params) {
        self.0.set_params(&Self::get_biquad_params(params))
    }
}

impl AudioFilter for BassBoostFilter {
    fn apply_one(&mut self, sample: f32) -> f32 {
        self.0.apply_one(sample)
    }
    fn apply_multi(&mut self, buffer: &mut [f32]) {
        self.0.apply_multi(buffer)
    }
    fn reset(&mut self) {
        self.0.reset()
    }
}

pub trait StreamFilter: Send {
    fn apply(&mut self, frame: Frame) -> Frame;
    fn reset(&mut self);
}

pub struct MultichannelFilter<T: AudioFilterWithParams<T>> {
    params: T::Params,
    filters: PerChannel<T>,
}

impl<T: AudioFilterWithParams<T>> MultichannelFilter<T> {
    pub fn new(params: T::Params) -> MultichannelFilter<T> {
        MultichannelFilter {
            params: params,
            filters: PerChannel::new(),
        }
    }

    pub fn set_params(&mut self, params: T::Params) {
        for (_, f) in self.filters.iter() {
            f.set_params(&params);
        }
        self.params = params
    }
}

impl<T: AudioFilter + AudioFilterWithParams<T> + Send> StreamFilter for MultichannelFilter<T> {
    fn apply(&mut self, mut frame: Frame) -> Frame {
        for (c, pcm) in frame.iter_channels() {
            if !self.filters.have_channel(c) {
                self.filters.set(c, T::new(&self.params))
            }

            self.filters.get_mut(c).unwrap().apply_multi(pcm);
        }
        frame
    }

    fn reset(&mut self) {
        self.filters.clear();
    }
}

pub struct PerChannelFilter<T: AudioFilterWithParams<T>> {
    params: PerChannel<T::Params>,
    filters: PerChannel<T>,
}

impl<T: AudioFilter + AudioFilterWithParams<T> + Send> PerChannelFilter<T> {
    pub fn new(params: PerChannel<T::Params>) -> PerChannelFilter<T> {
        let mut result = PerChannelFilter {
            params: params,
            filters: PerChannel::new(),
        };
        result.reset();
        result
    }

    pub fn set_params(&mut self, params: PerChannel<T::Params>) {
        self.params = params;
        self.reset();
    }
}

impl<T: AudioFilter + AudioFilterWithParams<T> + Send> StreamFilter for PerChannelFilter<T> {
    fn apply(&mut self, mut frame: Frame) -> Frame {
        for (c, f) in self.filters.iter() {
            match frame.get_channel_mut(c) {
                Some(pcm) => f.apply_multi(pcm),
                None => (),
            }
        }
        frame
    }

    fn reset(&mut self) {
        for (c, p) in self.params.iter() {
            self.filters.set(c, T::new(p))
        }
    }
}

#[derive(Clone)]
pub struct FirFilterParams {
    coefficients: Arc<Vec<FirCoef>>,
}

impl FirFilterParams {
    pub fn new(fir: Vec<FirCoef>, size: usize) -> FirFilterParams {
        FirFilterParams {
            coefficients: Arc::new(fir[0..size].to_vec()),
        }
    }
}

pub struct FirFilter {
    params: FirFilterParams,

    // Circular buffer for input samples.
    buffer: Vec<FirCoef>,

    // Current position in the buffer.
    buffer_pos: usize,

    window_size: usize,
}

impl AudioFilterWithParams<FirFilter> for FirFilter {
    type Params = FirFilterParams;

    fn new(params: &FirFilterParams) -> FirFilter {
        let size = params.coefficients.len();
        FirFilter {
            params: params.clone(),
            buffer: vec![0.0; size],
            buffer_pos: 0,
            window_size: size,
        }
    }

    fn set_params(&mut self, params: &FirFilterParams) {
        let size = params.coefficients.len();
        self.params = params.clone();
        self.buffer = vec![0.0; size];
        self.buffer_pos = 0;
        self.window_size = size;
    }
}

impl AudioFilter for FirFilter {
    fn apply_one(&mut self, x0: f32) -> f32 {
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
            ) + convolve(
                &self.buffer[0..(self.window_size - p1_size)],
                &self.params.coefficients[p1_size..self.window_size],
            )
        }
    }

    fn reset(&mut self) {
        self.buffer = vec![0.0; self.params.coefficients.len()];
    }
}

const CROSSFEED_LEVEL: f32 = 0.3;
const CROSSFEED_DELAY: f32 = 0.3;

pub struct CrossfeedFilter {
    enabled: bool,
    previous: Frame,
    left_cross_filter: BiquadFilter,
    left_straight_filter: BiquadFilter,
    right_cross_filter: BiquadFilter,
    right_straight_filter: BiquadFilter,
}

impl CrossfeedFilter {
    pub fn new(sample_rate: f64) -> CrossfeedFilter {
        CrossfeedFilter {
            enabled: true,
            previous: Frame::new(1.0, time::Time::now(), 0),
            left_straight_filter: BiquadFilter::new(&BiquadParams::low_shelf_filter(
                sample_rate as BqCoef,
                200.0,
                1.1,
                -2.5,
            )),
            left_cross_filter: BiquadFilter::new(&BiquadParams::low_pass_filter(
                sample_rate as BqCoef,
                3000.0,
                0.8,
            )),
            right_straight_filter: BiquadFilter::new(&BiquadParams::low_shelf_filter(
                sample_rate as BqCoef,
                200.0,
                1.1,
                -2.5,
            )),
            right_cross_filter: BiquadFilter::new(&BiquadParams::low_pass_filter(
                sample_rate as BqCoef,
                3000.0,
                0.8,
            )),
        }
    }

    fn crossfeed(
        out: &mut [f32],
        inp: &[f32],
        other: &[f32],
        prev: &[f32],
        delay: usize,
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
            out[out_pos] +=
                cross_filter.apply_one(prev[prev.len() + out_pos - delay]) * CROSSFEED_LEVEL;
            out_pos += 1;
        }

        while out_pos < out.len() {
            out[out_pos] += cross_filter.apply_one(other[out_pos - delay]) * CROSSFEED_LEVEL;
            out_pos += 1;
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn apply(&mut self, mut frame: Frame) -> Frame {
        if !self.enabled {
            return frame;
        }
        let mut out = Frame::new(frame.sample_rate, frame.timestamp, frame.len());
        out.gain = frame.gain;

        let delay = (CROSSFEED_DELAY * frame.sample_rate as f32 / 1000.0) as usize;

        frame.ensure_channel(CHANNEL_FL);
        frame.ensure_channel(CHANNEL_FR);

        CrossfeedFilter::crossfeed(
            out.ensure_channel(CHANNEL_FL),
            frame.get_channel(CHANNEL_FL).unwrap(),
            frame.get_channel(CHANNEL_FR).unwrap(),
            self.previous.ensure_channel(CHANNEL_FR),
            delay,
            &mut self.left_straight_filter,
            &mut self.left_cross_filter,
        );

        CrossfeedFilter::crossfeed(
            out.ensure_channel(CHANNEL_FR),
            frame.get_channel(CHANNEL_FR).unwrap(),
            frame.get_channel(CHANNEL_FL).unwrap(),
            self.previous.ensure_channel(CHANNEL_FL),
            delay,
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

impl AudioFilterWithParams<CascadingFilter> for CascadingFilter {
    type Params = BiquadParams;

    fn new(params: &BiquadParams) -> CascadingFilter {
        CascadingFilter {
            filter_1: BiquadFilter::new(params),
            filter_2: BiquadFilter::new(params),
        }
    }

    fn set_params(&mut self, params: &Self::Params) {
        self.filter_1.set_params(params);
        self.filter_2.set_params(params);
    }
}

impl AudioFilter for CascadingFilter {
    fn apply_one(&mut self, sample: f32) -> f32 {
        self.filter_2.apply_one(self.filter_1.apply_one(sample))
    }
    fn reset(&mut self) {
        self.filter_1.reset();
        self.filter_2.reset();
    }
}

pub type MultiBiquadParams = Vec<BiquadParams>;

pub struct MultiBiquadFilter {
    filters: Vec<BiquadFilter>,
}

impl AudioFilterWithParams<MultiBiquadFilter> for MultiBiquadFilter {
    type Params = MultiBiquadParams;

    fn new(params: &Self::Params) -> MultiBiquadFilter {
        MultiBiquadFilter {
            filters: params
                .iter()
                .map(|p| BiquadFilter::new(p))
                .collect::<Vec<BiquadFilter>>(),
        }
    }

    fn set_params(&mut self, params: &Self::Params) {
        let c = std::cmp::min(params.len(), self.filters.len());
        for i in 0..c {
            self.filters[i].set_params(&params[i])
        }
        if params.len() > c {
            for i in c..params.len() {
                self.filters.push(BiquadFilter::new(&params[i]));
            }
        } else if self.filters.len() > params.len() {
            self.filters.truncate(params.len());
        }
    }
}

impl AudioFilter for MultiBiquadFilter {
    fn apply_one(&mut self, mut sample: f32) -> f32 {
        for f in self.filters.iter_mut() {
            sample = f.apply_one(sample);
        }
        sample
    }

    fn apply_multi(&mut self, buffer: &mut [f32]) {
        for f in self.filters.iter_mut() {
            f.apply_multi(buffer);
        }
    }

    fn reset(&mut self) {
        for f in self.filters.iter_mut() {
            f.reset();
        }
    }
}

fn parse_param(s: &str, expected_prefix: &str) -> Option<BqCoef> {
    if !s.starts_with(expected_prefix) {
        return None;
    }
    s[expected_prefix.len()..].parse::<BqCoef>().ok()
}

fn parse_filter_values(parts: &[&str]) -> Option<BiquadParams> {
    Some(BiquadParams {
        b0: parse_param(parts[1], "b0=")?,
        b1: parse_param(parts[2], "b1=")?,
        b2: parse_param(parts[3], "b2=")?,
        a1: -parse_param(parts[4], "a1=")?,
        a2: -parse_param(parts[5], "a2=")?,
    })
}

pub fn parse_biquad_values(contents: String) -> Option<MultiBiquadParams> {
    let contents = contents
        .replace("\n", "")
        .replace("\r", "")
        .replace(" ", "");
    let parts = contents.split(",").collect::<Vec<&str>>();
    if parts.len() % 6 != 0 {
        return None;
    }

    let mut params = vec![];

    for i in 0..(parts.len() / 6) {
        let f = parse_filter_values(&parts[i * 6..i * 6 + 6])?;
        if !f.is_identity() {
            params.push(f);
        }
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

pub fn load_biquad_values(filename: &str) -> Result<MultiBiquadParams> {
    parse_biquad_values(fs::read_to_string(&filename)?)
        .ok_or_else(|| Error::from_string("Failed to parse ".to_owned() + &filename))
}

pub fn parse_biquad_line(sample_rate: f64, parts: &[&str]) -> Option<BiquadParams> {
    if parts.len() == 0 {
        return None;
    }
    let mut q = None;
    let mut f0 = None;
    let mut gain_db = None;

    let mut pos = 1;
    while pos < parts.len() {
        match parts[pos] {
            "Fc" => {
                if pos + 2 >= parts.len() || parts[pos + 2] != "Hz" || f0.is_some() {
                    return None;
                }
                f0 = Some(parts[pos + 1].parse::<f64>().ok()?);
                pos += 3;
            }
            "Gain" => {
                if pos + 2 >= parts.len() || parts[pos + 2] != "dB" || gain_db.is_some() {
                    return None;
                }
                gain_db = Some(parts[pos + 1].parse::<f64>().ok()?);
                pos += 3;
            }
            "Q" => {
                if pos + 1 >= parts.len() || q.is_some() {
                    return None;
                }
                q = Some(parts[pos + 1].parse::<f64>().ok()?);
                pos += 2;
            }
            _ => return None,
        }
    }

    let q = q.unwrap_or(1.0);

    let filter_name = parts[0];

    let filter = match filter_name {
        "LS" => BiquadParams::low_shelf_filter(sample_rate, f0?, q, gain_db?),
        "HS" => BiquadParams::high_shelf_filter(sample_rate, f0?, q, gain_db?),
        "LP" => BiquadParams::low_pass_filter(sample_rate, f0?, q),
        "HP" => BiquadParams::high_pass_filter(sample_rate, f0?, q),
        "PK" => BiquadParams::peaking_filter(sample_rate, f0?, q, gain_db?),
        "None" => BiquadParams::identity_filter(),
        _ => return None,
    };
    Some(filter)
}

pub fn parse_biquad_definition(sample_rate: f64, contents: String) -> Result<MultiBiquadParams> {
    let mut params = vec![];

    for line in contents.split("\n") {
        let parts = line.split_whitespace().collect::<Vec<&str>>();
        if parts.len() == 0 {
            continue;
        }
        let filter = parse_biquad_line(sample_rate, &parts)
            .ok_or_else(|| Error::from_string(format!("Can't parse filter line \"{}\"", line)))?;
        if !filter.is_identity() {
            params.push(filter);
        }
    }

    if params.is_empty() {
        Err(Error::new("Invalid biquad filter config"))
    } else {
        Ok(params)
    }
}

pub fn parse_biquad_config(sample_rate: f64, contents: String) -> Result<MultiBiquadParams> {
    let mut params = vec![];

    for line in contents.split("\n") {
        if !line.starts_with("Filter") {
            continue;
        }
        let colon_pos = line
            .find(":")
            .ok_or_else(|| Error::from_string(format!("Can't parse filter line \"{}\"", line)))?;
        let (_name, description) = line.split_at(colon_pos + 1);
        let parts = description.split_whitespace().collect::<Vec<&str>>();

        if parts.len() < 2 {
            return Err(Error::from_string(format!(
                "Can't parse filter line \"{}\"",
                line
            )));
        }

        println!("{:?}", parts);
        if parts[0] != "ON" {
            continue;
        }

        let filter = parse_biquad_line(sample_rate, &parts[1..])
            .ok_or_else(|| Error::from_string(format!("Can't parse filter line \"{}\"", line)))?;
        if !filter.is_identity() {
            params.push(filter);
        }
    }

    if params.is_empty() {
        Err(Error::new("Invalid biquad filter config"))
    } else {
        Ok(params)
    }
}

pub fn load_biquad_config(sample_rate: f64, filename: &str) -> Result<MultiBiquadParams> {
    parse_biquad_config(sample_rate, fs::read_to_string(&filename)?)
        .map_err(|e| Error::from_string(format!("Failed to parse {}: {}", filename, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_biquad_values() {
        let text = "biquad1,
b0=1.0023810711050223,
b1=-1.9968421156220293,
b2=0.9944879025811898,
a1=1.9968534162959661,
a2=-0.9968576730122751,
biquad2,
b0=1.0,
b1=0.0,
b2=0.0,
a1=0.0,
a2=0.0";
        let params = parse_biquad_values(text.to_owned()).unwrap();
        assert!(params.len() == 1);
        assert!(params[0].b0 == 1.0023810711050223);
        assert!(params[0].b1 == -1.9968421156220293);
        assert!(params[0].b2 == 0.9944879025811898);
        assert!(params[0].a1 == -1.9968534162959661);
        assert!(params[0].a2 == 0.9968576730122751);
    }

    #[test]
    fn test_parse_biquad_config() {
        let text = "Some text
Filter  1: ON  LS       Fc    23.0 Hz  Gain  16.0 dB
Filter  2: ON  PK       Fc    36.1 Hz  Gain -12.0 dB  Q  2.00
Filter  1: ON  HP       Fc    2300.0 Hz  Q 1.2
Filter  3: ON  PK       Fc    74.9 Hz  Gain -12.0 dB  Q  4.00
Filter  4: ON  None   
Filter  5: ON  None   
";
        let params = parse_biquad_config(48000.0, text.to_owned()).unwrap();
        assert!(
            params
                == vec![
                    BiquadParams::low_shelf_filter(48000.0, 23.0, 1.0, 16.0),
                    BiquadParams::peaking_filter(48000.0, 36.1, 2.0, -12.0),
                    BiquadParams::high_pass_filter(48000.0, 2300.0, 1.2),
                    BiquadParams::peaking_filter(48000.0, 74.9, 4.0, -12.0),
                ]
        );
    }

    #[test]
    fn test_parse_biquad_definition() {
        let text = "LS       Fc    23.0 Hz  Gain  16.0 dB
PK       Fc    36.1 Hz  Gain -12.0 dB  Q  2.00
HP       Fc    2300.0 Hz  Q 1.2
PK       Fc    74.9 Hz  Gain -12.0 dB  Q  4.00
";
        let params = parse_biquad_definition(48000.0, text.to_owned()).unwrap();
        assert!(
            params
                == vec![
                    BiquadParams::low_shelf_filter(48000.0, 23.0, 1.0, 16.0),
                    BiquadParams::peaking_filter(48000.0, 36.1, 2.0, -12.0),
                    BiquadParams::high_pass_filter(48000.0, 2300.0, 1.2),
                    BiquadParams::peaking_filter(48000.0, 74.9, 4.0, -12.0),
                ]
        );
    }
}
