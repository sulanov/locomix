use std::cmp;
use std::f32::consts::PI;
use std::f64::consts::PI as PI64;
use std::sync::Arc;
use std::collections::HashMap;
use base::convolve;

struct QuadFunction {
    a: f32,
    b: f32,
    c: f32,
    mid: f32,
}

impl QuadFunction {
    pub fn new(at0: f32, athalf: f32, at1: f32) -> QuadFunction {
        QuadFunction {
            a: 2.0 * at0 + 2.0 * at1 - 4.0 * athalf,
            b: -3.0 * at0 - at1 + 4.0 * athalf,
            c: at0,
            mid: athalf,
        }
    }
    // pub fn eval(&self, x: f32) -> f32 {
    //  x * (x * self.a + self.b) + self.c
    // }
    pub fn eval2(&self, x2: f32, x: f32) -> f32 {
        x2 * self.a + x * self.b + self.c
    }
}

pub struct Resampler {
    i_freq: usize,
    o_freq: usize,
    size: usize,
    queue: Vec<f32>,
    win: Vec<QuadFunction>,

    i_pos: usize,
    o_pos: usize,
}

fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        return 1.0;
    }
    x.sin() / x
}

fn sinc64(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    x.sin() / x
}

impl Resampler {
    pub fn new(i_freq: usize, o_freq: usize, size: usize) -> Resampler {
        Resampler {
            i_freq: i_freq,
            o_freq: o_freq,
            size: size,
            queue: Vec::with_capacity(size * 2 + 1),
            win: (-(size as i32 + 1)..(size as i32) + 1)
                .map(|x| {
                    let sign = (if x >= 0 {
                                    1 - 2 * (x % 2)
                                } else {
                                    1 + 2 * (x % 2)
                                }) as f32;
                    QuadFunction::new(sign * sinc(x as f32 * PI / size as f32) / PI,
                                      sign * sinc((x as f32 + 0.5) * PI / size as f32) / PI,
                                      sign * sinc((x as f32 + 1.0) * PI / size as f32) / PI)
                })
                .collect(),
            i_pos: 0,
            o_pos: 0,
        }
    }

    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity((input.len() * self.o_freq / self.i_freq) as usize + 1);

        let mut o_pos = self.o_pos;
        let freq_ratio = self.i_freq as f64 / self.o_freq as f64;
        loop {
            let o_pos_i_freq = (o_pos as f64 * freq_ratio - self.i_pos as f64) as f32;
            let i_mid = o_pos_i_freq.floor() as usize;
            let i_shift = o_pos_i_freq.fract();
            let i_shift_2 = i_shift * i_shift;

            if i_mid + self.size >= self.queue.len() + input.len() {
                break;
            }

            let result: f32;
            if i_shift == 0.0 {
                result = if i_mid < self.queue.len() {
                    self.queue[i_mid]
                } else {
                    input[i_mid - self.queue.len()]
                }
            } else {
                let mut start = if i_mid + 1 > self.size {
                    i_mid - self.size + 1
                } else {
                    0
                };
                let sin_value = (i_shift * PI).sin();
                let queue_len = self.queue.len();
                let mut x = i_shift + i_mid as f32 - start as f32;
                let mut win_pos = 1 + i_mid + self.size - start;

                let mut sum = 0.0;
                if i_shift == 0.5 {
                    if start < queue_len {
                        for i in start..queue_len {
                            sum += self.queue[i] * self.win[win_pos].mid / x;
                            x -= 1.0;
                            win_pos -= 1;
                        }
                        start = queue_len;
                    }

                    for i in (start - queue_len)..(i_mid + self.size + 1 - queue_len) {
                        sum += input[i] * self.win[win_pos].mid / x;
                        x -= 1.0;
                        win_pos -= 1;
                    }
                } else {
                    if start < queue_len {
                        for i in start..queue_len {
                            sum += self.queue[i] / x * self.win[win_pos].eval2(i_shift_2, i_shift);
                            x -= 1.0;
                            win_pos -= 1;
                        }
                        start = queue_len;
                    }

                    for i in (start - queue_len)..(i_mid + self.size + 1 - queue_len) {
                        sum += input[i] / x * self.win[win_pos].eval2(i_shift_2, i_shift);
                        x -= 1.0;
                        win_pos -= 1;
                    }
                }

                result = sin_value * sum
            };

            output.push(result);
            o_pos += 1;
        }

        let samples_to_keep = cmp::min(self.size * 2 + 1, self.queue.len() + input.len());
        let samples_to_remove = self.queue.len() + input.len() - samples_to_keep;
        if samples_to_remove > self.queue.len() {
            self.queue.clear();
            self.queue
                .extend_from_slice(&input[(input.len() - samples_to_keep)..]);
        } else {
            self.queue.drain(0..samples_to_remove);
            self.queue.extend_from_slice(input);
        }

        self.i_pos = self.i_pos + samples_to_remove;
        self.o_pos = o_pos;
        while self.i_pos > self.i_freq && self.o_pos > self.o_freq {
            self.i_pos -= self.i_freq;
            self.o_pos -= self.o_freq;
        }

        output
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
struct ResamplerConfig {
    freq_num: usize,
    freq_denom: usize,
    size: usize,
}

struct ResamplerTable {
    config: ResamplerConfig,
    kernel: Vec<Vec<f32>>,
}

impl ResamplerTable {
    pub fn new(config: ResamplerConfig) -> ResamplerTable {
        let mut kernel = Vec::with_capacity((config.freq_denom + 1) / 2);
        for n in 0..config.freq_denom {
            let mut row = Vec::with_capacity(config.size + 1);
            for i in 0..(config.size * 2) {
                let x = config.size as f64 - i as f64 + n as f64 / config.freq_denom as f64;
                row.push((sinc64(PI64 * x) * sinc64(PI64 * x / config.size as f64)) as f32);
            }
            kernel.push(row);
        }

        ResamplerTable {
            config: config,
            kernel: kernel,
        }
    }
}

pub struct FastResampler {
    table: Arc<ResamplerTable>,
    queue: Vec<f32>,

    // Input position for the next output sample relative to the
    // first sample in the queue.
    i_pos: usize,
    i_pos_frac: usize,
}

impl FastResampler {
    fn new(table: Arc<ResamplerTable>) -> FastResampler {
        FastResampler {
            table: table,
            queue: Vec::new(),
            i_pos: 0,
            i_pos_frac: 0,
        }
    }

    pub fn resample(&mut self, input: &[f32]) -> Vec<f32> {
        let table = &self.table;
        let mut output = Vec::with_capacity((input.len() * table.config.freq_denom /
                                             table.config.freq_num) as
                                            usize + 1);

        loop {
            if self.i_pos + table.config.size >= self.queue.len() + input.len() {
                break;
            }

            let mut result: f32 = 0.0;
            let queue_len = self.queue.len();

            if self.i_pos_frac == 0 {
                result = if self.i_pos < queue_len {
                    self.queue[self.i_pos]
                } else {
                    input[self.i_pos - queue_len]
                }
            } else {
                let mut start = if self.i_pos + 1 > table.config.size {
                    self.i_pos - table.config.size + 1
                } else {
                    0
                };

                let kernel = &self.table.kernel[self.i_pos_frac];
                let mut k_pos = table.config.size - self.i_pos + start;

                if start < queue_len {
                    result += convolve(&self.queue[start..queue_len],
                                       &kernel[k_pos..(k_pos + queue_len - start)]);
                    k_pos += queue_len - start;
                    start = queue_len;
                }

                let len = self.i_pos + table.config.size - start;
                result += convolve(
                    &input[(start - queue_len)..(self.i_pos + table.config.size - queue_len)],
                    &kernel[k_pos..(k_pos + len)]);
            };

            output.push(result);

            self.i_pos_frac += table.config.freq_num;
            self.i_pos += self.i_pos_frac / table.config.freq_denom;
            self.i_pos_frac = self.i_pos_frac % table.config.freq_denom;
        }

        let samples_to_keep = cmp::min(table.config.size * 2 + 1, self.queue.len() + input.len());
        let samples_to_remove = self.queue.len() + input.len() - samples_to_keep;
        if samples_to_remove > self.queue.len() {
            self.queue.clear();
            self.queue
                .extend_from_slice(&input[(input.len() - samples_to_keep)..]);
        } else {
            self.queue.drain(0..samples_to_remove);
            self.queue.extend_from_slice(input);
        }

        self.i_pos -= samples_to_remove;

        output
    }
}

pub struct ResamplerFactory {
    table_cache: HashMap<ResamplerConfig, Arc<ResamplerTable>>,
}

fn get_greatest_common_divisor(mut a: usize, mut b: usize) -> usize {
    // Euclid's algorithm.
    while b > 0 {
        let c = a % b;
        a = b;
        b = c;
    }
    return a;
}

impl ResamplerFactory {
    pub fn new() -> ResamplerFactory {
        ResamplerFactory { table_cache: HashMap::new() }
    }

    pub fn create_resampler(&mut self, i_freq: usize, o_freq: usize, size: usize) -> FastResampler {
        let gcd = get_greatest_common_divisor(i_freq, o_freq);
        let freq_num = i_freq / gcd;
        let freq_denom = o_freq / gcd;

        let config = ResamplerConfig {
            freq_num: freq_num,
            freq_denom: freq_denom,
            size: size,
        };

        let table = self.table_cache
            .entry(config)
            .or_insert_with(|| Arc::new(ResamplerTable::new(config)))
            .clone();

        FastResampler::new(table)
    }
}


#[cfg(test)]
mod tests {
    use resampler;
    use std::time::Instant;

    fn val(i: usize, rate: usize, freq: f64) -> f64 {
        let t = i as f64 / rate as f64;
        (t * freq * 2.0 * ::std::f64::consts::PI).sin()
    }

    fn get_power(s: &[f64]) -> f64 {
        s.iter().fold(0f64, |r, &v| r + v * v) / (s.len() as f64)
    }

    #[test]
    fn it_works() {
        let irate: usize = 44100;
        let orate: usize = 96000;
        let f: f64 = 17433.41;

        let mut buf = Vec::<f32>::new();
        for i in 0..(irate * 2) {
            let v = val(i, irate, f);
            buf.push(((v * 32768.0).round() / 32768.0) as f32);
            // buf.push(v as f32);
        }

        let mut factory = resampler::ResamplerFactory::new();
        let mut r = factory.create_resampler(irate, orate, 200);
        let mut out = Vec::new();

        let start = Instant::now();

        const RANGE: usize = 441;

        for i in 0..(buf.len() / RANGE) {
            let s = i * RANGE;
            let e = (i + 1) * RANGE;
            out.extend_from_slice(&r.resample(&buf[s..e]));
        }

        let d = Instant::now() - start;
        println!("{} {} ", d.as_secs(), d.subsec_nanos() as f32 / 1000000.0);
        assert!(out.len() > orate as usize - 200);

        let out64: Vec<f64> = out.iter().map(|x| -> f64 { *x as f64 }).collect();

        for i in 100..200 {
            println!("{} {} {}", i, out[i], {
                out64[i] - val(i, orate, f)
            });
        }
        for i in 1..(out.len() / RANGE) {
            let s = i * RANGE;
            let e = (i + 1) * RANGE;
            let signal = get_power(&out64[s..e]);
            let noise_signal: Vec<f64> = ((s + 1)..e)
                .map(|i| out64[i] - val(i, orate, f))
                .collect();
            let noise = get_power(&noise_signal);
            let nsr = noise / signal;
            let nsr_db = 10.0 * nsr.log(10.0);
            println!("{} {} {} {}", signal, noise, 1.0 / nsr, nsr_db);
            // Target: -97
            assert!(nsr_db < -97.0);
        }
    }
}
