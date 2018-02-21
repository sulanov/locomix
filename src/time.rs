extern crate libc;

use std;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::cmp::{Ord, Ordering};
use std::iter::Sum;

const NANOS_PER_SEC: i64 = 1_000_000_000;
const NANOS_PER_MILLI: i64 = 1_000_000;
const NANOS_PER_MICRO: i64 = 1_000;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct TimeDelta(i64);

impl TimeDelta {
    pub fn zero() -> TimeDelta {
        TimeDelta(0)
    }

    pub fn nanoseconds(nanos: i64) -> TimeDelta {
        TimeDelta(nanos)
    }

    pub fn seconds(secs: i64) -> TimeDelta {
        TimeDelta(secs * NANOS_PER_SEC)
    }

    pub fn milliseconds(ms: i64) -> TimeDelta {
        TimeDelta(ms * NANOS_PER_MILLI)
    }

    pub fn milliseconds_f(ms: f32) -> TimeDelta {
        TimeDelta((ms * NANOS_PER_MILLI as f32) as i64)
    }

    pub fn microseconds(us: i64) -> TimeDelta {
        TimeDelta(us * NANOS_PER_MICRO)
    }

    pub fn as_duration(&self) -> std::time::Duration {
        if self.0 <= 0 {
            return std::time::Duration::new(0, 0);
        }
        std::time::Duration::new(
            (self.0 / NANOS_PER_SEC) as u64,
            (self.0 % NANOS_PER_SEC) as u32,
        )
    }

    pub fn in_nanoseconds(&self) -> i64 {
        self.0
    }

    pub fn in_seconds(&self) -> i64 {
        self.0 / NANOS_PER_SEC
    }

    pub fn in_seconds_f(&self) -> f64 {
        self.0 as f64 / NANOS_PER_SEC as f64
    }

    pub fn in_milliseconds(&self) -> i64 {
        self.0 / NANOS_PER_MILLI
    }

    pub fn in_microseconds(&self) -> i64 {
        self.0 / NANOS_PER_MICRO
    }
    pub fn in_microseconds_f(&self) -> f64 {
        self.0 as f64 / NANOS_PER_MICRO as f64
    }


    pub fn abs(&self) -> TimeDelta {
        TimeDelta(self.0.abs())
    }
}

impl Add for TimeDelta {
    type Output = TimeDelta;

    fn add(self, rhs: TimeDelta) -> TimeDelta {
        TimeDelta(self.0 + rhs.0)
    }
}

impl AddAssign for TimeDelta {
    fn add_assign(&mut self, rhs: TimeDelta) {
        self.0 += rhs.0;
    }
}

impl Sub for TimeDelta {
    type Output = TimeDelta;

    fn sub(self, rhs: TimeDelta) -> TimeDelta {
        TimeDelta(self.0 - rhs.0)
    }
}

impl SubAssign for TimeDelta {
    fn sub_assign(&mut self, rhs: TimeDelta) {
        self.0 -= rhs.0
    }
}

impl Mul<i64> for TimeDelta {
    type Output = TimeDelta;

    fn mul(self, rhs: i64) -> TimeDelta {
        TimeDelta(self.0 * rhs)
    }
}

impl Mul<f32> for TimeDelta {
    type Output = TimeDelta;

    fn mul(self, rhs: f32) -> TimeDelta {
        TimeDelta((self.0 as f64 * rhs as f64) as i64)
    }
}

impl MulAssign<i64> for TimeDelta {
    fn mul_assign(&mut self, rhs: i64) {
        self.0 *= rhs;
    }
}

impl Div<i64> for TimeDelta {
    type Output = TimeDelta;

    fn div(self, rhs: i64) -> TimeDelta {
        TimeDelta(self.0 / rhs)
    }
}

impl DivAssign<i64> for TimeDelta {
    fn div_assign(&mut self, rhs: i64) {
        *self = *self / rhs;
    }
}

impl Div<TimeDelta> for TimeDelta {
    type Output = i64;

    fn div(self, rhs: TimeDelta) -> i64 {
        (self.0 / rhs.0)
    }
}

impl Sum for TimeDelta {
    fn sum<I: Iterator<Item = TimeDelta>>(iter: I) -> TimeDelta {
        iter.fold(TimeDelta::zero(), |a, b| a + b)
    }
}

impl<'a> Sum<&'a TimeDelta> for TimeDelta {
    fn sum<I: Iterator<Item = &'a TimeDelta>>(iter: I) -> TimeDelta {
        iter.fold(TimeDelta(0), |a, b| a + *b)
    }
}

impl Ord for TimeDelta {
    fn cmp(&self, other: &TimeDelta) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for TimeDelta {
    fn partial_cmp(&self, other: &TimeDelta) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Time(i64);

impl Time {
    pub fn zero() -> Time {
        Time(0)
    }

    pub fn now() -> Time {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        unsafe {
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
        }
        Time((ts.tv_sec as i64) * NANOS_PER_SEC + (ts.tv_nsec as i64))
    }
}

impl Add<TimeDelta> for Time {
    type Output = Time;

    fn add(self, delta: TimeDelta) -> Time {
        Time(self.0 + delta.in_nanoseconds())
    }
}

impl AddAssign<TimeDelta> for Time {
    fn add_assign(&mut self, delta: TimeDelta) {
        self.0 += delta.in_nanoseconds();
    }
}

impl Sub<TimeDelta> for Time {
    type Output = Time;

    fn sub(self, other: TimeDelta) -> Time {
        Time(self.0 - other.in_nanoseconds())
    }
}

impl SubAssign<TimeDelta> for Time {
    fn sub_assign(&mut self, other: TimeDelta) {
        self.0 -= other.in_nanoseconds();
    }
}

impl Sub<Time> for Time {
    type Output = TimeDelta;

    fn sub(self, other: Time) -> TimeDelta {
        TimeDelta::nanoseconds(self.0 - other.0)
    }
}

impl fmt::Debug for Time {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("Time({})", self.0))
    }
}

impl Ord for Time {
    fn cmp(&self, other: &Time) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Time {
    fn partial_cmp(&self, other: &Time) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for TimeDelta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("TimeDelta({})", self.0))
    }
}
