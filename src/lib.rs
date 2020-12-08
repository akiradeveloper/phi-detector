//! This is an implementation of Phi Accrual Failure Detector.
//! 
//! To reduce the memory footprint, pings or intervals aren't actually stored
//! but only two values to calculate normal distribution are maintained.
//! This not only reduces the memory footprint to the constant value
//! but also the computational cost for each ping down to constant.
//! 
//! Why does the memory footprint matter? Think about your application communicates
//! with thousand of remote servers and you want to maintain failure detector for each server.
//! Apparently, it is too wasting to cost 100MB to only for the failure detector.

use std::time::{Duration, Instant};

pub fn phi_from_prob(x: f64) -> f64 {
    assert!(0. <= x  && x <= 1.);
    -f64::log10(x)
}
/// Set of recent N ping intervals.
pub struct PingWindow {
    n: usize,
    last_ping: Instant,
    sum: f64,
    sum2: f64,
}
impl PingWindow {
    pub fn new() -> Self {
        let now = Instant::now();
        // initially, we have a super long value in the window
        // this value's the contribution to the normal distribution will be rapidly diluted
        // as actual heartbeats (usually much shorter than the initial value) fill the window.
        let deadline = Duration::from_secs(5);
        Self {
            n: 1,
            last_ping: now,
            sum: deadline.as_millis() as f64,
            sum2: 0.,
        }
    }
    pub fn last_ping(&self) -> Instant {
        self.last_ping
    }
    pub fn add_ping(&mut self, ping: Instant) {
        assert!(ping > self.last_ping);
        // window size too large is found meaningless in experiment.
        // not only that, may harm by counting in old values. (e.g. latency change, overflow)
        // the experiment shows the error rate saturate around n=10000.
        if self.n == 10000 {
            self.sum = self.sum / self.n as f64 * (self.n-1) as f64;
            // suppose each value has equal contribution to the variance.
            self.sum2 = self.sum2 / self.n as f64 * (self.n-1) as f64;
            self.n -= 1;
        }
        let v = (ping - self.last_ping).as_millis() as f64;
        self.last_ping = ping;
        self.sum += v;
        self.n += 1;
        let mu = self.sum / self.n as f64;
        self.sum2 += (v - mu) * (v - mu);
    }
    /// Make the current normal distribution based on the ping history.
    pub fn normal_dist(&self) -> NormalDist {
        let n = self.n;
        let mu = self.sum / n as f64;
        let sigma = f64::sqrt(self.sum2 / n as f64);
        NormalDist {
            mu, sigma,
        }
    }
}
/// Normal distribution from the ping history.
pub struct NormalDist {
    mu: f64,
    sigma: f64,
}
impl NormalDist {
    /// Mean
    pub fn mu(&self) -> Duration {
        Duration::from_millis(self.mu as u64)
    }
    /// Standard diviation
    pub fn sigma(&self) -> Duration {
        Duration::from_millis(self.sigma as u64)
    }
    /// Calculate integral [x, inf]
    /// This is a monotonically decreasing function.
    fn integral(&self, x: f64) -> f64 {
        // any small sigma rounds up to 1ms
        // which is negligible in the latency context.
        let sigma = if self.sigma < 1. {
            1.
        } else {
            self.sigma
        };
        let y = (x - self.mu) / sigma;
        let e = f64::exp(-y * (1.5976 + 0.070566 * y * y));
        if x > self.mu {
            e / (1. + e)
        } else {
            1. - 1./(1. + e)
        }
    }
    /// Calculate the phi from the current normal distribution
    /// and the duration from the last ping.
    pub fn phi(&self, elapsed: Duration) -> f64 {
        let x = elapsed.as_millis() as f64;
        let y = self.integral(x); // 0 <= y < = 1
        phi_from_prob(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_phi_detector() {
        let mut window = PingWindow::new();
        for _ in 0..100 {
            window.add_ping(Instant::now());
        }
        loop {
            let t = Instant::now() - window.last_ping();
            let dist = window.normal_dist();
            let phi = dist.phi(t);
            dbg!(phi);
            if phi > 10. {
                break;
            }
            tokio::time::delay_for(Duration::from_millis(10)).await;
        }
    }
    #[test]
    fn test_values() {
        let window = PingWindow::new();
        let dist = window.normal_dist();
        dbg!(dist.mu());
        dbg!(dist.sigma());
    }
}