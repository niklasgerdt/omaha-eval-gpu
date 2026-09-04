use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalMode {
    Auto,
    Exhaustive,
    MonteCarlo { samples: u64, seed: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Backend {
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityResult {
    pub win: f64,
    pub tie: f64,
    pub loss: f64,
    pub win_low: Option<f64>,
    pub tie_low: Option<f64>,
    pub loss_low: Option<f64>,
    pub trial_count: u64,
    pub mode: EvalMode,
    pub confidence_interval: Option<(f64, f64)>,
}
