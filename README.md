# Omaha Poker Hand Evaluator

A high-performance Omaha poker hand evaluator written in Rust, supporting both CPU and GPU (via `wgpu`) backends. It handles exhaustive enumeration and Monte Carlo simulations for Hand-vs-Hand, Hand-vs-Range, and Range-vs-Range scenarios, including Omaha Hi/Lo support.

## Features

- **High Performance**: Zero-allocation CPU evaluator and persistent GPU context for low-overhead throughput.
- **Parallel Processing**: Multi-threaded validation bench using `rayon` for massive throughput on multi-core CPUs.
- **Cross-Platform GPU Support**: Powered by `wgpu` (Metal, Vulkan, CUDA).
- **Omaha Hi/Lo**: Support for 8-or-better low hand evaluation.
- **Flexible Ranges**: Supports exact hands and rank-based range patterns (e.g., `AA`, `AKQJ`).
- **Validation Bench**: Built-in tool to validate accuracy against Pokerstove and benchmark against `ps-eval`.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
plo-eval-gpu = { path = "." } # Or relevant git/crates.io link
```

### Prerequisites

- **CPU**: Standard Rust toolchain.
- **GPU**: A system with support for Vulkan, Metal, or CUDA.

## Usage

### Basic Hand Evaluation

```rust
use plo_eval_gpu::{Card, Hand, Board, evaluate_omaha_hand};

fn main() {
    let hand = Hand::new([
        Card::from_str("As").unwrap(),
        Card::from_str("Ks").unwrap(),
        Card::from_str("Qh").unwrap(),
        Card::from_str("Jh").unwrap(),
    ]);
    let board = vec![
        Card::from_str("Ts").unwrap(),
        Card::from_str("Js").unwrap(),
        Card::from_str("Qs").unwrap(),
    ];
    
    let rank = evaluate_omaha_hand(&hand, &board);
    println!("Hand Rank: {:?}", rank);
}
```

### Equity Calculation (Hand vs Range)

```rust
use plo_eval_gpu::{Card, Hand, Range, Board, EvalMode, evaluate_hand_vs_range};

fn main() {
    let hero = Hand::new([
        Card::from_str("As").unwrap(),
        Card::from_str("Ac").unwrap(),
        Card::from_str("Ks").unwrap(),
        Card::from_str("Kc").unwrap(),
    ]);
    let villain_range = Range::from_shorthand("QQ,JT98", &[]).unwrap();
    let board = Board::new(vec![]); // Pre-flop
    
    let result = evaluate_hand_vs_range(
        hero, 
        villain_range, 
        board, 
        EvalMode::MonteCarlo { samples: 100000, seed: 42 }, 
        false // Not Hi/Lo
    );
    
    println!("Win: {:.2}%, Tie: {:.2}%", result.win * 100.0, result.tie * 100.0);
}
```

## Running Tests

### Unit Tests
Run the library unit tests to verify core logic:
```bash
cargo test
```

### Validation Test Bench
The `validation` binary compares the evaluator's results against a test set (e.g., Pokerstove output) and benchmarks performance.

#### Basic Validation
```bash
cargo run --release --bin validation -- --input data/test_results_10.txt --backend cpu
```

#### Benchmarking against `ps-eval`
If you have the `ps-eval` binary, you can compare performance:
```bash
cargo run --release --bin validation -- --input data/test_results_10.txt --ps-eval path/to/ps-eval
```

#### CLI Options
- `-i, --input <PATH>`: Path to the test file (space-separated `hero villain [board] equity`).
- `-b, --backend <BACKEND>`: `cpu`, `metal`, `vulkan`, `cuda`, or `auto`. (Omaha Hi/Lo always runs on CPU — the GPU shader only implements Omaha Hi.)
- `-m, --mode <MODE>`: `exhaustive`, `monte-carlo`, or `auto`.
- `-s, --samples <N>`: Number of Monte Carlo samples (default: 100,000).
- `-t, --tolerance <F>`: Equity difference tolerance (default: 0.1).

## Documentation

For detailed technical specifications, see [docs/PokerHandEvaluator.md](docs/PokerHandEvaluator.md).
Release notes and historical test results are available in the [docs/](docs/) folder.

## Performance

The internal CPU evaluator is optimized for high-performance Omaha evaluation.
- **Single-Hand Eval**: ~3.9ms per case.
- **Parallel Validation**: ~250μs per case on multi-core systems.
- **Accuracy**: 100% pass rate within 0.1 tolerance for benchmark datasets.

## Milestone Release Pipeline (M2.3.C1)

The project now features a robust, automated release pipeline to ensure quality and performance.
- **Automated Workflow**: Using `scripts/milestone.sh` to manage branching, verification, and GitHub releases.
- **Integrated Verification**: Mandatory accuracy (0.1 tolerance) and performance benchmarking before every release.
- **Strict Versioning**: Automated tagging and PR management via `gh` CLI.

Usage:
```bash
./scripts/milestone.sh start M3    # Start a new milestone
./scripts/milestone.sh verify      # Run full test suite and benchmarks
./scripts/milestone.sh release     # Merge to master, tag, and push
```

For details, see [docs/MilestoneReleasePipeline.md](docs/MilestoneReleasePipeline.md).

## Project Structure & Documentation

The repository has been reorganized for better maintainability:
- `docs/`: Technical specifications, milestone roadmaps, and release notes.
- `scripts/`: Automation and utility scripts.
- `data/`: Standardized Pokerstove benchmark datasets (e.g., `pokerstove_full_db.txt`).

## GPU Acceleration

The library features a GPU backend powered by `wgpu`, covering every street:
- **River (5-card board)**: exhaustive single comparison per hand pair.
- **Turn/Flop (4/3-card board)**: exhaustive enumeration of the missing board cards.
- **Pre-flop and earlier (< 3 known board cards)**: Monte Carlo sampling, with each hand pair's samples split across multiple GPU threads (`mc_lanes` in `src/gpu.rs`, `MC_TARGET_PARALLELISM`/`MC_MAX_LANES` in `src/omaha.wgsl`) instead of one thread looping over every sample — this matters a lot for the common case of one hero hand vs. one villain hand, where a single-thread-per-pair design leaves almost the whole GPU idle.
- **Routing**: `Backend::Auto` tries the GPU first for every case and falls back to CPU per-case if the GPU is unavailable (no adapter) or didn't resolve a case; Omaha Hi/Lo always runs on CPU (no GPU implementation).
- **Synchronization & memory safety**: explicit device polling (`wgpu::Maintain::Wait`) around buffer writes/reads, safe `bytemuck`-based heap-allocated `GpuInput` buffers.
- **Batching**: up to 256 cases per GPU dispatch, each up to 128 hero hands x 128 villain hands.

Measured on Apple Silicon (Metal): a 256-case, 1-hand-vs-1-hand, 1000-sample-Monte-Carlo batch (`simulation`'s no-flop workload) runs in ~0.04s on GPU as of the lane-split redesign, down from ~0.59s before it — and under realistic 8-way concurrent load (10,240 cases across 8 threads, matching `simulate_plo_no_flop`'s access pattern) went from 22.67s to 0.81s. A real end-to-end run (`omppu run-10k-append`, CPU-bound before this change) went from 6.24s to 0.93s, i.e. GPU throughput for this workload now exceeds the 8-thread CPU path (~1,650 cases/sec) by roughly 7-8x, instead of trailing it by ~3.6x.
