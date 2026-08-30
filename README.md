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
untitled1 = { path = "." } # Or relevant git/crates.io link
```

### Prerequisites

- **CPU**: Standard Rust toolchain.
- **GPU**: A system with support for Vulkan, Metal, or CUDA.

## Usage

### Basic Hand Evaluation

```rust
use untitled1::{Card, Hand, Board, evaluate_omaha_hand};

fn main() {
    let hand = Hand::from_str("AsKsQhJh").unwrap();
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
use untitled1::{Hand, Range, Board, EvalMode, Backend, evaluate_hand_vs_range};

fn main() {
    let hero = Hand::from_str("AsAcKsKc").unwrap();
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
- `-b, --backend <BACKEND>`: `cpu`, `metal`, `vulkan`, `cuda`, or `auto`. (Note: GPU backends currently support rivered boards only).
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

## Milestone Release Pipeline (M2.2)

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

## GPU Acceleration (M2.1 Hardening)

The library features a robust GPU backend powered by `wgpu`. Recent improvements in **M2.1** have focused on:
- **Synchronization**: Explicit device polling (`wgpu::Maintain::Wait`) to ensure zero-equity race conditions are eliminated.
- **Memory Safety**: Replaced unsafe pointer arithmetic with safe `bytemuck` and heap-allocated `GpuInput` for GPU buffers.
- **Batching**: Optimized batch submission for up to 256 cases per GPU call.
- **Intelligent Routing**: `Backend::Auto` now dynamically routes workloads to CPU or GPU based on the street and workload size.
