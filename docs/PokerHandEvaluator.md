
# Omaha Poker Hand Evaluator Technical Specification

## 1. Introduction

This document serves as the technical specification for the Omaha Poker Hand Evaluator library. For general usage, installation, and how to run tests, please refer to the [README.md](../README.md).

## 4. Notation and Canonical Forms

### 4.1 Card Notation (§4.1.1)
Cards are represented by a 2-character string:
- **Rank**: `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `T` (10), `J` (Jack), `Q` (Queen), `K` (King), `A` (Ace).
- **Suit**: `s` (Spades), `h` (Hearts), `d` (Diamonds), `c` (Clubs).
Example: `As` (Ace of Spades), `Td` (Ten of Diamonds).

### 4.2 Hand and Board Representation (§4.1.2)
- A **Hand** is a sequence of 4 cards.
- A **Board** is a sequence of 0, 3, 4, or 5 cards.

### 4.3 Range Notation (§4.1.3)
Ranges are comma-separated lists of hands or rank-based patterns:
- **Exact Hand**: `AsKsQhJh`
- **Rank Pattern**: `AA` (any hand containing at least two Aces), `AKQJ` (any hand containing these four ranks).
- **Weighting**: Optional weight can be applied (implementation pending).

### 4.4 Canonical Ordering (§4.1.4)
To ensure consistent internal representation and avoid duplicate evaluations, hands and boards must be stored in canonical order:
1.  **Primary**: Rank descending (Ace is highest, 2 is lowest).
2.  **Secondary**: Fixed suit precedence: Spades (`s`) > Hearts (`h`) > Diamonds (`d`) > Clubs (`c`).

Example: `AcAsKh2d` is NOT canonical. Canonical order is `AsAcKh2d`.

---

### 7.2 Core Data Types (conceptual, not implementation-bound)

- `Card` — rank (2–A) + suit (4 suits), represented compactly (e.g., 0–51 index or bitmask); parsed from/rendered as the 2-character notation defined in §4.1.1.
- `Hand` — exactly 4 `Card`s (Omaha hole cards); always stored/serialized in the canonical order defined in §4.1.4 (rank descending, then fixed suit precedence).
- `Board` — 0, 3, 4, or 5 `Card`s; stored/serialized using the same canonical ordering rule as `Hand`.
- `Range` — collection of `(Hand, weight)` pairs, parsed from the comma-separated notation in §4.1.3; each `Hand` combo normalized per §4.1.4.
- `EquityRequest` — hero hand/range, optional villain hand/range, board, evaluation mode, sample count/seed.
- `EquityResult` — win/tie/loss percentages, trial count, mode, confidence interval, optional hand-category breakdown.

### 7.3 Hand Evaluation Algorithm

- Omaha scoring rule: best 5-card hand using **exactly 2 hole cards + exactly 3 board cards**.
- Use a fast 5-card hand evaluator (e.g., lookup-table-based, similar in spirit to Cactus Kev / Two Plus Two evaluators) adapted for the "choose 2 of 4, 3 of 5" combinatorics (C(4,2) × C(5,3) = 6 × 10 = 60 combinations per showdown to check per player).
- GPU kernels should implement the same lookup-table approach using GPU-resident tables for maximum throughput (avoid branch-heavy logic).

### 7.4 GPU Execution Model

- Each GPU thread evaluates one (board, hero hand, villain hand) triple, or is responsible for one Monte Carlo trial.
- Batches of trials are uploaded to GPU memory as flat arrays of card indices; results (win/tie/loss flags or scores) are read back and aggregated on CPU.
- Random number generation for Monte Carlo sampling on GPU uses a parallel-friendly PRNG (e.g., PCG or Xorshift) seeded per-thread deterministically from a master seed for reproducibility.
- **Street coverage**: GPU evaluation covers every street. Rivered boards (5 cards) get one exhaustive comparison per hand pair; turn/flop boards (4/3 cards) enumerate the missing board cards exhaustively; boards with fewer than 3 known cards use Monte Carlo sampling, with each hand pair's samples split across multiple GPU threads (see §7.5) rather than one thread looping over all of them. Omaha Hi/Lo has no GPU implementation and always runs on CPU.
- **M2.1 Hardening**: The GPU backend has been hardened to resolve zero-equity results by adding explicit device polling before buffer writes and using safe memory initialization patterns.

### 7.5 GPU Monte Carlo Lane Splitting

For a hand pair with fewer than 3 known board cards, one GPU thread running all `samples` trials sequentially is a poor use of the GPU — for `simulation`'s no-flop workload (1 hero hand vs. 1 villain hand per case), it means exactly 1 of the up to 128x128 GPU threads available per case does any work at all. Instead, each pair's samples are split across `mc_lanes(pair_count)` threads (`src/gpu.rs`), each running roughly `samples / lanes` trials with its own RNG stream (seeded from the case seed, case index, pair index and lane index) and contributing a `1/lanes`-sized share of the pair's total weight to the shared atomic accumulator (`src/omaha.wgsl`). `lanes` shrinks as `pair_count` grows (target ~4096 total MC threads per case, capped at 64 lanes/pair) so range-vs-range cases with many pairs don't blow up the dispatch size — those already get parallelism from the pair dimension. The host-side (`mc_lanes` in `gpu.rs`) and shader-side (`mc_lanes` in `omaha.wgsl`) formulas must stay identical, since both derive `pair_index`/`lane` from the same `global_id.x` and the host computes the dispatch's workgroup count from the same arithmetic the shader uses to interpret it.

---

## 8. Public API Sketch (conceptual)

> This is illustrative only — exact signatures to be finalized during implementation.

- `evaluate_hand_vs_hand(hero: Hand, villain: Hand, board: Board, mode: EvalMode) -> EquityResult`
- `evaluate_hand_vs_range(hero: Hand, villain_range: Range, board: Board, mode: EvalMode) -> EquityResult`
- `evaluate_range_vs_range(hero_range: Range, villain_range: Range, board: Board, mode: EvalMode) -> EquityResult`
- `random_hand(dead_cards: &[Card], rng_seed: Option<u64>) -> Hand`
- `EvalMode::{Auto, Exhaustive, MonteCarlo { samples: u64, seed: u64 }}`
- `Backend::{Auto, Cpu, Cuda, Vulkan, Metal}`

9. Validation and Release Pipeline

To ensure quality and performance, all milestone releases follow a strict automated pipeline.

1.  **Branching**: Each milestone is developed on a `milestone/MX` branch using `gh cli`.
2.  **Accuracy**: Must achieve a 100% pass rate against the Pokerstove dataset with 0.1 tolerance.
3.  **Performance**: Must meet benchmark speed targets derived from `docs/test_results.log`.
4.  **Merge**: Merged to `master` and tagged using `gh cli` after all checks pass.

For the full pipeline specification, see [docs/MilestoneReleasePipeline.md](MilestoneReleasePipeline.md).

---

## 10. Validation Plan Against Pokerstove Dataset

1. **Input format**: Define an agreed schema for the 1,000,000-sample checklist (e.g., CSV/JSON with hero hand, villain hand/range, board, expected win%/tie%/lose%).
2. **Harness**: A `validation` binary/test target loads the checklist, runs each case through the library (choosing enumeration or Monte Carlo per the recorded mode/sample size), and compares results.
3. **Tolerance**:
   - Exact enumeration cases: results must match Pokerstove to within 0.1 (10%).
   - Monte Carlo cases: results must fall within statistical tolerance based on sample count. A 100% pass rate on the 100-case dataset is achieved even with a reduced sample count of 10,000 using the 0.1 equity tolerance.
4. **Reporting**:
   - Harness outputs a summary (pass rate, execution speed, max deviation) to console.
   - A `test_results.log` file is updated after every run in the `docs/` folder, recording a summary of the run and performance metrics broken down by query type (Pre-flop, Flop, Turn, River).
   - **NEW**: New results are prepended to the top of `test_results.log` for immediate visibility.
   - **NEW**: The harness supports benchmarking against `ps-eval` if provided via the `--ps-eval` flag.
5. **Regression protection**: This dataset becomes part of the automated test suite, run on every change to evaluation logic.

---

## 10. Test Bench Input Format

The validation harness supports a space-separated format for quick testing against Pokerstove results:
`hero villain [board] equity`

- **Hero/Villain**: Comma-separated hands or ranges (e.g., `3s4d5cQh` or `6dJh8h9h,Td8d2d5h`).
- **Board**: Optional space-separated card list (e.g., `7d7h8c`).
- **Equity**: The expected win percentage as a float (0-100), where `nan` indicates an invalid or uncalculatable result (skipped).

Example: `7h4sAh7c 6dJh8h9h,Td8d2d5h 3c8dAs 62.561`

---

## 11. Performance Optimization and Benchmarking

The library is optimized for high-throughput evaluation, particularly for range-vs-range scenarios.

### 11.1 GPU Context Management
To minimize overhead, the GPU backend uses a global persistent context (initialized via `once_cell`). This ensures that expensive operations such as device selection and shader compilation occur only once per session.

### 11.2 CPU Evaluator
The CPU evaluator is implemented to be zero-allocation in the inner loops, using stack-based arrays and manual sorting/counting to avoid the overhead of `HashMap` and `Vec`.

### 11.3 Benchmarking and Sample Efficiency
The internal evaluator is designed to be highly efficient, significantly outperforming process-based wrappers like `ps-eval` due to zero-allocation logic and avoiding process spawning overhead.

#### EvalMode::Auto Performance
In `EvalMode::Auto`, the library intelligently selects between Exhaustive and Monte Carlo evaluation:
- **Pre-flop**: No board cards. The library uses exhaustive enumeration.
- **Flop**: 2 cards remaining. Enumerating all combinations (C(44,2) = 946) is faster and more accurate than Monte Carlo sampling, so `Auto` defaults to Exhaustive.
- **Result**: In `Auto` mode, execution time is independent of the `--samples` parameter for Flop and Pre-flop cases. Average internal time is **~35ms** per case for the 100-case dataset.

#### Forced Monte Carlo (Speed vs. Accuracy)
When Monte Carlo is forced (`--mode monte-carlo`), the sample count significantly impacts performance:
- **100,000 samples**: ~468ms per case.
- **10,000 samples**: ~46ms per case.
- **Observation**: A 10x reduction in samples yields a 10x speedup while still maintaining a **100% pass rate** within the 0.1 tolerance for the benchmark dataset.

### 11.4 Parallelization
The validation bench leverages the `rayon` library to process test cases in parallel across all available CPU cores. This enables high-throughput validation of large datasets (e.g., `data/test_results_db.txt` in ~61 seconds).

#### 11.5 Hardware Performance vs ps-eval
- **Internal Single-Hand**: ~1.8ms per case.
- **ps-eval Wrapper**: ~450ms per case (dominated by process overhead).
- **Internal Range-vs-Range**: ~35ms per case (`Auto` mode).
- **Large Dataset Throughput**: ~250μs per case (parallelized CPU/GPU Auto).

Note: The internal evaluator is now highly competitive with `ps-eval` for single-hand evaluations while maintaining full support for complex range-vs-range scenarios.

---

## 12. Testing Strategy

- **Unit tests**: Card/deck integrity, no-duplicate validation, hand ranking correctness on known hands (straight flush vs quads, etc.).
- **Property tests**: Randomized hands should never produce win+tie+lose ≠ 100%, no card reused across hero/villain/board, etc.
- **Cross-backend consistency tests**: CPU, CUDA, Vulkan, and Metal backends must produce identical results (within floating-point tolerance) on the same fixed-seed inputs.
- **Golden-file/regression tests** using the Pokerstove dataset (see §9).
- **Benchmark suite** (Criterion or custom) tracking throughput regressions per backend.

---

## 13. Build & Packaging

- Managed via Cargo workspace with feature flags:
  - `cpu` (always available, default)
  - `cuda` (Linux only, requires CUDA toolkit)
  - `vulkan` (Linux, optionally macOS via MoltenVK, requires Vulkan SDK/driver)
  - `metal` (macOS only)
- `Backend::Auto` detects available hardware/drivers at runtime and selects the best backend, falling back to `cpu` if none are available.
- CI matrix: Linux (with/without CUDA), macOS (Apple Silicon + Intel), each running the full test + validation suite where hardware allows (GPU-specific tests skipped gracefully on unsupported CI runners, but CPU path always tested).

---

## 14. Open Questions / Decisions Needed

1. Whether Omaha Hi/Lo (8-or-better) support is required for v1 (Resolved: Included in M1).
2. Range notation support level (Resolved: Shorthand notation implemented in M1).
3. Required precision/tolerance thresholds for Monte Carlo validation against Pokerstove (Resolved: 0.1 tolerance accepted for M1).
4. Target minimum GPU hardware/driver versions for CUDA/Vulkan/Metal.

---

## 15. Milestones
1. **M1 (COMPLETED)** — Core types, high-performance CPU/GPU evaluators, Omaha Hi/Lo support, range-vs-range calculations, parallel validation bench, and comprehensive documentation.
2. **M2.1 (COMPLETED)** — Hardened GPU backend with improved synchronization and memory safety. Resolved intermittent zero-equity results.
3. **M3** — Texas Hold'em as a first-class variant: 2-card hands, best-5-of-7 showdown on `evaluate_5_cards`, Hold'em range notation, CPU + GPU equity, validation dataset. See [docs/Milestone3.md](Milestone3.md). Omaha behaviour must not regress.
4. **M4 (Backend selection: partially completed)** — `Backend::Auto` tries GPU first and falls back to CPU per-case (see §7.4/§7.5); the GPU now covers every street via exhaustive turn/flop enumeration and lane-split Monte Carlo for boards under 3 cards, so the originally proposed street/case-size *selection heuristic* (docs/Milestone4.md) is superseded — GPU is attempted unconditionally rather than gated by a threshold. Multi-node support and enhanced range weighting remain open.
5. **M5** — Omaha Hi/Lo capability and comprehensive validation against split-pot test sets.
6. **M6** — Web-based visualization tools and distributed evaluation.
