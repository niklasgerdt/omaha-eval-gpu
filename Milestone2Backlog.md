# Milestone 2 Backlog — GPU Incomplete-Board Support & Beyond

## Overview

Milestone 1 delivered a correct, high-performance CPU evaluator and a GPU (`wgpu`) backend that is currently restricted to **rivered boards (exactly 5 cards)**. All Pre-flop, Flop, and Turn range-vs-range queries silently fall back to CPU, so the GPU is idle for the majority of real-world workloads (see `test_results.log`, where 246,401/246,402 Flop cases ran on CPU).

This backlog defines the work required to extend GPU acceleration to **incomplete boards (0, 3, and 4 known cards)**, along with related correctness, testing, and follow-on M2 items. This directly targets the current GPU limitation documented in `PokerHandEvaluator.md` §7.4.

## Goals

1. Remove the "5-card boards only" restriction on the GPU backend.
2. Achieve GPU-side board completion for Turn and Flop via exhaustive enumeration.
3. Achieve GPU-side board completion for Pre-flop (and any case with a large enough runout space) via per-thread Monte Carlo sampling with a reproducible PRNG.
4. Preserve bit-exact/statistical parity with the CPU backend (per `test_gpu_vs_cpu` and the validation harness).
5. Keep Omaha Hi/Lo out of scope for the first sub-milestone unless time permits (tracked as a separate epic below).

---

## Epic 1: GPU Incomplete-Board Evaluation (Preflop / Flop / Turn)

### 1.1 Fix board-padding sentinel bug (Blocker) ✓
- **Problem**: `GpuInput.board` pads unused slots with `0u32`, which decodes to a real card (Two of Spades). This corrupts dead-card exclusion checks once boards shorter than 5 cards are allowed through.
- **Task**: Change padding value to an out-of-range sentinel (e.g. `255u32`) on both the Rust (`gpu.rs`) and WGSL (`omaha.wgsl`) sides. Update all `is_dead`/overlap comparisons to treat the sentinel as "never matches."
- **Acceptance**: Unit test constructs a board with `board_len < 5` and confirms no false-positive dead-card collisions against Two of Spades.
- **Status**: Completed. Rust side uses `255u32` for padding. WGSL side treats `255u32` as sentinel. Verified by `test_gpu_padding_sentinel`.

### 1.2 Exhaustive Turn completion kernel ✓
- **Task**: Add a WGSL function that, given a 4-card board, loops over all 52 card indices, skips dead cards (hero + villain + board), and evaluates the resulting 5-card board for each pair. Accumulate win/tie/count.
- **Acceptance**: For a fixed hero/villain/4-card-board input, GPU win/tie/loss matches CPU `EvalMode::Exhaustive` result within `1e-6`.
- **Status**: Completed. Implemented in `omaha.wgsl`. Verified against `data/test_results_100.txt`.

### 1.3 Exhaustive Flop completion kernel ✓
- **Task**: Add a nested-loop WGSL function (`i`, `j > i` over 52 indices, skipping dead cards) for 3-card boards (≤ 990 iterations per thread).
- **Acceptance**: GPU result matches CPU exhaustive result within `1e-6` for representative Flop test cases (reuse cases from `data/test_results_10.txt` / `test_results_100.txt`).
- **Status**: Completed. Implemented in `omaha.wgsl`. Verified against `data/test_results_100.txt`.

### 1.4 Monte Carlo Pre-flop kernel ✓
- **Task**: Implement a per-thread PRNG (xorshift32 or PCG) seeded from `input.seed XOR pair_index` (or similar) and a partial Fisher–Yates draw of the missing cards from a per-thread filtered deck array. Accumulate win/tie over `input.samples` trials, dividing once per thread before the final atomic write (not per-sample) to minimize atomic contention.
- **Acceptance**:
  - Deterministic: same `seed` + same inputs → same result, across repeated runs.
  - Statistical parity: GPU Monte Carlo result falls within the same confidence interval as CPU Monte Carlo result (same seed/sample count) for a representative Pre-flop test set.
- **Status**: Completed. PCG hash based PRNG implemented. Fisher-Yates draw for missing cards. Verified by validation bench.

### 1.5 Rust-side `EvalMode::Auto` resolution before GPU dispatch ✓
- **Task**: Mirror the CPU semantics in `evaluate_hand_vs_hand` (`Auto` → Exhaustive for `board_len >= 3`, Monte Carlo fallback otherwise) *before* populating `GpuInput`, so `mode`/`samples`/`seed` are always concretely set rather than defaulting to `0`.
- **Acceptance**: `Backend::Auto`/`Backend::Metal`/etc. produce non-zero, correctly-classified `mode` values in `GpuInput` for all board lengths in the validation dataset.
- **Status**: Completed in `src/gpu.rs`.

### 1.6 Remove the board-length guard ✓
- **Task**: Replace `if board.0.len() != 5 { return None; }` in `run_gpu_range_evaluation` with validation for `{0, 3, 4, 5}` (reject anything else with a typed error rather than silently falling back).
- **Acceptance**: `Backend::Cuda|Vulkan|Metal` no longer returns `None` for Pre-flop/Flop/Turn inputs; `Backend::Auto` actually dispatches to GPU for these cases when a GPU is available.
- **Status**: Completed. Guard updated to allow `0, 3, 4, 5` card boards.

### 1.7 (Stretch) Range size beyond 128 hands
- **Task**: Investigate tiling/batching strategy so hero/villain range sizes are not hard-capped at 128 combos (current `GpuInput` fixed-size arrays). Out of scope for the init
