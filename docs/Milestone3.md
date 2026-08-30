# Milestone 3: Intelligent Backend Selection

## Overview
Milestone 3 (M3) focuses on optimizing the evaluation pipeline by implementing an intelligent, automatic selection between CPU and GPU backends. This decision is based on the characteristics of the input parameters to maximize throughput and minimize latency.

## Key Insights
Based on historical performance data and architectural constraints:
- **GPU Strength**: Massive parallelization. Ideal for "large" problems where the overhead of memory transfer and kernel dispatch is amortized over a high number of evaluations (e.g., large range-vs-range scenarios or batch processing).
- **CPU Strength**: Low latency and high single-thread speed. Superior for "small" problems (e.g., single hand-vs-hand evaluations) due to the absence of context switching and GPU synchronization overhead.
- **Current Limitation**: GPU evaluation currently only supports **Rivered boards (5 cards)** and does not support **Omaha Hi/Lo**.

## M3 Objectives

### 1. Heuristic-Based Backend Selector
Implement a selection logic within `Backend::Auto` that considers:
- **Case Size**: The product of Hero range size and Villain range size.
- **Batch Size**: The number of concurrent equity queries requested.
- **Complexity**: Evaluation mode (Exhaustive vs. Monte Carlo).
- **Street**: CPU is currently required for Pre-flop, Flop, and Turn.

### 2. Proposed Thresholds
Initial heuristics to be refined via benchmarking:
- **Single Range Evaluation**: 
  - Use **GPU** if `hero_range.len() * villain_range.len() > 10,000`.
  - Use **CPU** otherwise.
- **Batch Evaluation**:
  - Use **GPU** if `total_pairs > 5,000` or `batch_size > 32`.
- **Street-Based**:
  - Always use **CPU** for boards with < 5 cards (until GPU kernels are updated).
- **Mode-Based**:
  - Always use **CPU** for Omaha Hi/Lo.

### 3. Implementation Plan
- [ ] **Enhance `Backend::Auto`**: Refactor `run_range_evaluation` and `run_range_evaluation_batch` in `src/lib.rs` to use the new heuristic.
- [ ] **Dynamic Profiling**: Add internal telemetry to log execution times for CPU vs GPU on similar workloads to fine-tune thresholds.
- [ ] **Fallback Mechanism**: Ensure seamless fallback to CPU if GPU resources are exhausted or initialization fails.
- [ ] **Benchmarking Suite**: Create a dedicated benchmark in `benches/` to validate the heuristics across different hardware (Metal, Vulkan, CUDA).

## Success Metrics
- **Latency**: Reduce average response time for small range-vs-range queries by avoiding GPU overhead.
- **Throughput**: Increase total cases per second for large datasets by saturating the GPU only when beneficial.
- **Efficiency**: Zero manual configuration required by the user to achieve optimal performance.
