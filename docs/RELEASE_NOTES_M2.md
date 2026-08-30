### Milestone 2: GPU Acceleration (Metal)

### Summary
- Successfully implemented Metal (GPU) acceleration for Omaha range vs. range evaluations using `wgpu`.
- Achieved 100% pass rate on validation datasets compared to CPU implementation.

### Changes
- Introduced `src/gpu.rs` for GPU context management and kernel execution.
- Developed `src/omaha.wgsl` compute shader for massively parallel equity calculation.
- Added batching support in `src/bin/validation.rs` to leverage GPU parallelism efficiently.
- Updated `Backend` enum to support `Metal`, `Cuda`, and `Vulkan` (with Metal currently active on macOS).

### Performance Analysis
- **Small Datasets (100 cases):**
  - **CPU:** ~660ms (Total time)
  - **Metal:** ~2.7s (Total time)
  - **Observation:** CPU is faster for small batches due to zero GPU initialization and data transfer overhead.
- **Large/Complex Queries (Pre-flop):**
  - **CPU:** ~122ms (Avg per case)
  - **Metal:** ~27ms (Avg per case)
  - **Observation:** Metal is significantly more efficient for computationally intensive tasks once the initial overhead is amortized.
- **Conclusion:** GPU acceleration is preferred for large-scale range evaluations and complex pre-flop scenarios, while CPU remains optimal for single-hand or small-batch evaluations.

### Verification
- All 95 valid cases in `data/test_results_100.txt` passed on both CPU and Metal backends.
- Consistent results across multiple runs.
