### Milestone 3: Dynamic Backend Optimization

### Objective
Optimize the evaluation engine to automatically choose between CPU and GPU backends based on the workload's characteristics.

### Tasks
- [ ] **Heuristic-based Backend Selection:**
  - Implement logic to select the backend based on batch size and query complexity (e.g., use CPU for < 50 cases, GPU for larger batches).
  - Take into account the type of query (Pre-flop vs. Post-flop) when deciding.
- [ ] **Initialization Optimization:**
  - Evaluate lazy vs. eager GPU initialization to minimize the impact on the first evaluation.
- [ ] **Improved Batching:**
  - Fine-tune batch sizes for different hardware to maximize throughput.
- [ ] **Cross-Platform Verification:**
  - Verify Vulkan and CUDA backends on supported hardware.
