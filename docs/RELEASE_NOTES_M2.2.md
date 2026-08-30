### Milestone 2.2: Release Pipeline & Infrastructure Cleanup

### Summary
- Implemented the Milestone Release Pipeline and reorganized the project infrastructure for production readiness.
- Validated system integrity with a 100% pass rate on the full 246,000+ case Pokerstove dataset.

### Changes
- **Automation**: Created `scripts/milestone.sh` to automate the milestone lifecycle (start, verify, release) using GitHub CLI (`gh`).
- **Pipeline**: Defined `docs/MilestoneReleasePipeline.md` with strict requirements for accuracy (0.1 tolerance) and performance.
- **GPU Hardening**:
    - Migrated `GpuInput` structures from stack to heap to resolve infrastructure-level stack overflows.
    - Improved `Backend::Auto` to intelligently route workloads (e.g., rivered boards to GPU, early streets to CPU).
    - Simplified `omaha.wgsl` shader for better stability.
- **Project Structure**:
    - Reorganized all documentation into the `docs/` folder.
    - Renamed test data files in `data/` to standardized `pokerstove_*` naming.
    - Cleaned up `.idea` files and other local artifacts from Git tracking.

### Performance Verification
- **Accuracy**: 100% pass rate (246,401 cases) with 0.1 tolerance.
- **CPU Bench (Flop)**: ~2.9ms avg per case.
- **GPU Bench (River)**: Significant throughput for large batch evaluations.
- **Stability**: Resolved all known stack overflow issues during large-scale validation runs.

### Verification
- Full regression suite passed via `./scripts/milestone.sh verify`.
- Pipeline process tested by self-releasing Milestone M2.2.
