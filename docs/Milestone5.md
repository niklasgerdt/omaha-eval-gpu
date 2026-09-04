# Milestone 5: Omaha Hi/Lo Capability

## Overview
Milestone 5 (M5) focuses on extending the evaluator's capabilities to support Omaha Hi/Lo (8-or-better) across all backends (CPU and GPU) and ensuring rigorous validation against industry-standard test sets, similar to the process established for PLO.

## Objectives

### 1. Hi/Lo Evaluation Logic
- **Split Pot Support**: Implement the logic to correctly identify the "Hi" hand (standard Omaha) and the "Lo" hand (8-or-better).
- **GPU Kernel Extension**: Update the WGSL shaders to support low hand evaluation. This involves checking for 5 unique ranks between 8 and Ace.
- **Result Aggregation**: Update `EquityResult` and the aggregation logic to handle both Hi and Lo equities (Win/Tie/Loss for both halves of the pot).

### 2. Validation & Testing
- **Test Set Acquisition**: Identify or generate comprehensive test sets for Omaha Hi/Lo, covering edge cases like:
  - Multiple players qualifying for Low.
  - No one qualifying for Low (Hi takes the whole pot).
  - Protected Low hands.
  - Quartering (splitting one half of the pot).
- **Harness Update**: Update the `validation` binary to support Hi/Lo comparison.
- **Accuracy Target**: Achieve a 100% pass rate within 0.1 tolerance against Pokerstove or equivalent high-fidelity evaluators for Hi/Lo.

### 3. API & Integration
- **Backend Transparency**: Ensure `Backend::Auto` correctly routes Hi/Lo queries to the appropriate backend (initially CPU until GPU kernels are verified).
- **Feature Parity**: Ensure that Hand-vs-Hand, Hand-vs-Range, and Range-vs-Range all support the `hi_lo` flag.

## Proposed Thresholds & Performance
- **Low Hand Logic**: Low evaluation adds complexity. The intelligent selector from M4 should be tuned to account for the increased computational cost of Hi/Lo.
- **Throughput**: Maintain high throughput for split-pot calculations by optimizing the combined Hi+Lo evaluation path.

## Implementation Plan
- [ ] **Low Hand Evaluator**: Implement a fast 8-or-better evaluator for the CPU.
- [ ] **GPU Low Hand Support**: Port the low hand logic to WGSL shaders.
- [ ] **Hi/Lo Equity Logic**: Update the equity calculators to track low hand results.
- [ ] **Hi/Lo Validation Data**: Add `data/pokerstove_hilo_sample.txt`.
- [ ] **Harness Enhancement**: Update `src/bin/validation.rs` to handle Hi/Lo output.

## Success Metrics
- **Correctness**: 100% match on known Hi/Lo test cases.
- **Performance**: Hi/Lo evaluation should not exceed 2x the time of standard Hi evaluation.
- **Completeness**: All range types (Exact, Rank Pattern) work for Hi/Lo.
