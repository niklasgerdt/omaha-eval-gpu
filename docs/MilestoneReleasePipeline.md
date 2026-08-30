# Milestone Release Pipeline Specification

This document defines the automated pipeline for releasing new milestones of the Omaha Poker Hand Evaluator. The pipeline ensures code quality, accuracy, and performance before any merge to the `master` branch.

## 1. Branching Strategy

We use the GitHub CLI (`gh`) for all repository operations.

### 1.1 Starting a Milestone
When work begins on a new milestone (e.g., M3), a dedicated feature branch must be created:
```bash
gh repo fork # If working from a fork
git checkout -b milestone/M3
```

### 1.2 Development
All commits to the milestone branch should follow the project's commit message conventions.

---

## 2. Pre-Release Requirements

Before a milestone is considered "Done" and ready for merge, it must pass the following quality gates.

### 2.1 Functional Integrity
All unit tests and integration tests must pass:
```bash
cargo test
```

### 2.2 Accuracy Check
The `validation` tool must be run against the full benchmark dataset (`data/pokerstove_full_db.txt`).
- **Target**: 100% pass rate.
- **Tolerance**: 0.1 (10% equity difference).
```bash
cargo run --release --bin validation -- --input data/pokerstove_full_db.txt --tolerance 0.1
```

### 2.3 Performance Benchmark
The execution speed must meet or exceed the targets derived from the historical `docs/test_results.log`.

**Target Thresholds (Release Build):**
- **CPU Flop Evaluation**: < 3.0ms per case.
- **CPU Pre-flop Evaluation**: < 65.0ms per case.
- **GPU (Metal/Vulkan/CUDA) River Evaluation**: < 30.0ms per case (batch avg).
- **Parallel Throughput**: > 4,000 cases per second (for large datasets).

Verification command:
```bash
# Verify CPU performance
cargo run --release --bin validation -- --input data/pokerstove_sample_100.txt --backend cpu

# Verify GPU performance (if applicable)
cargo run --release --bin validation -- --input data/pokerstove_sample_100.txt --backend auto
```

---

## 3. Merge and Release Process

Once all requirements in Section 2 are met:

### 3.1 Pull Request
Create a Pull Request using `gh`:
```bash
gh pr create --title "Release Milestone M3" --body "Summary of changes and verification results."
```

### 3.2 Merge to Master
After review, merge the PR into the `master` branch:
```bash
gh pr merge --merge --delete-branch
```

### 3.3 Tagging
Tag the `master` branch with the milestone version:
```bash
git checkout master
git pull origin master
git tag -a M3.0 -m "Release Milestone 3.0"
git push origin M3.0
```

---

## 4. Derived Performance Targets

Based on the `docs/test_results.log` (Run: 2026-08-30T02:05:35), the following baseline was established on the development hardware:

| Query Type | Backend | Avg Time | Target |
| :--- | :--- | :--- | :--- |
| Flop | CPU | 5.15ms | < 3.0ms* |
| Pre-flop | CPU | 122.03ms | < 65.0ms* |
| Flop | Metal | 27.36ms | < 30.0ms |
| Overall (DB) | CPU | 267μs/case | < 250μs/case |

*\*Note: CPU targets are set aggressively based on the 246k case database run where average Flop time was 2.93ms.*
