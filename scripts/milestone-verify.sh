#!/bin/bash
set -euo pipefail

echo "Running Functional Integrity Checks (cargo test)..."
cargo test

echo "Running Accuracy Check (Tolerance 0.1)..."
cargo run --release --bin validation -- --input data/pokerstove_full_db.txt --tolerance 0.1 --output docs/test_results.log

echo "Running Performance Benchmarks..."
cargo run --release --bin validation -- --input data/pokerstove_sample_100.txt --backend cpu --output docs/test_results.log
cargo run --release --bin validation -- --input data/pokerstove_sample_100.txt --backend auto --output docs/test_results.log

echo "Review docs/test_results.log for detailed metrics."
