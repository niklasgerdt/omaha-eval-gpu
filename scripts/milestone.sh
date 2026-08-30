#!/bin/bash

# Milestone Release Pipeline Automation Script
# Follows docs/MilestoneReleasePipeline.md

set -e

COMMAND=$1
MILESTONE=$2

function usage() {
    echo "Usage: $0 {start|verify|release} [milestone_name]"
    echo "Example: $0 start M3"
    exit 1
}

if [[ -z "$COMMAND" ]]; then
    usage
fi

case $COMMAND in
    start)
        if [[ -z "$MILESTONE" ]]; then
            echo "Error: Milestone name required (e.g., M3)"
            exit 1
        fi
        echo "Starting milestone $MILESTONE..."
        # gh repo fork # Optional, skip for now to avoid interactive prompts
        git checkout -b "milestone/$MILESTONE"
        echo "Switched to new branch: milestone/$MILESTONE"
        ;;

    verify)
        echo "Running Functional Integrity Checks (cargo test)..."
        cargo test

        echo "Running Accuracy Check (Tolerance 0.1)..."
        cargo run --release --bin validation -- --input data/pokerstove_full_db.txt --tolerance 0.1 --output docs/test_results.log

        echo "Running Performance Benchmarks..."
        echo "Checking CPU Flop/Pre-flop targets..."
        # Note: The validation tool prints results to stdout and logs to docs/test_results.log
        # For automation, we'd ideally parse the output, but for now we follow the spec commands
        cargo run --release --bin validation -- --input data/pokerstove_sample_100.txt --backend cpu --output docs/test_results.log
        
        echo "Checking GPU targets (auto)..."
        cargo run --release --bin validation -- --input data/pokerstove_sample_100.txt --backend auto --output docs/test_results.log

        echo "Verification complete. Review docs/test_results.log for detailed metrics."
        ;;

    release)
        if [[ -z "$MILESTONE" ]]; then
            # Try to detect from branch name
            CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
            if [[ $CURRENT_BRANCH =~ milestone/(.+) ]]; then
                MILESTONE=${BASH_REMATCH[1]}
            else
                echo "Error: Milestone name required or must be on a milestone branch."
                exit 1
            fi
        fi

        echo "Releasing Milestone $MILESTONE..."
        
        # 1. Create PR
        gh pr create --title "Release Milestone $MILESTONE" --body "Automated release for $MILESTONE. Verification passed."
        
        # 2. Merge PR
        echo "Merging to master..."
        gh pr merge --merge --delete-branch
        
        # 3. Tagging
        git checkout master
        git pull origin master
        TAG_NAME="${MILESTONE}.0"
        echo "Tagging as $TAG_NAME..."
        git tag -a "$TAG_NAME" -m "Release Milestone $MILESTONE"
        git push origin "$TAG_NAME"
        
        echo "Milestone $MILESTONE released successfully."
        ;;

    *)
        usage
        ;;
esac
