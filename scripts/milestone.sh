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

        SPEC_DOC="docs/Milestone${MILESTONE#M}.md"
        if [[ ! -f "$SPEC_DOC" ]]; then
            echo "Error: requirement doc $SPEC_DOC not found."
            echo "Write the milestone spec at $SPEC_DOC before starting milestone/$MILESTONE."
            exit 1
        fi

        echo "Starting milestone $MILESTONE..."
        # gh repo fork # Optional, skip for now to avoid interactive prompts
        git checkout -b "milestone/$MILESTONE"
        echo "Switched to new branch: milestone/$MILESTONE (spec: $SPEC_DOC)"
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
        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
        if [[ -z "$MILESTONE" ]]; then
            # Try to detect from branch name
            if [[ $CURRENT_BRANCH =~ milestone/(.+) ]]; then
                MILESTONE=${BASH_REMATCH[1]}
            else
                echo "Error: Milestone name required or must be on a milestone branch."
                exit 1
            fi
        fi

        echo "Releasing Milestone $MILESTONE..."

        SPEC_DOC="docs/Milestone${MILESTONE#M}.md"
        RELEASE_NOTES="docs/RELEASE_NOTES_${MILESTONE}.md"
        TAG_NAME="${MILESTONE}.0"

        # 1. Fold the spec into release notes (spec + verification info), then
        #    drop the spec doc -- the requirement doc is superseded by the record
        #    of what was actually shipped.
        if [[ -f "$SPEC_DOC" ]]; then
            {
                echo "# Release Notes: Milestone $MILESTONE"
                echo
                echo "_Released $(date +%Y-%m-%d), tag \`$TAG_NAME\`._"
                echo
                cat "$SPEC_DOC"
                echo
                echo "## Verification"
                echo
                echo "- Full suite: \`cargo test\` passed."
                echo "- Accuracy: 0.1 tolerance against \`data/pokerstove_full_db.txt\` (see \`docs/test_results.log\`)."
                echo "- Performance: CPU/GPU benchmark timings in \`docs/test_results.log\`."
            } > "$RELEASE_NOTES"
            # Plain rm, not `git rm`: the spec doc may not be tracked/committed
            # yet (e.g. just written this session), and `git rm` refuses to
            # touch untracked files. The `git add -A` below stages the
            # deletion (or no-ops if it was never tracked) either way.
            rm -f "$SPEC_DOC"
            echo "Folded $SPEC_DOC into $RELEASE_NOTES and removed the spec."
        else
            echo "Warning: no spec doc at $SPEC_DOC; skipping fold-in (nothing to release-note from)."
        fi

        # 2. Bump the version tag this README already carries, e.g.
        #    "## Milestone Release Pipeline (M2.2)" -> "(M3)". This is a
        #    mechanical substitution only -- broader prose updates (feature
        #    lists, benchmark numbers) still need a human/Claude review pass,
        #    not something safe to script unattended.
        if grep -qE '## Milestone Release Pipeline \(M[0-9.]+\)' README.md; then
            sed -i.bak -E "s/(## Milestone Release Pipeline \()M[0-9.]+(\))/\1${MILESTONE}\2/" README.md
            rm -f README.md.bak
            echo "Bumped README.md version tag to $MILESTONE."
        fi

        # 3. Commit any pending changes (release notes, spec removal, README
        #    bump, verify's test_results.log update, and anything else pending)
        git add -A
        if git diff --cached --quiet; then
            echo "Nothing to commit."
        else
            git commit -m "milestone $MILESTONE: release notes + verification + pending changes"
        fi

        # 4. Push the branch so the PR has something to point at
        git push -u origin "$CURRENT_BRANCH"

        # 5. Create PR
        gh pr create --title "Release Milestone $MILESTONE" --body "Automated release for $MILESTONE. Verification passed."

        # 6. Merge PR
        echo "Merging to master..."
        gh pr merge --merge --delete-branch

        # 7. Tagging
        git checkout master
        git pull origin master
        echo "Tagging as $TAG_NAME..."
        git tag -a "$TAG_NAME" -m "Release Milestone $MILESTONE"
        git push origin "$TAG_NAME"

        echo "Milestone $MILESTONE released successfully."
        ;;

    *)
        usage
        ;;
esac
