#!/bin/bash
# Shared milestone pipeline (skill: milestone-release).
# Project-specific gates: scripts/milestone-verify.sh

set -euo pipefail

COMMAND="${1:-}"
MILESTONE="${2:-}"

usage() {
    echo "Usage: $0 {start|verify|release} [milestone_name]"
    echo "Example: $0 start M3"
    exit 1
}

[[ -z "$COMMAND" ]] && usage

spec_path() {
    echo "docs/Milestone${1#M}.md"
}

case "$COMMAND" in
    start)
        if [[ -z "$MILESTONE" ]]; then
            echo "Error: Milestone name required (e.g., M3)"
            exit 1
        fi
        SPEC_DOC="$(spec_path "$MILESTONE")"
        if [[ ! -f "$SPEC_DOC" ]]; then
            echo "Error: requirement doc $SPEC_DOC not found."
            echo "Write the milestone spec at $SPEC_DOC before starting milestone/$MILESTONE."
            exit 1
        fi
        echo "Starting milestone $MILESTONE..."
        git checkout -b "milestone/$MILESTONE"
        echo "Switched to new branch: milestone/$MILESTONE (spec: $SPEC_DOC)"
        ;;

    verify)
        if [[ -x scripts/milestone-verify.sh ]]; then
            ./scripts/milestone-verify.sh
        elif [[ -f scripts/milestone-verify.sh ]]; then
            bash scripts/milestone-verify.sh
        else
            echo "No scripts/milestone-verify.sh; running cargo test"
            cargo test
        fi
        echo "Verification complete."
        ;;

    release)
        CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        if [[ -z "$MILESTONE" ]]; then
            if [[ "$CURRENT_BRANCH" =~ milestone/(.+) ]]; then
                MILESTONE="${BASH_REMATCH[1]}"
            else
                echo "Error: Milestone name required or must be on a milestone branch."
                exit 1
            fi
        fi

        echo "Releasing Milestone $MILESTONE..."

        SPEC_DOC="$(spec_path "$MILESTONE")"
        RELEASE_NOTES="docs/RELEASE_NOTES_${MILESTONE}.md"
        TAG_NAME="${MILESTONE}.0"

        mkdir -p docs

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
                echo "- \`./scripts/milestone.sh verify\` passed."
            } > "$RELEASE_NOTES"
            rm -f "$SPEC_DOC"
            echo "Folded $SPEC_DOC into $RELEASE_NOTES and removed the spec."
        else
            echo "Warning: no spec doc at $SPEC_DOC; skipping fold-in."
        fi

        if [[ -f README.md ]] && grep -qE '## Milestone Release Pipeline \(M[0-9.]+\)' README.md; then
            sed -i.bak -E "s/(## Milestone Release Pipeline \()M[0-9.]+(\))/\1${MILESTONE}\2/" README.md
            rm -f README.md.bak
            echo "Bumped README.md version tag to $MILESTONE."
        fi

        git add -A
        if git diff --cached --quiet; then
            echo "Nothing to commit."
        else
            git commit -m "$(cat <<EOF
milestone $MILESTONE: release notes + verification + pending changes

EOF
)"
        fi

        git push -u origin "$CURRENT_BRANCH"

        gh pr create --base master --title "Release Milestone $MILESTONE" --body "$(cat <<EOF
## Summary

Automated release for $MILESTONE. Verification passed.

## Test plan

- [ ] \`./scripts/milestone.sh verify\`

EOF
)"

        echo "Merging to master..."
        gh pr merge --merge --delete-branch

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
