# Release Notes: Milestone M2.3.C1

_Released 2026-09-05, tag `M2.3.C1.0`._

# Milestone 2.3.C1: Release Pipeline Hardening

## Overview
M2.3.C1 closes gaps in `scripts/milestone.sh` found while actually running it:
`release` never committed or pushed anything, so it silently assumed work was
already on the remote before creating a PR. This milestone also adds the
process guarantees needed to keep milestone specs, release notes, and
README.md in sync without manual bookkeeping.

## Objectives

### 1. Fix `release`'s missing commit/push
- `release` stages (`git add -A`) and commits any pending changes before
  touching GitHub, guarded so an empty diff doesn't fail the `set -e` script.
- `release` pushes the current branch (`git push -u origin <branch>`) before
  `gh pr create` — previously `gh pr create` had nothing to point at.

### 2. Gate `start` on a written spec
- `start <name>` requires `docs/Milestone{N}.md` to exist first (`M3` →
  `docs/Milestone3.md`: strip the leading `M`, prepend `Milestone`, matching
  the existing file-naming convention). No spec, no branch.

### 3. Fold the spec into release notes on `release`
- `release` builds `docs/RELEASE_NOTES_M{name}.md` from the milestone's spec
  doc plus a verification appendix (test suite status, accuracy tolerance,
  pointer to `docs/test_results.log`, release date, git tag), then removes
  the spec doc (`git rm`) — the release notes become the permanent record,
  so the spec doesn't need to keep existing alongside it.
- If no spec doc exists (milestones that predate this convention, including
  this one — there was no `docs/Milestone2.3.C1.md` until this doc), the fold
  step warns and skips instead of failing the release.

### 4. Mechanical README.md version bump on `release`
- `release` bumps the `## Milestone Release Pipeline (Mx.y)` heading in
  README.md to the new milestone name. That's the limit of what the script
  rewrites unattended; broader prose (feature lists, benchmark numbers)
  still needs a human/Claude review pass before the release commit.

## Implementation Plan
- [x] `release`: commit pending changes, guarded for the no-op case.
- [x] `release`: push branch before `gh pr create`.
- [x] `start`: require `docs/Milestone{N}.md`, fail fast if missing.
- [x] `release`: fold spec into `docs/RELEASE_NOTES_M{name}.md`, `git rm` the spec.
- [x] `release`: bump README's version tag.
- [x] `docs/MilestoneReleasePipeline.md` updated to describe all of the above.

## Success Metrics
- Running `start Mx` → `verify` → `release` end-to-end produces a merged PR
  and a pushed tag without any manual `git commit`/`git push` in between.
- `start` refuses to create a branch with no matching spec doc.
- A released milestone leaves behind `docs/RELEASE_NOTES_M{name}.md` and no
  `docs/Milestone{N}.md` spec file.

## Verification

- Full suite: `cargo test` passed.
- Accuracy: 0.1 tolerance against `data/pokerstove_full_db.txt` (see `docs/test_results.log`).
- Performance: CPU/GPU benchmark timings in `docs/test_results.log`.
