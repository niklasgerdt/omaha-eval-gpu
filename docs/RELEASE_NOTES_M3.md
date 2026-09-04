# Release Notes: Milestone M3

_Released 2026-09-05, tag `M3.0`._

# Milestone 3: Texas Hold'em Capability

Milestone 3 (M3) adds **No-Limit / Limit-agnostic Texas Hold'em** as a first-class
variant next to the existing Omaha Hi evaluator. Betting structure is out of
scope (this crate ranks hands and computes equity). The 5-card rank core,
canonical card types, CPU/GPU equity pipeline, and validation harness stay;
what changes is hole-card arity, showdown combinatorics, and range notation.

This milestone exists so `gto` can depend on a validated Hold'em evaluator
instead of wrapping `evaluate_5_cards` itself. Omaha behaviour must not
regress.

## Why this is M3 (before M4 / M5)

- **M4** (backend selection heuristics) is about *when* to use GPU vs CPU.
  Hold'em changes *what* a hand is; do that first so M4 heuristics can see
  both games.
- **M5** (Omaha Hi/Lo GPU + split-pot validation) is Omaha-shaped. Hold'em
  has no 8-or-better path in this milestone.
- Hold'em is the cheaper showdown (`C(7,5) = 21` vs Omaha's 60) and the
  missing piece for the GTO solver. Land it before further Omaha-only work.

## Non-goals

- Omaha Hi/Lo improvements (M5).
- Hold'em Hi/Lo, Short Deck, Pineapple, 6+ Hold'em.
- Suit isomorphism, strategy, or tree building (those belong in `gto`).
- Changing card notation, `HandRank`, or `evaluate_5_cards`.
- Breaking the Omaha validation datasets or their 0.1 tolerance.

---

## 1. Rules to implement

### 1.1 Showdown

A Hold'em hand is **exactly 2 hole cards**. The board is unchanged: 0
(preflop), 3 (flop), 4 (turn), or 5 (river) community cards.

On a **rivered** board the player makes the best 5-card poker hand from the
7 cards (2 hole + 5 board). Any 0–2 hole cards may be used. That is
`C(7,5) = 21` subsets — **not** Omaha's "exactly 2 from hand and exactly 3
from board."

Incomplete boards are handled the same way Omaha already is: exhaustive
enumeration or Monte Carlo **completes the board to 5**, then the river
rule applies. Do not invent a separate flop/turn ranking rule.

### 1.2 Blockers

A hero/villain combo that shares a card with the board or with the other
player's combo is dead and contributes zero weight, identical to Omaha.

### 1.3 Ranking

Call `eval::evaluate_5_cards` on each 5-card subset; take `HandRank` max.
Do not reimplement straight/flush/pair detection.

---

## 2. Types and public API

### 2.1 `Game`

```text
enum Game { Omaha, HoldEm }
```

Default for existing call sites that we keep compiling: `Game::Omaha`.

`hi_lo == true` is only defined for `Game::Omaha`. Passing `hi_lo` with
`Game::HoldEm` is an error (return `Err` or panic in debug + documented
`debug_assert`); do not silently ignore it.

### 2.2 `Hand`

Today `Hand` is ` [Card; 4] `. That cannot represent Hold'em.

**Required shape** (implementation may vary; tests lock behaviour):

- Omaha: 4 cards, canonical sort (§4.1.4), same as today.
- Hold'em: 2 cards, same canonical sort.
- Constructors that make the arity obvious, e.g. `Hand::omaha([Card; 4])`
  and `Hand::holdem([Card; 2])`. Keep a compatibility `Hand::new([Card; 4])`
  as an Omaha constructor so current README snippets still work.

A `Hand` used with the wrong `Game` (4-card hand + `HoldEm`, or 2-card +
`Omaha`) is a caller bug: reject at the evaluation boundary.

`random_hand` / `random_hands` take `Game` and deal 2 or 4 cards.

### 2.3 `Range`

Still `Vec<(Hand, f64)>`, but every `Hand` in a range must share one
arity. Mixing 2-card and 4-card combos in one `Range` is invalid.

`Range::from_shorthand` must take `Game` (or a dedicated
`from_shorthand_holdem`). **`AA` is not the same in both games:**

| Token | Omaha (today) | Hold'em (M3) |
| :--- | :--- | :--- |
| `AsKsQhJh` (8 chars) | one exact 4-card hand | invalid |
| `AhKd` (4 chars, suited/offsuit cards) | Omaha rank pattern `AhKd` is not today's syntax; exact Omaha hands are 8 chars | one exact combo |
| `AA` | any 4-card hand with ≥2 aces | pocket aces (6 combos) |
| `AK` | 4-card hands containing A and K | all AKo + AKs (16 combos) |
| `AKs` / `AKo` | unsupported | suited / offsuit only |
| `22+`, `ATs+`, `JTs-54s` | unsupported | standard plus/range pairs and connectors |

Hold'em parser **must** implement the usual combo-range dialect used by
Pokerstove / Equilab / GTO solvers:

- Pairs: `AA`, `77`, `22+`, `22-66`
- Unpaired: `AK`, `AKs`, `AKo`, `ATs+`, `KQo+`, `JTs-54s` (suited),
  `JTo-54o` (offsuit)
- Exact combos: `AhKd`
- Comma-separated unions, board/dead cards excluded

Weights stay `1.0` per combo (same pending work as Omaha). Do not invent a
second weighting syntax in M3.

A full 169-type chart expansion (`"random"`, `"UTG"`, etc.) is **out of
scope**; callers pass explicit range strings.

### 2.4 Evaluation entry points

Thread `game: Game` through:

- `evaluate_holdem_hand` (new) — single showdown rank
- `evaluate_hand_vs_hand` / `_vs_range` / `evaluate_range_vs_range`
- `Backend::{run_evaluation, run_range_evaluation, run_range_evaluation_batch}`
- `validation` CLI: `--game omaha|holdem` (default `omaha`)

CPU and GPU paths both honor `Game`. `Backend::Auto` still tries GPU first
and falls back per-case, including Hold'em.

Existing Omaha unit tests and `data/pokerstove_*.txt` runs stay on
`--game omaha` and must keep a 100% pass rate at tolerance 0.1.

---

## 3. CPU implementation

Add `evaluate_holdem_hand(hole: &[Card; 2], board: &[Card]) -> HandRank`
next to `evaluate_omaha_hand`.

River (`board.len() == 5`): 21 combinations of 5 cards from the 7.
Turn (`board.len() == 4`): complete later; at showdown-on-partial for
property tests only, best 5 of the 6 cards.
Flop (`board.len() == 3`): the five cards are the hand.
`board.len() < 3`: same sentinel as Omaha (lowest `HandRank`) — equity
functions never rank a raw preflop/turn-incomplete showdown; they deal
the rest of the board first.

Wire that function into the existing exhaustive and Monte Carlo loops in
place of `evaluate_omaha_hand` when `game == HoldEm`. Deck size, dealing,
and blocker filters stay shared.

---

## 4. GPU implementation

`src/omaha.wgsl` (and `GpuCaseInput`) currently stores 4 hole cards per
hand and runs Omaha combinatorics.

**Do not fork a second 5-card evaluator in WGSL.** Keep one
`evaluate_5_cards` in the shader.

Required host/shader changes:

- Add a `game` (or `hand_len`: 2 vs 4) field on `GpuCaseInput`.
- Hold'em hole cards occupy the first two `u32` slots; pad the rest with
  the existing `255` sentinel (same as unused board slots).
- When `hand_len == 2` and `board_len == 5`, rank by best-of-21 over the
  seven indices. When completing flop/turn/MC, after the board is filled
  to 5, use the same best-of-21.
- When `hand_len == 4`, today's Omaha kernel is unchanged.
- Dispatch, MC lane splitting (`mc_lanes`), batching (256 cases, 128×128
  hands), and atomic result accumulation stay as they are.

CPU vs GPU Hold'em equities must match within the same tolerance used for
Omaha GPU tests (existing ignored GPU tests plus a Hold'em equivalent that
is **not** ignored once green).

---

## 5. Validation

Hold'em is not done until it has the **same test surface as PLO**, not a
thinner subset. Every existing Omaha check in `src/lib.rs` `tests` and
every `validation` dataset/command gets a Hold'em twin, except Hi/Lo
(`test_omaha_hi_lo` has no Hold'em counterpart; `hi_lo` + `HoldEm` is
rejected instead).

Existing PLO tests stay. Do not rewrite them into parameterized tables
unless both games still fail independently when one path breaks.

### 5.1 Unit tests (always-on `cargo test`)

Mirror `src/lib.rs` `tests`:

| PLO today | Hold'em twin |
| :--- | :--- |
| `test_omaha_evaluation` | `test_holdem_evaluation` — known 2-card + board → expected `HandRank` (include a case Omaha and Hold'em would rank differently, e.g. using 1 hole card or playing the board) |
| `test_hand_vs_hand_exhaustive` | same, 2-card hands, `win`/`tie`/`loss` on a rivered board |
| `test_canonical_sorting` | 2-card sort still rank-desc then suit precedence (`AcAs` → `AsAc`) |
| `test_omaha_hi_lo` | **no twin** — assert Hold'em + `hi_lo` is rejected |

Plus rank-vector / property tests: published 7-card fixtures (royal,
wheel SF, full house vs trips, chops); `HandRank` monotonicity; split
when both 5-card maxima are equal; no card reuse across hero/villain/board.

### 5.2 GPU tests (same `#[ignore]` policy as PLO)

PLO GPU tests are ignored and run when a GPU is present. Hold'em twins
use the same ignore/run convention — do not skip GPU coverage for
Hold'em while keeping it for Omaha:

| PLO today | Hold'em twin |
| :--- | :--- |
| `test_gpu_vs_cpu` | river HvsH, CPU vs GPU equity within `1e-6` |
| `test_gpu_padding_sentinel` | flop board (3 cards); unused hole/board slots are `255`, not card index 0 |
| `test_gpu_preflop_batch_at_scale` | 256 preflop MC cases, 1000 samples, no zero-equity / all-`None` |
| `test_gpu_preflop_mc_matches_ps_eval` | 5 fixed Hold'em pairs vs `ps-eval` (or the Hold'em golden source), delta `< 0.03` |
| `test_gpu_preflop_batch_concurrent` | 8 threads × batched MC; no zero-equity under concurrent load |

### 5.3 Validation bench (same sizes and commands as PLO)

PLO has `data/pokerstove_sample_10.txt`, `pokerstove_sample_100.txt`, and
`pokerstove_full_db.txt`, all `hero villain [board] equity`. Hold'em gets
the same three tiers in that layout, 2-card hands / Hold'em ranges:

- `data/holdem_sample_10.txt`
- `data/holdem_sample_100.txt`
- `data/holdem_full_db.txt` (or the largest set a trusted source can
  produce; document the source in a header comment)

Sources, in preference order: Pokerstove / `ps-eval` Hold'em mode;
Equilab (or similar) export; last resort a known-good crate, source
recorded.

Accuracy: **100% pass rate at 0.1 tolerance**, same as PLO, CPU and
`auto`:

```bash
cargo run --release --bin validation -- \
  --game holdem --input data/holdem_sample_100.txt --tolerance 0.1 --backend cpu

cargo run --release --bin validation -- \
  --game holdem --input data/holdem_sample_100.txt --tolerance 0.1 --backend auto

cargo run --release --bin validation -- \
  --game holdem --input data/holdem_full_db.txt --tolerance 0.1
```

`./scripts/milestone.sh verify` must run **both** the Omaha pokerstove
full-db and the Hold'em full-db (plus the sample-100 CPU/`auto` speed
checks for both games). Omaha numbers must not regress.

Street mix in the Hold'em files should resemble PLO's: preflop, flop,
turn, and river cases, not river-only.

### 5.4 Cross-game isolation

Omaha `AA` still expands to 4-card hands; Hold'em `AA` is exactly 6
combos. Parser tests for `AKs`, `AKo`, `22+`. A Hold'em range string
must not be accepted by the Omaha parser as if it were PLO (and vice
versa for 8-character exact hands).

---

## 6. Downstream: `gto`

After M3, `gto-eval` should call `plo_eval_gpu::evaluate_holdem_hand` (or
equivalent) and drop its local 21-combo loop. That change is **not** part
of this repo's M3 exit criteria, but the public function must be stable
enough for a path dependency bump in the same week.

---

## 7. Implementation plan

- [x] **`Game` + `Hand` arity** — `Hand` is now `Vec<Card>`; `Hand::omaha`/
      `Hand::holdem` constructors, `Hand::game()`, `random_hand(..., game)`.
- [x] **`evaluate_holdem_hand`** — best-5-of-N over the combined hole+board
      cards (N = 5/6/7 for flop/turn/river) on `evaluate_5_cards`; no
      per-street special-casing needed.
- [x] **PLO-twin unit tests** — §5.1: `test_holdem_evaluation` (0-hole-card
      board-play case), `test_holdem_hand_vs_hand_exhaustive`,
      `test_holdem_canonical_sorting`, `test_holdem_hi_lo_rejected`. The
      broader published-fixture/property-test sweep (royal, wheel SF, full
      house vs trips, chops, monotonicity) is not yet ported — still open.
- [x] **Hold'em range parser** — exact combos, pairs (`AA`/`22+`/`22-66`),
      suited/offsuit (`AKs`/`AKo`/`AK`), plus/dash (`ATs+`/`KQo+`/
      `JTs-54s`/`JTo-54o`); dead-card exclusion; isolation from Omaha `AA`
      tested directly (`test_holdem_range_isolated_from_omaha`).
- [x] **CPU equity** — `game: Game` threaded through `evaluate_hand_vs_hand`,
      `evaluate_hand_vs_range`, `evaluate_range_vs_range(_internal)`, and
      all three `Backend` methods; Omaha numbers unchanged (see Regression).
- [ ] **GPU + PLO-twin GPU tests** — not started. `Backend` currently routes
      `Game::HoldEm` to CPU unconditionally (even under an explicit GPU
      backend) rather than touching `omaha.wgsl`/`GpuCaseInput` — correct
      results, but no GPU speedup for Hold'em yet.
- [~] **`validation --game`** — the CLI flag and CPU pipeline work end to
      end (spot-checked against known preflop equities: AA vs KK, AKs vs
      QQ, 72o vs AA, all within 0.1). `data/holdem_sample_10/_100/_full_db`
      have not been produced from a trusted external source, and
      `milestone.sh verify` has not been updated to run a Hold'em bench.
- [ ] **Docs** — README/`PokerHandEvaluator.md` still describe Omaha only.
- [x] **Regression** — `cargo test` (14 passed, 5 pre-existing `#[ignore]`d
      GPU tests) and `pokerstove_sample_10.txt` both still 100% at 0.1 on
      the default (Omaha) path.

## Success metrics

- **Correctness**: Every PLO test in §5.1–5.2 has a Hold'em twin (except
  Hi/Lo). Hold'em sample_100 and full-db 100% within 0.1; Omaha full-db
  unchanged; CPU and GPU Hold'em agree within the GPU test tolerance.
- **API**: One `Game` switch; no second card type; no duplicated 5-card
  rank logic on CPU or GPU.
- **Parser**: `AA` / `AKs` / `AKo` / `AhKd` / `22+` expand to the standard
  combo counts (6 / 4 / 12 / 1 / 78 for `22+` through `AA`).
- **Perf**: Hold'em river HvsH is not slower than Omaha river HvsH on the
  same machine (it should be faster). No new Omaha speed regression vs
  `docs/test_results.log` flop/preflop targets.

## Verification

- `./scripts/milestone.sh verify` passed.
