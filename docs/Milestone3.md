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

### 5.1 Rank vectors

Unit tests: published 7-card Hold'em fixtures (royal, wheel straight
flush, full house vs trips, chopped boards). Property tests: `HandRank`
monotonicity, split pots when both 5-card maxima are equal, no card
reuse.

### 5.2 Equity dataset

Add `data/holdem_sample.txt` (and a larger `data/holdem_full_db.txt` if a
source is available) in the **same** `hero villain [board] equity` layout
as §10, with 2-card hands / Hold'em ranges.

Sources, in preference order:

1. Pokerstove / `ps-eval` Hold'em mode, if the local binary supports it.
2. Another established equity tool (Equilab export, or a small Python
   golden set generated once and checked in).
3. Cross-check against the CPU evaluator of a known-good crate only as a
   last resort, and record the source in the dataset header comment.

Accuracy target: **100% pass rate at 0.1 tolerance**, same as Omaha.
Run:

```bash
cargo run --release --bin validation -- \
  --game holdem --input data/holdem_sample.txt --tolerance 0.1 --backend cpu

cargo run --release --bin validation -- \
  --game holdem --input data/holdem_sample.txt --tolerance 0.1 --backend auto
```

Omaha full-db verification remains mandatory on `./scripts/milestone.sh verify`.

### 5.3 Cross-game isolation

A test that Omaha `AA` expansion still yields 4-card hands, and Hold'em
`AA` yields exactly 6 combos. Parser tests for `AKs`, `AKo`, `22+`.

---

## 6. Downstream: `gto`

After M3, `gto-eval` should call `plo_eval_gpu::evaluate_holdem_hand` (or
equivalent) and drop its local 21-combo loop. That change is **not** part
of this repo's M3 exit criteria, but the public function must be stable
enough for a path dependency bump in the same week.

---

## 7. Implementation plan

- [ ] **`Game` + `Hand` arity** — constructors, rejects, `random_hand(game)`.
- [ ] **`evaluate_holdem_hand`** — 21-combo river path on `evaluate_5_cards`;
      unit + property tests.
- [ ] **Hold'em range parser** — exact, pair, `s`/`o`, plus/dash; dead cards.
- [ ] **CPU equity** — thread `game` through HvsH / HvsR / RvsR / Auto mode
      (exhaustive vs MC) without changing Omaha numbers.
- [ ] **GPU** — `hand_len` on `GpuCaseInput`, padded 2-card hands, best-of-21
      in WGSL; CPU/GPU parity tests.
- [ ] **`validation --game`** — holdem sample dataset; Omaha default unchanged.
- [ ] **Docs** — README examples for Hold'em; §4.2 / §7.2 / §7.3 in
      `PokerHandEvaluator.md` describe both hole-card counts and both
      showdown rules.
- [ ] **Regression** — `cargo test` + Omaha `pokerstove_full_db.txt` still
      100% at 0.1.

## Success metrics

- **Correctness**: Hold'em sample set 100% within 0.1; Omaha full-db
  unchanged; CPU and GPU Hold'em agree within the GPU test tolerance.
- **API**: One `Game` switch; no second card type; no duplicated 5-card
  rank logic on CPU or GPU.
- **Parser**: `AA` / `AKs` / `AKo` / `AhKd` / `22+` expand to the standard
  combo counts (6 / 4 / 12 / 1 / 78 for `22+` through `AA`).
- **Perf**: Hold'em river HvsH is not slower than Omaha river HvsH on the
  same machine (it should be faster). No new Omaha speed regression vs
  `docs/test_results.log` flop/preflop targets.
