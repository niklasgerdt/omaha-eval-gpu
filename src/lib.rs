//! Pot-Limit Omaha (PLO) hand evaluation and equity calculation.
//!
//! - [`Card`], [`Hand`] (4 cards), [`Board`] (0/3/4/5 cards) and [`Range`]
//!   (weighted list of hands) are the core domain types; see
//!   `docs/PokerHandEvaluator.md` §4 for notation and canonical ordering.
//! - [`eval::evaluate_5_cards`] / [`evaluate_omaha_hand`] rank a single
//!   showdown on the CPU; `eval_fast` adapts that logic to the flat
//!   card-index representation the GPU shader also uses.
//! - [`evaluate_hand_vs_hand`], [`evaluate_hand_vs_range`] and
//!   [`evaluate_range_vs_range`] compute equity for CPU-only callers;
//!   [`Backend`] wraps the same operations with GPU routing and fallback.
//! - The `gpu` module holds the `wgpu` compute pipeline
//!   (`src/omaha.wgsl`) that mirrors the CPU evaluator for exhaustive
//!   river evaluation and parallelized Monte Carlo sampling on earlier
//!   streets.

pub mod eval;
pub mod eval_fast;
pub mod gpu;

use crate::eval::HandRank;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

/// A card suit. Ordered `Spades > Hearts > Diamonds > Clubs` per §4.1.4,
/// which breaks ties when two cards share a [`Rank`] in canonical ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Suit {
    Spades = 0,
    Hearts = 1,
    Diamonds = 2,
    Clubs = 3,
}

impl PartialOrd for Suit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Suit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // §4.1.4: Spades > Hearts > Diamonds > Clubs
        // Our enum is defined in this order, but we want descending precedence.
        // Or if we define Spades=0, Hearts=1, etc., then Spades is "smallest" numerically.
        // Let's check how we want to sort. Rank descending, then Suit precedence.
        // "Spades > Hearts > Diamonds > Clubs"
        (*self as u8).cmp(&(*other as u8))
    }
}

/// A card rank, `Two` through `Ace`. Derives `Ord` so higher ranks compare
/// greater, matching poker hand strength (used directly by [`eval`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Rank {
    Two = 2,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Rank {
    pub fn all() -> impl Iterator<Item = Rank> {
        [
            Rank::Two,
            Rank::Three,
            Rank::Four,
            Rank::Five,
            Rank::Six,
            Rank::Seven,
            Rank::Eight,
            Rank::Nine,
            Rank::Ten,
            Rank::Jack,
            Rank::Queen,
            Rank::King,
            Rank::Ace,
        ]
        .into_iter()
    }

    /// Parses the rank character of §4.1.1 notation (`2`-`9`, `T`, `J`,
    /// `Q`, `K`, `A`; case-insensitive). Inverse of [`Rank::to_char`].
    pub fn from_char(c: char) -> Option<Rank> {
        match c.to_ascii_uppercase() {
            '2' => Some(Rank::Two),
            '3' => Some(Rank::Three),
            '4' => Some(Rank::Four),
            '5' => Some(Rank::Five),
            '6' => Some(Rank::Six),
            '7' => Some(Rank::Seven),
            '8' => Some(Rank::Eight),
            '9' => Some(Rank::Nine),
            'T' => Some(Rank::Ten),
            'J' => Some(Rank::Jack),
            'Q' => Some(Rank::Queen),
            'K' => Some(Rank::King),
            'A' => Some(Rank::Ace),
            _ => None,
        }
    }

    /// Renders the rank as its §4.1.1 notation character. Inverse of
    /// [`Rank::from_char`]; used by [`Card`]'s `Display` impl.
    pub fn to_char(self) -> char {
        match self {
            Rank::Two => '2',
            Rank::Three => '3',
            Rank::Four => '4',
            Rank::Five => '5',
            Rank::Six => '6',
            Rank::Seven => '7',
            Rank::Eight => '8',
            Rank::Nine => '9',
            Rank::Ten => 'T',
            Rank::Jack => 'J',
            Rank::Queen => 'Q',
            Rank::King => 'K',
            Rank::Ace => 'A',
        }
    }
}

impl Suit {
    /// All four suits, in enum declaration order (`Spades` first).
    pub fn all() -> impl Iterator<Item = Suit> {
        [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs].into_iter()
    }

    /// Parses the suit character of §4.1.1 notation (`s`, `h`, `d`, `c`;
    /// case-insensitive). Inverse of [`Suit::to_char`].
    pub fn from_char(c: char) -> Option<Suit> {
        match c.to_ascii_lowercase() {
            's' => Some(Suit::Spades),
            'h' => Some(Suit::Hearts),
            'd' => Some(Suit::Diamonds),
            'c' => Some(Suit::Clubs),
            _ => None,
        }
    }

    /// Renders the suit as its §4.1.1 notation character. Inverse of
    /// [`Suit::from_char`]; used by [`Card`]'s `Display` impl.
    pub fn to_char(self) -> char {
        match self {
            Suit::Spades => 's',
            Suit::Hearts => 'h',
            Suit::Diamonds => 'd',
            Suit::Clubs => 'c',
        }
    }
}

/// A single playing card. `Ord` sorts by the canonical §4.1.4 order (rank
/// descending, then suit precedence), which [`Hand::new`] and [`Board::new`]
/// rely on to normalize storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Card {
    pub rank: Rank,
    pub suit: Suit,
}

impl PartialOrd for Card {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Card {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // §4.1.4: Primary: Rank descending, Secondary: Suit precedence (Spades > Hearts > Diamonds > Clubs)
        match other.rank.cmp(&self.rank) {
            std::cmp::Ordering::Equal => self.suit.cmp(&other.suit),
            ord => ord,
        }
    }
}

impl Card {
    pub fn new(rank: Rank, suit: Suit) -> Self {
        Self { rank, suit }
    }

    /// Parses §4.1.1 notation, e.g. `"As"` (Ace of Spades). Returns `None`
    /// for anything other than exactly rank-char + suit-char. Inverse of
    /// the `Display` impl below.
    pub fn from_str(s: &str) -> Option<Self> {
        if s.len() != 2 {
            return None;
        }
        let mut chars = s.chars();
        let r = Rank::from_char(chars.next()?)?;
        let s = Suit::from_char(chars.next()?)?;
        Some(Card::new(r, s))
    }
}

/// Renders §4.1.1 notation, e.g. `Card::new(Rank::Ace, Suit::Spades)` ->
/// `"As"`. Inverse of [`Card::from_str`].
impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.rank.to_char(), self.suit.to_char())
    }
}

/// A full, unshuffled 52-card deck in `Suit::all()` x `Rank::all()` order.
/// The single source of deck construction shared by [`Range::from_shorthand`]
/// and [`random_hands`], so deck order/composition only needs to be right
/// in one place.
pub fn full_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for s in Suit::all() {
        for r in Rank::all() {
            deck.push(Card::new(r, s));
        }
    }
    deck
}

/// An Omaha hole-card hand: exactly 4 cards, always held in canonical
/// (§4.1.4) order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hand(pub [Card; 4]);

impl Hand {
    /// Sorts `cards` into canonical order and wraps them as a `Hand`.
    pub fn new(mut cards: [Card; 4]) -> Self {
        cards.sort();
        Self(cards)
    }
}

/// The community cards: 0 (pre-flop), 3 (flop), 4 (turn) or 5 (river)
/// cards, held in canonical (§4.1.4) order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Board(pub Vec<Card>);

impl Board {
    /// Sorts `cards` into canonical order and wraps them as a `Board`.
    pub fn new(mut cards: Vec<Card>) -> Self {
        cards.sort();
        Self(cards)
    }
}

/// A weighted collection of possible hands, e.g. a villain's estimated
/// holding range. Each `(Hand, f64)` pair is a hand and its relative
/// weight; weights need not sum to 1 (evaluators normalize by total
/// weight, not count).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    pub hands: Vec<(Hand, f64)>,
}

impl Range {
    /// Parses §4.1.3 shorthand range notation into a `Range`, expanding
    /// each comma-separated part into one or more concrete hands (all
    /// with weight `1.0` — per-hand weighting is not yet supported):
    ///
    /// - An exact 8-character hand, e.g. `"AsKsQhJh"`.
    /// - A 2- or 4-rank pattern, e.g. `"AA"` (any hand with 2+ Aces) or
    ///   `"AKQJ"` (any hand containing all four ranks) — expanded into every
    ///   matching 4-card combo from the remaining deck.
    ///
    /// `dead_cards` are excluded from both the deck used for pattern
    /// expansion and from any resulting hand (hands that would reuse a
    /// dead card are dropped rather than erroring).
    pub fn from_shorthand(s: &str, dead_cards: &[Card]) -> Result<Self, String> {
        let mut hands = Vec::new();
        let parts = s.split(',').map(|p| p.trim());

        for part in parts {
            if part.is_empty() {
                continue;
            }

            // §4.1.3 Range Notation
            // 1. Exact hand (8 chars, e.g. "AsKsQhJh")
            if part.len() == 8 {
                let mut cards = [Card::new(Rank::Two, Suit::Spades); 4];
                let mut valid = true;
                for i in 0..4 {
                    if let Some(c) = Card::from_str(&part[i * 2..i * 2 + 2]) {
                        cards[i] = c;
                    } else {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    if !cards.iter().any(|c| dead_cards.contains(c)) {
                        hands.push((Hand::new(cards), 1.0));
                    }
                    continue;
                }
            }

            // 2. Rank Pattern (e.g. "AA" or "AKQJ")
            if part.len() == 2 || part.len() == 4 {
                let ranks: Vec<Rank> = part
                    .chars()
                    .map(|c| Rank::from_char(c).ok_or(format!("Invalid rank: {}", c)))
                    .collect::<Result<Vec<_>, _>>()?;

                let deck: Vec<Card> = full_deck()
                    .into_iter()
                    .filter(|c| !dead_cards.contains(c))
                    .collect();

                let mut combos = Vec::new();
                for c1_idx in 0..deck.len() {
                    for c2_idx in c1_idx + 1..deck.len() {
                        for c3_idx in c2_idx + 1..deck.len() {
                            for c4_idx in c3_idx + 1..deck.len() {
                                let h = [deck[c1_idx], deck[c2_idx], deck[c3_idx], deck[c4_idx]];
                                let mut r_counts = std::collections::HashMap::new();
                                for c in &h {
                                    *r_counts.entry(c.rank).or_insert(0) += 1;
                                }

                                let mut match_found = true;
                                if ranks.len() == 2 && ranks[0] == ranks[1] {
                                    // "AA" -> at least two Aces
                                    if *r_counts.get(&ranks[0]).unwrap_or(&0) < 2 {
                                        match_found = false;
                                    }
                                } else {
                                    // "AK" or "AKQJ" -> at least one of each rank
                                    for r in &ranks {
                                        if *r_counts.get(r).unwrap_or(&0) < 1 {
                                            match_found = false;
                                            break;
                                        }
                                    }
                                }

                                if match_found {
                                    combos.push((Hand::new(h), 1.0));
                                }
                            }
                        }
                    }
                }
                hands.extend(combos);
                continue;
            }

            // 3. Simple list of cards (catch-all for cases like "6dJh8h9h")
            if part.len() % 2 == 0 {
                let mut cards = Vec::new();
                let mut valid = true;
                for i in 0..(part.len() / 2) {
                    if let Some(c) = Card::from_str(&part[i * 2..i * 2 + 2]) {
                        cards.push(c);
                    } else {
                        valid = false;
                        break;
                    }
                }
                if valid && cards.len() == 4 {
                    if !cards.iter().any(|c| dead_cards.contains(c)) {
                        let mut arr = [Card::new(Rank::Two, Suit::Spades); 4];
                        arr.copy_from_slice(&cards);
                        hands.push((Hand::new(arr), 1.0));
                    }
                    continue;
                }
            }

            return Err(format!("Unsupported range notation: '{}'", part));
        }

        if hands.is_empty() {
            return Err("Empty range".to_string());
        }

        Ok(Self { hands })
    }
}

/// How an equity calculation is carried out.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalMode {
    /// Exhaustive when the remaining board is small enough to enumerate
    /// cheaply (see `evaluate_hand_vs_hand`'s `board.len() >= 3` check),
    /// Monte Carlo (10,000 samples) otherwise.
    Auto,
    /// Enumerate every possible completion of the board exactly. Exact but
    /// requires `board.len() >= 3` (river needs 0 completions, turn needs
    /// C(46,1), flop needs C(45,2) — pre-flop/turn-less boards are not
    /// supported by the CPU exhaustive path, only by GPU MC or CPU MC).
    Exhaustive,
    /// Deal `samples` random completions of the board with the given RNG
    /// `seed` and average the outcomes. The only mode usable when fewer
    /// than 3 board cards are known (see [`evaluate_hand_vs_hand`]).
    MonteCarlo { samples: u64, seed: u64 },
}

/// Selects which compute backend(s) an equity calculation may use.
///
/// `Auto` always tries the GPU (`src/omaha.wgsl`) first and falls back to
/// the CPU evaluator per-case whenever the GPU is unavailable (no adapter,
/// e.g. headless CI) or returns `None` for a given case. Omaha Hi/Lo is not
/// implemented on the GPU, so `hi_lo` calculations always run on CPU
/// regardless of the selected backend. `Cuda`/`Vulkan`/`Metal` all currently
/// dispatch to the same `wgpu`-selected adapter as `Auto`'s GPU path (wgpu
/// picks the concrete backend at adapter-request time); they exist as
/// explicit backend-pinning hooks for future per-backend tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Backend {
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

impl Backend {
    /// Whether this backend can be used at all. GPU backends do not check
    /// hardware availability here — that's discovered lazily on first GPU
    /// call and surfaces as a per-call `None`/CPU-fallback instead.
    pub fn is_available(&self) -> bool {
        match self {
            Backend::Auto => true,
            Backend::Cpu => true,
            Backend::Cuda | Backend::Vulkan | Backend::Metal => true,
        }
    }

    /// Single hand-vs-hand equity, routed per [`Backend`]'s rules above.
    pub fn run_evaluation(
        &self,
        hero: &Hand,
        villain: &Hand,
        board: &Board,
        mode: &EvalMode,
        hi_lo: bool,
    ) -> Option<EquityResult> {
        if !self.is_available() {
            return None;
        }
        match self {
            Backend::Cpu => Some(evaluate_hand_vs_hand(
                hero.clone(),
                villain.clone(),
                board.clone(),
                mode.clone(),
                hi_lo,
            )),
            Backend::Auto => {
                // Try GPU first
                if let Some(res) = crate::gpu::run_gpu_evaluation(hero, villain, board, mode, hi_lo)
                {
                    return Some(res);
                }
                Some(evaluate_hand_vs_hand(
                    hero.clone(),
                    villain.clone(),
                    board.clone(),
                    mode.clone(),
                    hi_lo,
                ))
            }
            Backend::Cuda | Backend::Vulkan | Backend::Metal => {
                crate::gpu::run_gpu_evaluation(hero, villain, board, mode, hi_lo)
            }
        }
    }

    /// Range-vs-range equity for a single `(hero_range, villain_range,
    /// board, mode)` case, routed per [`Backend`]'s rules above. Prefer
    /// [`Backend::run_range_evaluation_batch`] when evaluating many cases —
    /// it amortizes GPU dispatch overhead across up to 256 cases per call.
    pub fn run_range_evaluation(
        &self,
        hero_range: &Range,
        villain_range: &Range,
        board: &Board,
        mode: &EvalMode,
        hi_lo: bool,
    ) -> EquityResult {
        match self {
            Backend::Cpu => evaluate_range_vs_range_internal(
                hero_range.clone(),
                villain_range.clone(),
                board.clone(),
                mode.clone(),
                hi_lo,
            ),
            Backend::Auto => {
                if !hi_lo {
                    if let Some(res) =
                        crate::gpu::run_gpu_range_evaluation(hero_range, villain_range, board, mode)
                    {
                        return res;
                    }
                }
                evaluate_range_vs_range_internal(
                    hero_range.clone(),
                    villain_range.clone(),
                    board.clone(),
                    mode.clone(),
                    hi_lo,
                )
            }
            _ => {
                if !hi_lo {
                    if let Some(res) =
                        crate::gpu::run_gpu_range_evaluation(hero_range, villain_range, board, mode)
                    {
                        return res;
                    }
                }
                evaluate_range_vs_range_internal(
                    hero_range.clone(),
                    villain_range.clone(),
                    board.clone(),
                    mode.clone(),
                    hi_lo,
                )
            }
        }
    }

    /// Range-vs-range equity for many independent cases at once. `hi_lo`
    /// applies uniformly to every case (all-CPU) since the GPU shader only
    /// implements Omaha Hi. For `!hi_lo`, cases are chunked into batches of
    /// up to 256 (the GPU shader's fixed per-dispatch case limit) and sent
    /// to the GPU together; any case the GPU can't or didn't resolve (e.g.
    /// no adapter available) falls back to the CPU individually, so a
    /// single bad/unsupported case never drags the whole batch to CPU.
    pub fn run_range_evaluation_batch(
        &self,
        cases: &[(Range, Range, Board, EvalMode)],
        hi_lo: bool,
    ) -> Vec<EquityResult> {
        if hi_lo {
            return cases
                .iter()
                .map(|(h, v, b, m)| {
                    evaluate_range_vs_range_internal(
                        h.clone(),
                        v.clone(),
                        b.clone(),
                        m.clone(),
                        hi_lo,
                    )
                })
                .collect();
        }

        match self {
            Backend::Cpu => cases
                .iter()
                .map(|(h, v, b, m)| {
                    evaluate_range_vs_range_internal(
                        h.clone(),
                        v.clone(),
                        b.clone(),
                        m.clone(),
                        hi_lo,
                    )
                })
                .collect(),
            Backend::Auto | Backend::Metal | Backend::Cuda | Backend::Vulkan => {
                let mut all_results = Vec::with_capacity(cases.len());
                for chunk in cases.chunks(256) {
                    let gpu_results = crate::gpu::run_gpu_range_evaluation_batch(chunk);
                    for (i, res) in gpu_results.into_iter().enumerate() {
                        if let Some(r) = res {
                            all_results.push(r);
                        } else {
                            let (h, v, b, m) = &chunk[i];
                            all_results.push(evaluate_range_vs_range_internal(
                                h.clone(),
                                v.clone(),
                                b.clone(),
                                m.clone(),
                                hi_lo,
                            ));
                        }
                    }
                }
                all_results
            }
        }
    }
}

/// Hi (and, if requested, Lo) equity for one hero-vs-villain or
/// range-vs-range calculation. `win + tie + loss == 1.0` for the Hi side
/// (subject to floating-point rounding); the `_low` fields mirror that for
/// the Omaha Hi/Lo low side and are `None` whenever `hi_lo` was `false` or
/// no qualifying low hand existed for either side. A single scalar equity
/// (e.g. for display) is `win + tie / 2.0` (Hi) and, if applicable,
/// `win_low.unwrap() + tie_low.unwrap() / 2.0` split against the Hi share
/// of the pot per standard Hi/Lo rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityResult {
    pub win: f64,
    pub tie: f64,
    pub loss: f64,
    pub win_low: Option<f64>,
    pub tie_low: Option<f64>,
    pub loss_low: Option<f64>,
    /// Number of board completions (exhaustive) or samples (Monte Carlo)
    /// this result was computed from.
    pub trial_count: u64,
    /// The mode actually used (with `Auto` already resolved to a concrete
    /// mode), not necessarily the mode passed in by the caller.
    pub mode: EvalMode,
    /// Reserved for a future statistical confidence interval on Monte Carlo
    /// results; always `None` today.
    pub confidence_interval: Option<(f64, f64)>,
}

/// Best 5-card Omaha Hi hand `hand` can make with `board`, trying every
/// legal "exactly 2 from hand + exactly 3 from board" combination.
/// `board.len()` must be 0 or in `3..=5`; boards with fewer than 3 cards
/// have no valid combination and return the lowest possible [`HandRank`].
pub fn evaluate_omaha_hand(hand: &Hand, board: &[Card]) -> HandRank {
    let mut best_rank = HandRank::HighCard(Rank::Two, Rank::Two, Rank::Two, Rank::Two, Rank::Two);

    if board.len() < 3 {
        return best_rank;
    }

    // Combinations of 2 cards from hand (4C2 = 6)
    let hand_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    // Combinations of 3 cards from board (C(board.len(), 3))
    for &(h1, h2) in &hand_indices {
        for i in 0..board.len() {
            for j in (i + 1)..board.len() {
                for k in (j + 1)..board.len() {
                    let cards = [hand.0[h1], hand.0[h2], board[i], board[j], board[k]];
                    let rank = crate::eval::evaluate_5_cards(&cards);
                    if rank > best_rank {
                        best_rank = rank;
                    }
                }
            }
        }
    }
    best_rank
}

/// Best Omaha Lo (8-or-better) hand `hand` can make with `board`, or `None`
/// if neither hand nor board combination yields a qualifying low (5 cards
/// ranked 8 or under, Ace playing low). The result is 5 ranks sorted
/// descending by *low* value (Ace = 1) for easy lexicographic comparison
/// between two low hands (lower is better).
pub fn evaluate_omaha_hand_low(hand: &Hand, board: &[Card]) -> Option<[Rank; 5]> {
    let mut best_low: Option<[Rank; 5]> = None;

    if board.len() < 3 {
        return None;
    }

    // Combinations of 2 cards from hand (4C2 = 6)
    let hand_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    for &(h1, h2) in &hand_indices {
        for i in 0..board.len() {
            for j in (i + 1)..board.len() {
                for k in (j + 1)..board.len() {
                    let cards = [hand.0[h1], hand.0[h2], board[i], board[j], board[k]];
                    if let Some(low_rank) = crate::eval::evaluate_5_cards_low(&cards) {
                        if let Some(best) = best_low {
                            let mut cur_vals = [0u32; 5];
                            let mut best_vals = [0u32; 5];
                            for n in 0..5 {
                                cur_vals[n] = if low_rank[n] == Rank::Ace {
                                    1
                                } else {
                                    low_rank[n] as u32
                                };
                                best_vals[n] = if best[n] == Rank::Ace {
                                    1
                                } else {
                                    best[n] as u32
                                };
                            }
                            if cur_vals < best_vals {
                                best_low = Some(low_rank);
                            }
                        } else {
                            best_low = Some(low_rank);
                        }
                    }
                }
            }
        }
    }
    best_low
}

/// CPU-only single hand-vs-hand equity (no GPU routing — see [`Backend`]
/// for that). `mode` resolves as follows:
/// - `Exhaustive` or `Auto` with `board.len() >= 3`: enumerates every legal
///   completion of the board exactly.
/// - `MonteCarlo { samples, seed }`: deals `samples` random completions.
/// - `Auto` with `board.len() < 3`: no exact enumeration path exists for
///   pre-flop/pre-flop-adjacent boards on CPU, so this falls back to
///   `MonteCarlo { samples: 10000, seed: 42 }`.
pub fn evaluate_hand_vs_hand(
    hero: Hand,
    villain: Hand,
    board: Board,
    mode: EvalMode,
    hi_lo: bool,
) -> EquityResult {
    let mut deck = [Card::new(Rank::Two, Suit::Spades); 52];
    let mut deck_len = 0;
    for s in Suit::all() {
        for r in Rank::all() {
            let card = Card::new(r, s);
            if !hero.0.contains(&card) && !villain.0.contains(&card) && !board.0.contains(&card) {
                deck[deck_len] = card;
                deck_len += 1;
            }
        }
    }

    match mode {
        EvalMode::Exhaustive | EvalMode::Auto if board.0.len() >= 3 => {
            let remaining = 5 - board.0.len();
            let mut hero_wins = 0;
            let mut villain_wins = 0;
            let mut ties = 0;
            let mut hero_wins_low = 0;
            let mut villain_wins_low = 0;
            let mut ties_low = 0;
            let mut low_count = 0;
            let mut count = 0;

            let mut process_full_board = |fb: &[Card]| {
                let hero_rank = evaluate_omaha_hand(&hero, fb);
                let villain_rank = evaluate_omaha_hand(&villain, fb);
                if hero_rank > villain_rank {
                    hero_wins += 1;
                } else if hero_rank < villain_rank {
                    villain_wins += 1;
                } else {
                    ties += 1;
                }

                if hi_lo {
                    let hero_low = evaluate_omaha_hand_low(&hero, fb);
                    let villain_low = evaluate_omaha_hand_low(&villain, fb);
                    match (hero_low, villain_low) {
                        (Some(h), Some(v)) => {
                            let mut h_vals = [0u32; 5];
                            let mut v_vals = [0u32; 5];
                            for i in 0..5 {
                                h_vals[i] = if h[i] == Rank::Ace { 1 } else { h[i] as u32 };
                                v_vals[i] = if v[i] == Rank::Ace { 1 } else { v[i] as u32 };
                            }
                            if h_vals < v_vals {
                                hero_wins_low += 1;
                            } else if h_vals > v_vals {
                                villain_wins_low += 1;
                            } else {
                                ties_low += 1;
                            }
                            low_count += 1;
                        }
                        (Some(_), None) => {
                            hero_wins_low += 1;
                            low_count += 1;
                        }
                        (None, Some(_)) => {
                            villain_wins_low += 1;
                            low_count += 1;
                        }
                        (None, None) => {}
                    }
                }
            };

            if remaining == 0 {
                process_full_board(&board.0);
                count += 1;
            } else if remaining == 1 {
                for i in 0..deck_len {
                    let mut fb = [Card::new(Rank::Two, Suit::Spades); 5];
                    fb[0..board.0.len()].copy_from_slice(&board.0);
                    fb[board.0.len()] = deck[i];
                    process_full_board(&fb);
                    count += 1;
                }
            } else if remaining == 2 {
                for i in 0..deck_len {
                    for j in i + 1..deck_len {
                        let mut fb = [Card::new(Rank::Two, Suit::Spades); 5];
                        fb[0..board.0.len()].copy_from_slice(&board.0);
                        fb[board.0.len()] = deck[i];
                        fb[board.0.len() + 1] = deck[j];
                        process_full_board(&fb);
                        count += 1;
                    }
                }
            }

            EquityResult {
                win: hero_wins as f64 / count as f64,
                tie: ties as f64 / count as f64,
                loss: villain_wins as f64 / count as f64,
                win_low: if hi_lo && low_count > 0 {
                    Some(hero_wins_low as f64 / low_count as f64)
                } else {
                    None
                },
                tie_low: if hi_lo && low_count > 0 {
                    Some(ties_low as f64 / low_count as f64)
                } else {
                    None
                },
                loss_low: if hi_lo && low_count > 0 {
                    Some(villain_wins_low as f64 / low_count as f64)
                } else {
                    None
                },
                trial_count: count as u64,
                mode: EvalMode::Exhaustive,
                confidence_interval: None,
            }
        }
        EvalMode::MonteCarlo { samples, seed } => {
            let mut rng = Pcg64::seed_from_u64(seed);
            let mut hero_wins = 0;
            let mut villain_wins = 0;
            let mut ties = 0;
            let mut hero_wins_low = 0;
            let mut villain_wins_low = 0;
            let mut ties_low = 0;
            let mut low_count = 0;
            let remaining = 5 - board.0.len();

            let mut deck_vec: Vec<Card> = deck[0..deck_len].to_vec();

            for _ in 0..samples {
                deck_vec.shuffle(&mut rng);
                let mut full_board = [Card::new(Rank::Two, Suit::Spades); 5];
                full_board[0..board.0.len()].copy_from_slice(&board.0);
                for i in 0..remaining {
                    full_board[board.0.len() + i] = deck_vec[i];
                }

                let hero_rank = evaluate_omaha_hand(&hero, &full_board);
                let villain_rank = evaluate_omaha_hand(&villain, &full_board);
                if hero_rank > villain_rank {
                    hero_wins += 1;
                } else if hero_rank < villain_rank {
                    villain_wins += 1;
                } else {
                    ties += 1;
                }

                if hi_lo {
                    let hero_low = evaluate_omaha_hand_low(&hero, &full_board);
                    let villain_low = evaluate_omaha_hand_low(&villain, &full_board);
                    match (hero_low, villain_low) {
                        (Some(h), Some(v)) => {
                            let mut h_vals = [0u32; 5];
                            let mut v_vals = [0u32; 5];
                            for i in 0..5 {
                                h_vals[i] = if h[i] == Rank::Ace { 1 } else { h[i] as u32 };
                                v_vals[i] = if v[i] == Rank::Ace { 1 } else { v[i] as u32 };
                            }
                            if h_vals < v_vals {
                                hero_wins_low += 1;
                            } else if h_vals > v_vals {
                                villain_wins_low += 1;
                            } else {
                                ties_low += 1;
                            }
                            low_count += 1;
                        }
                        (Some(_), None) => {
                            hero_wins_low += 1;
                            low_count += 1;
                        }
                        (None, Some(_)) => {
                            villain_wins_low += 1;
                            low_count += 1;
                        }
                        (None, None) => {}
                    }
                }
            }

            EquityResult {
                win: hero_wins as f64 / samples as f64,
                tie: ties as f64 / samples as f64,
                loss: villain_wins as f64 / samples as f64,
                win_low: if hi_lo && low_count > 0 {
                    Some(hero_wins_low as f64 / low_count as f64)
                } else {
                    None
                },
                tie_low: if hi_lo && low_count > 0 {
                    Some(ties_low as f64 / low_count as f64)
                } else {
                    None
                },
                loss_low: if hi_lo && low_count > 0 {
                    Some(villain_wins_low as f64 / low_count as f64)
                } else {
                    None
                },
                trial_count: samples,
                mode: EvalMode::MonteCarlo { samples, seed },
                confidence_interval: None,
            }
        }
        _ => {
            let samples = 10000;
            let seed = 42;
            evaluate_hand_vs_hand(
                hero,
                villain,
                board,
                EvalMode::MonteCarlo { samples, seed },
                hi_lo,
            )
        }
    }
}

/// CPU-only equity for one hero hand against a weighted villain [`Range`]:
/// runs [`evaluate_hand_vs_hand`] against each villain hand (skipping any
/// that share a card with `hero` or `board`) and combines the results as a
/// weighted average. `trial_count` on the result is the sum of every
/// sub-evaluation's trial count.
pub fn evaluate_hand_vs_range(
    hero: Hand,
    villain_range: Range,
    board: Board,
    mode: EvalMode,
    hi_lo: bool,
) -> EquityResult {
    let mut total_win = 0.0;
    let mut total_tie = 0.0;
    let mut total_loss = 0.0;
    let mut total_win_low = 0.0;
    let mut total_tie_low = 0.0;
    let mut total_loss_low = 0.0;
    let mut total_weight = 0.0;
    let mut total_weight_low = 0.0;
    let mut total_trials = 0;

    for (villain_hand, weight) in &villain_range.hands {
        // Skip if villain hand overlaps with hero hand or board
        if villain_hand
            .0
            .iter()
            .any(|c| hero.0.contains(c) || board.0.contains(c))
        {
            continue;
        }

        let res = evaluate_hand_vs_hand(
            hero.clone(),
            villain_hand.clone(),
            board.clone(),
            mode.clone(),
            hi_lo,
        );
        total_win += res.win * weight;
        total_tie += res.tie * weight;
        total_loss += res.loss * weight;
        if let (Some(wl), Some(tl), Some(ll)) = (res.win_low, res.tie_low, res.loss_low) {
            total_win_low += wl * weight;
            total_tie_low += tl * weight;
            total_loss_low += ll * weight;
            total_weight_low += weight;
        }
        total_weight += weight;
        total_trials += res.trial_count;
    }

    if total_weight == 0.0 {
        return EquityResult {
            win: 0.0,
            tie: 0.0,
            loss: 0.0,
            win_low: None,
            tie_low: None,
            loss_low: None,
            trial_count: 0,
            mode,
            confidence_interval: None,
        };
    }

    EquityResult {
        win: total_win / total_weight,
        tie: total_tie / total_weight,
        loss: total_loss / total_weight,
        win_low: if hi_lo && total_weight_low > 0.0 {
            Some(total_win_low / total_weight_low)
        } else {
            None
        },
        tie_low: if hi_lo && total_weight_low > 0.0 {
            Some(total_tie_low / total_weight_low)
        } else {
            None
        },
        loss_low: if hi_lo && total_weight_low > 0.0 {
            Some(total_loss_low / total_weight_low)
        } else {
            None
        },
        trial_count: total_trials,
        mode,
        confidence_interval: None,
    }
}

/// Range-vs-range equity, routed through `backend` (see [`Backend`] for GPU
/// routing/fallback rules). Thin wrapper over
/// [`Backend::run_range_evaluation`] — prefer calling that directly (or
/// [`Backend::run_range_evaluation_batch`] for many cases) if you already
/// have a `&Backend` and want to avoid the ownership shuffle here.
pub fn evaluate_range_vs_range(
    hero_range: Range,
    villain_range: Range,
    board: Board,
    mode: EvalMode,
    hi_lo: bool,
    backend: Backend,
) -> EquityResult {
    backend.run_range_evaluation(&hero_range, &villain_range, &board, &mode, hi_lo)
}

fn evaluate_range_vs_range_internal(
    hero_range: Range,
    villain_range: Range,
    board: Board,
    mode: EvalMode,
    hi_lo: bool,
) -> EquityResult {
    let mut total_win = 0.0;
    let mut total_tie = 0.0;
    let mut total_loss = 0.0;
    let total_win_low = 0.0;
    let total_tie_low = 0.0;
    let total_loss_low = 0.0;
    let mut total_weight = 0.0;
    let total_weight_low = 0.0;
    let mut total_trials = 0;

    for (hero_hand, hero_weight) in &hero_range.hands {
        // Skip if hero hand overlaps with board
        if hero_hand.0.iter().any(|c| board.0.contains(c)) {
            continue;
        }

        let mut sub_win = 0.0;
        let mut sub_tie = 0.0;
        let mut sub_loss = 0.0;
        let mut sub_weight = 0.0;
        let mut sub_trials = 0;

        for (villain_hand, villain_weight) in &villain_range.hands {
            // Skip if villain hand overlaps with hero hand or board
            if villain_hand
                .0
                .iter()
                .any(|c| hero_hand.0.contains(c) || board.0.contains(c))
            {
                continue;
            }

            let res = evaluate_hand_vs_hand(
                hero_hand.clone(),
                villain_hand.clone(),
                board.clone(),
                mode.clone(),
                hi_lo,
            );
            sub_win += res.win * villain_weight;
            sub_tie += res.tie * villain_weight;
            sub_loss += res.loss * villain_weight;
            sub_weight += villain_weight;
            sub_trials += res.trial_count;
        }

        if sub_weight > 0.0 {
            total_win += (sub_win / sub_weight) * hero_weight;
            total_tie += (sub_tie / sub_weight) * hero_weight;
            total_loss += (sub_loss / sub_weight) * hero_weight;
            total_weight += hero_weight;
            total_trials += sub_trials;
        }
    }

    if total_weight == 0.0 {
        return EquityResult {
            win: 0.0,
            tie: 0.0,
            loss: 0.0,
            win_low: None,
            tie_low: None,
            loss_low: None,
            trial_count: 0,
            mode,
            confidence_interval: None,
        };
    }

    EquityResult {
        win: total_win / total_weight,
        tie: total_tie / total_weight,
        loss: total_loss / total_weight,
        win_low: if hi_lo && total_weight_low > 0.0 {
            Some(total_win_low / total_weight_low)
        } else {
            None
        },
        tie_low: if hi_lo && total_weight_low > 0.0 {
            Some(total_tie_low / total_weight_low)
        } else {
            None
        },
        loss_low: if hi_lo && total_weight_low > 0.0 {
            Some(total_loss_low / total_weight_low)
        } else {
            None
        },
        trial_count: total_trials,
        mode,
        confidence_interval: None,
    }
}

/// Deals one random 4-card hand from the deck minus `dead_cards`. `rng_seed`
/// gives a reproducible deal; `None` seeds from OS entropy. For dealing
/// multiple non-overlapping hands (e.g. hero + villain), prefer
/// [`random_hands`] over calling this repeatedly — repeated independent
/// calls reshuffle the whole deck each time and are not guaranteed disjoint.
pub fn random_hand(dead_cards: &[Card], rng_seed: Option<u64>) -> Hand {
    random_hands(1, dead_cards, rng_seed).pop().unwrap()
}

/// Deals `count` non-overlapping 4-card hands from a single shuffle of the deck
/// (minus `dead_cards`).
pub fn random_hands(count: usize, dead_cards: &[Card], rng_seed: Option<u64>) -> Vec<Hand> {
    let mut deck: Vec<Card> = full_deck()
        .into_iter()
        .filter(|c| !dead_cards.contains(c))
        .collect();

    let mut rng = if let Some(seed) = rng_seed {
        Pcg64::seed_from_u64(seed)
    } else {
        Pcg64::from_entropy()
    };

    deck.shuffle(&mut rng);
    (0..count)
        .map(|i| {
            Hand::new([
                deck[i * 4],
                deck[i * 4 + 1],
                deck[i * 4 + 2],
                deck[i * 4 + 3],
            ])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omaha_evaluation() {
        // AsAcKhKd on AsKs2d board
        let hero = Hand::new([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::King, Suit::Diamonds),
        ]);
        let board = vec![
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::Two, Suit::Diamonds),
        ];

        // Omaha: exactly 2 from hand, 3 from board.
        // Hand cards: As, Ac, Kh, Kd
        // Board cards: Ah, Ks, 2d
        // Possible combinations (2 from hand + 3 from board):
        // 1. {As, Ac} + {Ah, Ks, 2d} -> AsAcAhKs2d (AAA K 2) -> Three of a Kind Aces
        // 2. {As, Kh} + {Ah, Ks, 2d} -> AsKhAhKs2d (AA KK 2) -> Two Pair Aces and Kings
        // ...
        // There are NO 5 cards from board in Omaha evaluation if we only have 3 board cards.
        // Wait, if the board has only 3 cards, there is only ONE combination: 2 from hand + ALL 3 from board.

        let rank = evaluate_omaha_hand(&hero, &board);
        match rank {
            HandRank::ThreeOfAKind(Rank::Ace, Rank::King, Rank::Two) => (),
            _ => panic!("Expected Three of a Kind Aces, got {:?}", rank),
        }
    }

    #[test]
    fn test_hand_vs_hand_exhaustive() {
        let hero = Hand::new([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Two, Suit::Hearts),
            Card::new(Rank::Three, Suit::Diamonds),
        ]);
        let villain = Hand::new([
            Card::new(Rank::King, Suit::Spades),
            Card::new(Rank::King, Suit::Clubs),
            Card::new(Rank::Four, Suit::Hearts),
            Card::new(Rank::Five, Suit::Diamonds),
        ]);
        let board = Board::new(vec![
            Card::new(Rank::Ace, Suit::Hearts),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::Seven, Suit::Diamonds),
            Card::new(Rank::Eight, Suit::Clubs),
            Card::new(Rank::Nine, Suit::Spades),
        ]);

        let result = evaluate_hand_vs_hand(hero, villain, board, EvalMode::Exhaustive, false);
        assert_eq!(result.win, 1.0);
        assert_eq!(result.loss, 0.0);
        assert_eq!(result.tie, 0.0);
    }

    #[test]
    fn test_omaha_hi_lo() {
        let hero = Hand::new([
            Card::new(Rank::Ace, Suit::Spades),
            Card::new(Rank::Two, Suit::Clubs),
            Card::new(Rank::King, Suit::Hearts),
            Card::new(Rank::King, Suit::Diamonds),
        ]);
        let villain = Hand::new([
            Card::new(Rank::Ace, Suit::Clubs),
            Card::new(Rank::Three, Suit::Clubs),
            Card::new(Rank::Queen, Suit::Hearts),
            Card::new(Rank::Queen, Suit::Diamonds),
        ]);
        // Board has 3 low cards: 3, 4, 5
        let board = Board::new(vec![
            Card::new(Rank::Three, Suit::Hearts),
            Card::new(Rank::Four, Suit::Diamonds),
            Card::new(Rank::Five, Suit::Clubs),
            Card::new(Rank::Ten, Suit::Spades),
            Card::new(Rank::Jack, Suit::Clubs),
        ]);

        let result = evaluate_hand_vs_hand(hero, villain, board, EvalMode::Exhaustive, true);

        // Hero low: A, 2 + 3, 4, 5 -> 5, 4, 3, 2, 1 (Best possible low)
        // Villain low: A, 3 + 4, 5, 3 is not possible because only 3 from board.
        // Villain low: A, 3 + 4, 5, 10 (no), A, 3 + 3 (no), A, 3 + 4, 5 and one more from hand? No, exactly 2 from hand.
        // Villain cards: A, 3, Q, Q. Board: 3, 4, 5, 10, J.
        // Villain best low: {A, 3} + {4, 5, 10} -> No (10 > 8). {A, 3} + {4, 5, J} -> No.
        // Wait, 3 from board must be used. Board has only {3, 4, 5} as low cards.
        // Villain must use {A, 3} and {3, 4, 5}? No, that's two 3s.
        // Low hand must have 5 unique ranks.
        // Villain low: {A, Q} + {3, 4, 5} -> No. {3, Q} + {3, 4, 5} -> No.
        // Villain has NO low hand.

        assert_eq!(result.win_low, Some(1.0));
    }

    #[test]
    fn test_canonical_sorting() {
        let cards = [
            Card::from_str("Ac").unwrap(),
            Card::from_str("As").unwrap(),
            Card::from_str("Kh").unwrap(),
            Card::from_str("2d").unwrap(),
        ];
        let hand = Hand::new(cards);
        // Canonical order: Rank descending, then Spades > Hearts > Diamonds > Clubs
        // Rank: A, A, K, 2
        // For A: Spades (s) > Clubs (c)
        // Expected: As, Ac, Kh, 2d
        assert_eq!(hand.0[0], Card::from_str("As").unwrap());
        assert_eq!(hand.0[1], Card::from_str("Ac").unwrap());
        assert_eq!(hand.0[2], Card::from_str("Kh").unwrap());
        assert_eq!(hand.0[3], Card::from_str("2d").unwrap());
    }

    #[test]
    #[ignore]
    fn test_gpu_vs_cpu() {
        let hero = Hand::new([
            Card::from_str("As").unwrap(),
            Card::from_str("Ac").unwrap(),
            Card::from_str("Ks").unwrap(),
            Card::from_str("Kc").unwrap(),
        ]);
        let villain = Hand::new([
            Card::from_str("Qs").unwrap(),
            Card::from_str("Qc").unwrap(),
            Card::from_str("Js").unwrap(),
            Card::from_str("Jc").unwrap(),
        ]);
        let board = Board::new(vec![
            Card::from_str("2s").unwrap(),
            Card::from_str("3s").unwrap(),
            Card::from_str("4s").unwrap(),
            Card::from_str("5s").unwrap(),
            Card::from_str("6h").unwrap(),
        ]);

        let cpu_res = evaluate_hand_vs_hand(
            hero.clone(),
            villain.clone(),
            board.clone(),
            EvalMode::Exhaustive,
            false,
        );
        let gpu_res =
            crate::gpu::run_gpu_evaluation(&hero, &villain, &board, &EvalMode::Exhaustive, false);

        if let Some(gpu_res) = gpu_res {
            assert!((cpu_res.win - gpu_res.win).abs() < 1e-6);
            assert!((cpu_res.tie - gpu_res.tie).abs() < 1e-6);
            assert!((cpu_res.loss - gpu_res.loss).abs() < 1e-6);
        } else {
            println!("GPU not available, skipping test");
        }
    }

    #[test]
    #[ignore]
    fn test_gpu_padding_sentinel() {
        let hero = Hand::new([
            Card::from_str("As").unwrap(),
            Card::from_str("Ac").unwrap(),
            Card::from_str("Ks").unwrap(),
            Card::from_str("Kc").unwrap(),
        ]);
        let villain = Hand::new([
            Card::from_str("Qs").unwrap(),
            Card::from_str("Qc").unwrap(),
            Card::from_str("Js").unwrap(),
            Card::from_str("Jc").unwrap(),
        ]);
        // 3-card board. Card index 0 (2s) is NOT on board.
        // If padding was 0, it might conflict.
        let board = Board::new(vec![
            Card::from_str("7s").unwrap(),
            Card::from_str("8s").unwrap(),
            Card::from_str("9s").unwrap(),
        ]);

        let gpu_res =
            crate::gpu::run_gpu_evaluation(&hero, &villain, &board, &EvalMode::Exhaustive, false);
        if let Some(res) = gpu_res {
            assert!(res.win + res.tie + res.loss > 0.0);
        }
    }

    #[test]
    #[ignore]
    fn test_gpu_preflop_batch_at_scale() {
        // Reproduces the real `simulation` workload: 256 cases, empty board,
        // 1000 Monte Carlo samples each, one hand per side.
        let mut rng_seed = 1u64;
        let mut cases = Vec::with_capacity(256);
        for _ in 0..256 {
            let hands = random_hands(2, &[], Some(rng_seed));
            rng_seed += 1;
            let hero_range = Range {
                hands: vec![(hands[0].clone(), 1.0)],
            };
            let villain_range = Range {
                hands: vec![(hands[1].clone(), 1.0)],
            };
            cases.push((
                hero_range,
                villain_range,
                Board::new(vec![]),
                EvalMode::MonteCarlo {
                    samples: 1000,
                    seed: 42,
                },
            ));
        }

        let gpu_results = crate::gpu::run_gpu_range_evaluation_batch(&cases);
        let zero_count = gpu_results
            .iter()
            .filter(
                |r| matches!(r, Some(res) if res.win == 0.0 && res.tie == 0.0 && res.loss == 0.0),
            )
            .count();
        let none_count = gpu_results.iter().filter(|r| r.is_none()).count();
        println!("zero-equity results: {zero_count}/256, None results: {none_count}/256");
        for (i, r) in gpu_results.iter().enumerate().take(5) {
            println!("case {i}: {r:?}");
        }
        assert_eq!(
            zero_count, 0,
            "GPU MC batch returned zero equities under real-world load"
        );
    }

    #[test]
    #[ignore]
    fn test_gpu_preflop_mc_matches_ps_eval() {
        // Fixed hands cross-checked against ps-eval's exhaustive equity:
        // AsAcKhKd/2h7c3d9s=0.6583, 2s9sJsTs/4c5h8hAc=0.4448,
        // 7cAcJhTs/5c6c9dTh=0.6144, KsQhJh4h/QcTd8h3h=0.5929,
        // 2c2h3sKs/4d5cKhTc=0.3855. This guards the lane-splitting math in
        // the MC branch (see MC_TARGET_PARALLELISM in omaha.wgsl) against
        // introducing bias, not just against crashing/returning zeros.
        let pairs: [(&str, &str, f64); 5] = [
            ("AsAcKhKd", "2h7c3d9s", 0.6583),
            ("2s9sJsTs", "4c5h8hAc", 0.4448),
            ("7cAcJhTs", "5c6c9dTh", 0.6144),
            ("KsQhJh4h", "QcTd8h3h", 0.5929),
            ("2c2h3sKs", "4d5cKhTc", 0.3855),
        ];
        for (hero_str, villain_str, expected_eq) in pairs {
            let hero = Hand::new([
                Card::from_str(&hero_str[0..2]).unwrap(),
                Card::from_str(&hero_str[2..4]).unwrap(),
                Card::from_str(&hero_str[4..6]).unwrap(),
                Card::from_str(&hero_str[6..8]).unwrap(),
            ]);
            let villain = Hand::new([
                Card::from_str(&villain_str[0..2]).unwrap(),
                Card::from_str(&villain_str[2..4]).unwrap(),
                Card::from_str(&villain_str[4..6]).unwrap(),
                Card::from_str(&villain_str[6..8]).unwrap(),
            ]);
            let hero_range = Range {
                hands: vec![(hero, 1.0)],
            };
            let villain_range = Range {
                hands: vec![(villain, 1.0)],
            };
            let res = crate::gpu::run_gpu_range_evaluation(
                &hero_range,
                &villain_range,
                &Board::new(vec![]),
                &EvalMode::MonteCarlo {
                    samples: 20000,
                    seed: 7,
                },
            )
            .expect("GPU should be available");
            let eq = res.win + res.tie / 2.0;
            let delta = (eq - expected_eq).abs();
            println!(
                "{} {} gpu={:.4} ps-eval={:.4} delta={:.4}",
                hero_str, villain_str, eq, expected_eq, delta
            );
            assert!(
                delta < 0.03,
                "{} vs {}: gpu={:.4} too far from ps-eval={:.4}",
                hero_str,
                villain_str,
                eq,
                expected_eq
            );
        }
    }

    #[test]
    #[ignore]
    fn test_gpu_preflop_batch_concurrent() {
        // Reproduces simulate_plo_no_flop's real access pattern: many rayon
        // threads hammering the batch GPU path concurrently.
        use std::sync::atomic::{AtomicU64, Ordering};
        let zero_total = std::sync::Arc::new(AtomicU64::new(0));
        let none_total = std::sync::Arc::new(AtomicU64::new(0));
        let case_total = std::sync::Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();
        for t in 0..8u64 {
            let zero_total = zero_total.clone();
            let none_total = none_total.clone();
            let case_total = case_total.clone();
            handles.push(std::thread::spawn(move || {
                for round in 0..5u64 {
                    let mut rng_seed = 1000 * t + 100 * round + 1;
                    let mut cases = Vec::with_capacity(256);
                    for _ in 0..256 {
                        let hands = random_hands(2, &[], Some(rng_seed));
                        rng_seed += 1;
                        let hero_range = Range {
                            hands: vec![(hands[0].clone(), 1.0)],
                        };
                        let villain_range = Range {
                            hands: vec![(hands[1].clone(), 1.0)],
                        };
                        cases.push((
                            hero_range,
                            villain_range,
                            Board::new(vec![]),
                            EvalMode::MonteCarlo {
                                samples: 1000,
                                seed: 42,
                            },
                        ));
                    }
                    let gpu_results = crate::gpu::run_gpu_range_evaluation_batch(&cases);
                    for r in &gpu_results {
                        case_total.fetch_add(1, Ordering::Relaxed);
                        match r {
                            None => {
                                none_total.fetch_add(1, Ordering::Relaxed);
                            }
                            Some(res) if res.win == 0.0 && res.tie == 0.0 && res.loss == 0.0 => {
                                zero_total.fetch_add(1, Ordering::Relaxed);
                            }
                            _ => {}
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let cases = case_total.load(Ordering::Relaxed);
        let zeros = zero_total.load(Ordering::Relaxed);
        let nones = none_total.load(Ordering::Relaxed);
        println!("concurrent: {cases} cases, {zeros} zero-equity, {nones} None (GPU unavailable)");
        assert_eq!(
            zeros, 0,
            "GPU MC batch returned zero equities under concurrent load"
        );
    }
}
