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

/// Which poker variant a [`Hand`]/[`Range`] holds and an evaluation runs
/// as. Omaha hands are 4 hole cards, showdown uses exactly 2 of them plus
/// exactly 3 board cards ([`evaluate_omaha_hand`]); Hold'em hands are 2
/// hole cards, showdown uses the best 5 of all hole+board cards
/// ([`evaluate_holdem_hand`]). `hi_lo` (Omaha Hi/Lo) is only defined for
/// `Game::Omaha` — see [`evaluate_hand_vs_hand`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Game {
    Omaha,
    HoldEm,
}

impl Default for Game {
    fn default() -> Self {
        Game::Omaha
    }
}

/// A hole-card hand: 4 cards for [`Game::Omaha`], 2 for [`Game::HoldEm`],
/// always held in canonical (§4.1.4) order. Which arity a given `Hand`
/// holds is exactly its `.0.len()` — [`Hand::game`] reads that back.
/// Constructing a `Hand` with any other length is a caller bug and not
/// checked here; the evaluation entry points (`evaluate_hand_vs_hand` and
/// friends) are where a `Game`/`Hand` arity mismatch is caught.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hand(pub Vec<Card>);

impl Hand {
    /// Sorts `cards` into canonical order and wraps them as an Omaha
    /// `Hand`. Kept as `new` (rather than requiring callers to switch to
    /// [`Hand::omaha`]) so existing 4-card call sites keep compiling.
    pub fn new(mut cards: [Card; 4]) -> Self {
        cards.sort();
        Self(cards.to_vec())
    }

    /// Sorts `cards` into canonical order and wraps them as an Omaha
    /// `Hand`. Identical to [`Hand::new`]; prefer this name at new call
    /// sites so the arity is explicit at a glance.
    pub fn omaha(mut cards: [Card; 4]) -> Self {
        cards.sort();
        Self(cards.to_vec())
    }

    /// Sorts `cards` into canonical order and wraps them as a Hold'em
    /// `Hand`.
    pub fn holdem(mut cards: [Card; 2]) -> Self {
        cards.sort();
        Self(cards.to_vec())
    }

    /// The [`Game`] this hand's arity implies. Panics if the hand was
    /// constructed with neither 2 nor 4 cards (a caller bug — see the
    /// struct docs).
    pub fn game(&self) -> Game {
        match self.0.len() {
            2 => Game::HoldEm,
            4 => Game::Omaha,
            n => panic!("Hand has {n} cards; expected 2 (Hold'em) or 4 (Omaha)"),
        }
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
    ///
    /// `game` selects the notation dialect: [`Game::Omaha`] parses the
    /// §4.1.3 syntax below (unchanged from before `Game` existed);
    /// [`Game::HoldEm`] parses the standard combo-range dialect
    /// (`AA`, `AKs`, `AKo`, `22+`, `ATs+`, `JTs-54s`, exact combos like
    /// `AhKd`) via [`expand_holdem_range`] instead — the two dialects
    /// share no token shapes worth unifying (Omaha's 8-char exact hand and
    /// Hold'em's 4-char exact combo already overlap in length only, not
    /// meaning).
    pub fn from_shorthand(s: &str, dead_cards: &[Card], game: Game) -> Result<Self, String> {
        let mut hands = Vec::new();
        let parts = s.split(',').map(|p| p.trim());

        for part in parts {
            if part.is_empty() {
                continue;
            }

            if game == Game::HoldEm {
                hands.extend(expand_holdem_range(part, dead_cards)?);
                continue;
            }

            // §4.1.3 Range Notation (Omaha)
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

/// `Rank` from its numeric value (`Two` = 2 .. `Ace` = 14, matching the
/// enum's own discriminants) — the inverse of `rank as u32`. Used by the
/// Hold'em range parser to step between ranks arithmetically (`22+`,
/// `ATs+`, `JTs-54s`), which the char-based `Rank::from_char` can't do.
fn rank_from_val(v: u32) -> Option<Rank> {
    match v {
        2 => Some(Rank::Two),
        3 => Some(Rank::Three),
        4 => Some(Rank::Four),
        5 => Some(Rank::Five),
        6 => Some(Rank::Six),
        7 => Some(Rank::Seven),
        8 => Some(Rank::Eight),
        9 => Some(Rank::Nine),
        10 => Some(Rank::Ten),
        11 => Some(Rank::Jack),
        12 => Some(Rank::Queen),
        13 => Some(Rank::King),
        14 => Some(Rank::Ace),
        _ => None,
    }
}

/// All `C(4,2) = 6` pocket-pair combos of `r`, minus any that touch a dead
/// card.
fn holdem_pair_combos(r: Rank, dead_cards: &[Card]) -> Vec<Hand> {
    let cards: Vec<Card> = Suit::all().map(|s| Card::new(r, s)).collect();
    let mut out = Vec::new();
    for i in 0..cards.len() {
        for j in i + 1..cards.len() {
            if !dead_cards.contains(&cards[i]) && !dead_cards.contains(&cards[j]) {
                out.push(Hand::holdem([cards[i], cards[j]]));
            }
        }
    }
    out
}

/// The 4 same-suit combos of `hi`+`lo` (one per suit), minus any that
/// touch a dead card.
fn holdem_suited_combos(hi: Rank, lo: Rank, dead_cards: &[Card]) -> Vec<Hand> {
    Suit::all()
        .filter_map(|s| {
            let (c1, c2) = (Card::new(hi, s), Card::new(lo, s));
            if dead_cards.contains(&c1) || dead_cards.contains(&c2) {
                None
            } else {
                Some(Hand::holdem([c1, c2]))
            }
        })
        .collect()
}

/// The 12 different-suit combos of `hi`+`lo` (every ordered suit pair),
/// minus any that touch a dead card.
fn holdem_offsuit_combos(hi: Rank, lo: Rank, dead_cards: &[Card]) -> Vec<Hand> {
    let mut out = Vec::new();
    for s1 in Suit::all() {
        for s2 in Suit::all() {
            if s1 == s2 {
                continue;
            }
            let (c1, c2) = (Card::new(hi, s1), Card::new(lo, s2));
            if !dead_cards.contains(&c1) && !dead_cards.contains(&c2) {
                out.push(Hand::holdem([c1, c2]));
            }
        }
    }
    out
}

/// Expands one comma-separated token of Hold'em range notation into
/// concrete 2-card combos, each with weight `1.0`. Supported shapes (see
/// `docs/Milestone3.md` §2.3 for the full table):
///
/// - Exact combo: `AhKd` (two `RankSuit` cards back to back).
/// - Pair: `AA`, `22+` (that rank through Ace), `22-66` (inclusive range).
/// - Unpaired, both suitedness: `AK` (all 16 combos).
/// - Unpaired, one suitedness: `AKs` (4), `AKo` (12).
/// - Unpaired plus: `ATs+` / `KQo+` — the *lower* rank climbs from the
///   given value up to (but not including) the higher rank, which stays
///   fixed.
/// - Unpaired dash range: `JTs-54s` / `JTo-54o` — both ranks step down
///   together, holding the gap between them constant (both sides must
///   already share that gap; this is not a generic connector search).
///
/// A full 169-combo chart expansion (`"random"`, positional aliases like
/// `"UTG"`, etc.) is out of scope — see `docs/Milestone3.md` §2.3.
fn expand_holdem_range(part: &str, dead_cards: &[Card]) -> Result<Vec<(Hand, f64)>, String> {
    let part = part.trim();
    if part.is_empty() {
        return Ok(Vec::new());
    }

    // Exact combo: two full "RankSuit" cards, e.g. "AhKd". Tried first —
    // it's the only shape that's exactly 4 chars *and* has a valid suit
    // char in positions 1 and 3, so it never collides with "ATs+"/"KQo+"
    // (position 1 there is 's'/'o', not a suit char).
    if part.len() == 4 {
        if let (Some(c1), Some(c2)) = (Card::from_str(&part[0..2]), Card::from_str(&part[2..4])) {
            return Ok(if c1 == c2 || dead_cards.contains(&c1) || dead_cards.contains(&c2) {
                Vec::new()
            } else {
                vec![(Hand::holdem([c1, c2]), 1.0)]
            });
        }
    }

    let chars: Vec<char> = part.chars().collect();
    if chars.len() < 2 {
        return Err(format!("Unsupported Hold'em range token: '{}'", part));
    }
    let r0 = Rank::from_char(chars[0]).ok_or_else(|| format!("Invalid rank in '{}'", part))?;
    let r1 = Rank::from_char(chars[1]).ok_or_else(|| format!("Invalid rank in '{}'", part))?;
    let rest: String = chars[2..].iter().collect();

    let combos: Vec<Hand> = if r0 == r1 {
        // Pair: "AA", "22+", "22-66"
        match rest.as_str() {
            "" => holdem_pair_combos(r0, dead_cards),
            "+" => (r0 as u32..=Rank::Ace as u32)
                .filter_map(rank_from_val)
                .flat_map(|r| holdem_pair_combos(r, dead_cards))
                .collect(),
            s if s.starts_with('-') && s.len() == 3 => {
                let oc: Vec<char> = s[1..].chars().collect();
                let r2 = Rank::from_char(oc[0]).ok_or_else(|| format!("Invalid rank in '{}'", part))?;
                let r3 = Rank::from_char(oc[1]).ok_or_else(|| format!("Invalid rank in '{}'", part))?;
                if r2 != r3 {
                    return Err(format!("'{}' mixes a pair with a non-pair in a dash range", part));
                }
                let (lo, hi) = if (r0 as u32) <= (r2 as u32) {
                    (r0 as u32, r2 as u32)
                } else {
                    (r2 as u32, r0 as u32)
                };
                (lo..=hi)
                    .filter_map(rank_from_val)
                    .flat_map(|r| holdem_pair_combos(r, dead_cards))
                    .collect()
            }
            _ => return Err(format!("Unsupported Hold'em range token: '{}'", part)),
        }
    } else {
        // Unpaired: "AK", "AKs", "AKo", "ATs+", "KQo+", "JTs-54s", "JTo-54o"
        let (hi, lo) = if (r0 as u32) >= (r1 as u32) { (r0, r1) } else { (r1, r0) };
        match rest.as_str() {
            "" => {
                let mut v = holdem_suited_combos(hi, lo, dead_cards);
                v.extend(holdem_offsuit_combos(hi, lo, dead_cards));
                v
            }
            "s" => holdem_suited_combos(hi, lo, dead_cards),
            "o" => holdem_offsuit_combos(hi, lo, dead_cards),
            "s+" => ((lo as u32)..(hi as u32))
                .filter_map(rank_from_val)
                .flat_map(|l| holdem_suited_combos(hi, l, dead_cards))
                .collect(),
            "o+" => ((lo as u32)..(hi as u32))
                .filter_map(rank_from_val)
                .flat_map(|l| holdem_offsuit_combos(hi, l, dead_cards))
                .collect(),
            s if (s.starts_with('s') || s.starts_with('o')) && s.len() == 5 && s.as_bytes()[1] == b'-' => {
                let suited = s.starts_with('s');
                let qualifier = s.chars().last().unwrap();
                if qualifier != (if suited { 's' } else { 'o' }) {
                    return Err(format!("'{}' mixes suited and offsuit across a dash range", part));
                }
                let rc: Vec<char> = s[2..4].chars().collect();
                let r2 = Rank::from_char(rc[0]).ok_or_else(|| format!("Invalid rank in '{}'", part))?;
                let r3 = Rank::from_char(rc[1]).ok_or_else(|| format!("Invalid rank in '{}'", part))?;
                if r2 == r3 {
                    return Err(format!("'{}' mixes a pair with a non-pair in a dash range", part));
                }
                let (hi2, lo2) = if (r2 as u32) >= (r3 as u32) { (r2, r3) } else { (r3, r2) };
                let gap = hi as i32 - lo as i32;
                if hi2 as i32 - lo2 as i32 != gap {
                    return Err(format!(
                        "'{}' dash range must keep a constant gap between ranks",
                        part
                    ));
                }
                let (start, end) = if (hi as u32) >= (hi2 as u32) {
                    (hi2 as u32, hi as u32)
                } else {
                    (hi as u32, hi2 as u32)
                };
                (start..=end)
                    .filter_map(rank_from_val)
                    .filter_map(|h| rank_from_val(h as u32 - gap as u32).map(|l| (h, l)))
                    .flat_map(|(h, l)| {
                        if suited {
                            holdem_suited_combos(h, l, dead_cards)
                        } else {
                            holdem_offsuit_combos(h, l, dead_cards)
                        }
                    })
                    .collect()
            }
            _ => return Err(format!("Unsupported Hold'em range token: '{}'", part)),
        }
    };

    Ok(combos.into_iter().map(|h| (h, 1.0)).collect())
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
    /// `game` gates GPU use: the shader only knows the 4-card Omaha layout
    /// (see `src/omaha.wgsl`), so `Game::HoldEm` always runs on CPU even
    /// under an explicit GPU backend — the alternative would be a hard
    /// `None`/panic on a 2-card hand the shader can't parse, which is worse
    /// than just computing the (correct) CPU answer.
    pub fn run_evaluation(
        &self,
        hero: &Hand,
        villain: &Hand,
        board: &Board,
        mode: &EvalMode,
        hi_lo: bool,
        game: Game,
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
                game,
            )),
            Backend::Auto => {
                // Try GPU first
                if game == Game::Omaha {
                    if let Some(res) =
                        crate::gpu::run_gpu_evaluation(hero, villain, board, mode, hi_lo)
                    {
                        return Some(res);
                    }
                }
                Some(evaluate_hand_vs_hand(
                    hero.clone(),
                    villain.clone(),
                    board.clone(),
                    mode.clone(),
                    hi_lo,
                    game,
                ))
            }
            Backend::Cuda | Backend::Vulkan | Backend::Metal => {
                if game == Game::Omaha {
                    crate::gpu::run_gpu_evaluation(hero, villain, board, mode, hi_lo)
                } else {
                    Some(evaluate_hand_vs_hand(
                        hero.clone(),
                        villain.clone(),
                        board.clone(),
                        mode.clone(),
                        hi_lo,
                        game,
                    ))
                }
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
        game: Game,
    ) -> EquityResult {
        match self {
            Backend::Cpu => evaluate_range_vs_range_internal(
                hero_range.clone(),
                villain_range.clone(),
                board.clone(),
                mode.clone(),
                hi_lo,
                game,
            ),
            Backend::Auto => {
                if !hi_lo && game == Game::Omaha {
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
                    game,
                )
            }
            _ => {
                if !hi_lo && game == Game::Omaha {
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
                    game,
                )
            }
        }
    }

    /// Range-vs-range equity for many independent cases at once. `hi_lo`
    /// applies uniformly to every case (all-CPU) since the GPU shader only
    /// implements Omaha Hi. For `!hi_lo` Omaha cases, cases are chunked into
    /// batches of up to 256 (the GPU shader's fixed per-dispatch case
    /// limit) and sent to the GPU together; any case the GPU can't or
    /// didn't resolve (e.g. no adapter available) falls back to the CPU
    /// individually, so a single bad/unsupported case never drags the
    /// whole batch to CPU. Hold'em cases always run on CPU (no GPU kernel
    /// yet — see `run_evaluation`'s doc comment) and are not sent to the
    /// GPU at all, not even to have it decline them.
    pub fn run_range_evaluation_batch(
        &self,
        cases: &[(Range, Range, Board, EvalMode)],
        hi_lo: bool,
        game: Game,
    ) -> Vec<EquityResult> {
        let cpu_all = |cases: &[(Range, Range, Board, EvalMode)]| {
            cases
                .iter()
                .map(|(h, v, b, m)| {
                    evaluate_range_vs_range_internal(
                        h.clone(),
                        v.clone(),
                        b.clone(),
                        m.clone(),
                        hi_lo,
                        game,
                    )
                })
                .collect::<Vec<_>>()
        };

        if hi_lo || game == Game::HoldEm {
            return cpu_all(cases);
        }

        match self {
            Backend::Cpu => cpu_all(cases),
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
                                game,
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

/// Best 5-card Hold'em hand `hole` (exactly 2 cards) can make with `board`,
/// trying every 5-card subset of the combined hole+board cards — *not*
/// Omaha's "exactly 2 from hand" rule, since Hold'em allows using 0, 1, or
/// 2 hole cards. `board.len()` must be 0 or in `3..=5`; boards with fewer
/// than 3 cards return the lowest possible [`HandRank`] (equity functions
/// deal the rest of the board first rather than ranking a partial one).
/// This single loop already covers flop (`board.len() == 3`, exactly 1
/// subset — all 5 cards are used), turn (`C(6,5) = 6` subsets) and river
/// (`C(7,5) = 21` subsets); there is no separate per-street rule to
/// maintain.
pub fn evaluate_holdem_hand(hole: &Hand, board: &[Card]) -> HandRank {
    let mut best_rank = HandRank::HighCard(Rank::Two, Rank::Two, Rank::Two, Rank::Two, Rank::Two);

    if board.len() < 3 {
        return best_rank;
    }

    let mut all = [Card::new(Rank::Two, Suit::Spades); 7];
    all[0] = hole.0[0];
    all[1] = hole.0[1];
    all[2..2 + board.len()].copy_from_slice(board);
    let n = 2 + board.len();

    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                for d in c + 1..n {
                    for e in d + 1..n {
                        let cards = [all[a], all[b], all[c], all[d], all[e]];
                        let rank = crate::eval::evaluate_5_cards(&cards);
                        if rank > best_rank {
                            best_rank = rank;
                        }
                    }
                }
            }
        }
    }
    best_rank
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
    game: Game,
) -> EquityResult {
    assert!(
        !(hi_lo && game == Game::HoldEm),
        "hi_lo is only defined for Game::Omaha"
    );
    let rank_of = |hand: &Hand, fb: &[Card]| match game {
        Game::Omaha => evaluate_omaha_hand(hand, fb),
        Game::HoldEm => evaluate_holdem_hand(hand, fb),
    };
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
                let hero_rank = rank_of(&hero, fb);
                let villain_rank = rank_of(&villain, fb);
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

                let hero_rank = rank_of(&hero, &full_board);
                let villain_rank = rank_of(&villain, &full_board);
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
                game,
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
    game: Game,
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
            game,
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
    game: Game,
) -> EquityResult {
    backend.run_range_evaluation(&hero_range, &villain_range, &board, &mode, hi_lo, game)
}

fn evaluate_range_vs_range_internal(
    hero_range: Range,
    villain_range: Range,
    board: Board,
    mode: EvalMode,
    hi_lo: bool,
    game: Game,
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
                game,
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

/// Deals one random hand (4 cards for [`Game::Omaha`], 2 for
/// [`Game::HoldEm`]) from the deck minus `dead_cards`. `rng_seed` gives a
/// reproducible deal; `None` seeds from OS entropy. For dealing multiple
/// non-overlapping hands (e.g. hero + villain), prefer [`random_hands`]
/// over calling this repeatedly — repeated independent calls reshuffle the
/// whole deck each time and are not guaranteed disjoint.
pub fn random_hand(dead_cards: &[Card], rng_seed: Option<u64>, game: Game) -> Hand {
    random_hands(1, dead_cards, rng_seed, game).pop().unwrap()
}

/// Deals `count` non-overlapping hands (4 cards each for [`Game::Omaha`],
/// 2 each for [`Game::HoldEm`]) from a single shuffle of the deck (minus
/// `dead_cards`).
pub fn random_hands(count: usize, dead_cards: &[Card], rng_seed: Option<u64>, game: Game) -> Vec<Hand> {
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
    let arity = match game {
        Game::Omaha => 4,
        Game::HoldEm => 2,
    };
    (0..count)
        .map(|i| {
            let mut cards = deck[i * arity..i * arity + arity].to_vec();
            cards.sort();
            Hand(cards)
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

        let result = evaluate_hand_vs_hand(hero, villain, board, EvalMode::Exhaustive, false, Game::Omaha);
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

        let result = evaluate_hand_vs_hand(hero, villain, board, EvalMode::Exhaustive, true, Game::Omaha);

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

    // --- M3: Hold'em (docs/Milestone3.md) ---

    #[test]
    fn test_holdem_evaluation() {
        // Hole cards Ah Kh don't touch the board at all: the board itself
        // (As Ks Qs Js Ts) is a royal flush. Omaha would be forced to use
        // exactly 2 hole cards here and could never see this; Hold'em's
        // "best 5 of 7" rule must find it using 0 hole cards.
        let hole = Hand::holdem([Card::from_str("Ah").unwrap(), Card::from_str("Kh").unwrap()]);
        let board = vec![
            Card::from_str("As").unwrap(),
            Card::from_str("Ks").unwrap(),
            Card::from_str("Qs").unwrap(),
            Card::from_str("Js").unwrap(),
            Card::from_str("Ts").unwrap(),
        ];
        let rank = evaluate_holdem_hand(&hole, &board);
        assert_eq!(rank, HandRank::StraightFlush(Rank::Ace));
    }

    #[test]
    fn test_holdem_hand_vs_hand_exhaustive() {
        let hero = Hand::holdem([Card::from_str("As").unwrap(), Card::from_str("Ac").unwrap()]);
        let villain = Hand::holdem([Card::from_str("4h").unwrap(), Card::from_str("5d").unwrap()]);
        let board = Board::new(vec![
            Card::from_str("Ah").unwrap(),
            Card::from_str("Kh").unwrap(),
            Card::from_str("7d").unwrap(),
            Card::from_str("8c").unwrap(),
            Card::from_str("9s").unwrap(),
        ]);

        let result = evaluate_hand_vs_hand(hero, villain, board, EvalMode::Exhaustive, false, Game::HoldEm);
        assert_eq!(result.win, 1.0);
        assert_eq!(result.loss, 0.0);
        assert_eq!(result.tie, 0.0);
    }

    #[test]
    fn test_holdem_canonical_sorting() {
        let hand = Hand::holdem([Card::from_str("Ac").unwrap(), Card::from_str("As").unwrap()]);
        assert_eq!(hand.0[0], Card::from_str("As").unwrap());
        assert_eq!(hand.0[1], Card::from_str("Ac").unwrap());
        assert_eq!(hand.game(), Game::HoldEm);
    }

    #[test]
    #[should_panic(expected = "hi_lo is only defined for Game::Omaha")]
    fn test_holdem_hi_lo_rejected() {
        let hero = Hand::holdem([Card::from_str("As").unwrap(), Card::from_str("Ac").unwrap()]);
        let villain = Hand::holdem([Card::from_str("4h").unwrap(), Card::from_str("5d").unwrap()]);
        let board = Board::new(vec![
            Card::from_str("Ah").unwrap(),
            Card::from_str("Kh").unwrap(),
            Card::from_str("7d").unwrap(),
            Card::from_str("8c").unwrap(),
            Card::from_str("9s").unwrap(),
        ]);
        let _ = evaluate_hand_vs_hand(hero, villain, board, EvalMode::Exhaustive, true, Game::HoldEm);
    }

    #[test]
    fn test_holdem_range_pair() {
        // "AA": all C(4,2) = 6 combos, every hand exactly {A,A}.
        let range = Range::from_shorthand("AA", &[], Game::HoldEm).unwrap();
        assert_eq!(range.hands.len(), 6);
        for (hand, _) in &range.hands {
            assert_eq!(hand.0.len(), 2);
            assert!(hand.0.iter().all(|c| c.rank == Rank::Ace));
        }
    }

    #[test]
    fn test_holdem_range_pair_plus_and_dash() {
        // "22+": every pair 22..AA, 13 ranks * 6 combos = 78.
        let plus = Range::from_shorthand("22+", &[], Game::HoldEm).unwrap();
        assert_eq!(plus.hands.len(), 13 * 6);

        // "22-66": pairs 22,33,44,55,66 -> 5 ranks * 6 combos = 30.
        let dash = Range::from_shorthand("22-66", &[], Game::HoldEm).unwrap();
        assert_eq!(dash.hands.len(), 5 * 6);
    }

    #[test]
    fn test_holdem_range_unpaired_suitedness() {
        // "AK": all 16 combos (4 suited + 12 offsuit).
        let both = Range::from_shorthand("AK", &[], Game::HoldEm).unwrap();
        assert_eq!(both.hands.len(), 16);

        // "AKs": suited only, 4 combos.
        let suited = Range::from_shorthand("AKs", &[], Game::HoldEm).unwrap();
        assert_eq!(suited.hands.len(), 4);

        // "AKo": offsuit only, 12 combos.
        let offsuit = Range::from_shorthand("AKo", &[], Game::HoldEm).unwrap();
        assert_eq!(offsuit.hands.len(), 12);
    }

    #[test]
    fn test_holdem_range_plus_and_dash_unpaired() {
        // "ATs+": AT s, AJs, AQs, AKs -> 4 ranks * 4 combos = 16.
        let ats_plus = Range::from_shorthand("ATs+", &[], Game::HoldEm).unwrap();
        assert_eq!(ats_plus.hands.len(), 4 * 4);

        // "KQo+": only KQo itself (Q is already one below K) -> 12 combos.
        let kqo_plus = Range::from_shorthand("KQo+", &[], Game::HoldEm).unwrap();
        assert_eq!(kqo_plus.hands.len(), 12);

        // "JTs-54s": connectors JT, T9, 98, 87, 76, 65, 54 -> 7 * 4 = 28.
        let dash_suited = Range::from_shorthand("JTs-54s", &[], Game::HoldEm).unwrap();
        assert_eq!(dash_suited.hands.len(), 7 * 4);

        // Same range, offsuit: 7 * 12 = 84.
        let dash_offsuit = Range::from_shorthand("JTo-54o", &[], Game::HoldEm).unwrap();
        assert_eq!(dash_offsuit.hands.len(), 7 * 12);
    }

    #[test]
    fn test_holdem_range_exact_combo_and_dead_cards() {
        let exact = Range::from_shorthand("AhKd", &[], Game::HoldEm).unwrap();
        assert_eq!(exact.hands.len(), 1);
        assert_eq!(exact.hands[0].0.0.len(), 2);

        // Removing one Ace leaves 3 remaining pair combos (C(3,2) = 3).
        let dead = [Card::from_str("As").unwrap()];
        let pair_minus_dead = Range::from_shorthand("AA", &dead, Game::HoldEm).unwrap();
        assert_eq!(pair_minus_dead.hands.len(), 3);
        assert!(pair_minus_dead
            .hands
            .iter()
            .all(|(h, _)| !h.0.contains(&dead[0])));
    }

    #[test]
    fn test_holdem_range_isolated_from_omaha() {
        // The same "AA" token means something different (and a different
        // hand arity) per game — the parsers must not bleed into each other.
        let omaha_aa = Range::from_shorthand("AA", &[], Game::Omaha).unwrap();
        let holdem_aa = Range::from_shorthand("AA", &[], Game::HoldEm).unwrap();
        assert_eq!(omaha_aa.hands[0].0.0.len(), 4);
        assert_eq!(holdem_aa.hands[0].0.0.len(), 2);
        assert_ne!(omaha_aa.hands.len(), holdem_aa.hands.len());
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
            Game::Omaha,
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
            let hands = random_hands(2, &[], Some(rng_seed), Game::Omaha);
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
                        let hands = random_hands(2, &[], Some(rng_seed), Game::Omaha);
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
