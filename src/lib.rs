pub mod eval;
pub mod eval_fast;
pub mod gpu;

use serde::{Deserialize, Serialize};
use crate::eval::HandRank;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64;

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
            Rank::Two, Rank::Three, Rank::Four, Rank::Five, Rank::Six, Rank::Seven,
            Rank::Eight, Rank::Nine, Rank::Ten, Rank::Jack, Rank::Queen, Rank::King, Rank::Ace,
        ].into_iter()
    }

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
}

impl Suit {
    pub fn all() -> impl Iterator<Item = Suit> {
        [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs].into_iter()
    }

    pub fn from_char(c: char) -> Option<Suit> {
        match c.to_ascii_lowercase() {
            's' => Some(Suit::Spades),
            'h' => Some(Suit::Hearts),
            'd' => Some(Suit::Diamonds),
            'c' => Some(Suit::Clubs),
            _ => None,
        }
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hand(pub [Card; 4]);

impl Hand {
    pub fn new(mut cards: [Card; 4]) -> Self {
        cards.sort();
        Self(cards)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Board(pub Vec<Card>);

impl Board {
    pub fn new(mut cards: Vec<Card>) -> Self {
        cards.sort();
        Self(cards)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    pub hands: Vec<(Hand, f64)>,
}

impl Range {
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
                     if let Some(c) = Card::from_str(&part[i*2..i*2+2]) {
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
                let ranks: Vec<Rank> = part.chars()
                    .map(|c| Rank::from_char(c).ok_or(format!("Invalid rank: {}", c)))
                    .collect::<Result<Vec<_>, _>>()?;
                
                let mut deck = Vec::new();
                for r in Rank::all() {
                    for s in Suit::all() {
                        let c = Card::new(r, s);
                        if !dead_cards.contains(&c) {
                            deck.push(c);
                        }
                    }
                }
                
                let mut combos = Vec::new();
                for c1_idx in 0..deck.len() {
                    for c2_idx in c1_idx+1..deck.len() {
                        for c3_idx in c2_idx+1..deck.len() {
                            for c4_idx in c3_idx+1..deck.len() {
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
                    if let Some(c) = Card::from_str(&part[i*2..i*2+2]) {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalMode {
    Auto,
    Exhaustive,
    MonteCarlo { samples: u64, seed: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Backend {
    Auto,
    Cpu,
    Cuda,
    Vulkan,
    Metal,
}

impl Backend {
    pub fn is_available(&self) -> bool {
        match self {
            Backend::Auto => true,
            Backend::Cpu => true,
            Backend::Cuda | Backend::Vulkan | Backend::Metal => true,
        }
    }

    pub fn run_evaluation(&self, hero: &Hand, villain: &Hand, board: &Board, mode: &EvalMode, hi_lo: bool) -> Option<EquityResult> {
        if !self.is_available() {
            return None;
        }
        match self {
            Backend::Cpu => Some(evaluate_hand_vs_hand(hero.clone(), villain.clone(), board.clone(), mode.clone(), hi_lo)),
            Backend::Auto => {
                // Try GPU first
                if let Some(res) = crate::gpu::run_gpu_evaluation(hero, villain, board, mode, hi_lo) {
                    Some(res)
                } else {
                    Some(evaluate_hand_vs_hand(hero.clone(), villain.clone(), board.clone(), mode.clone(), hi_lo))
                }
            }
            Backend::Cuda | Backend::Vulkan | Backend::Metal => {
                crate::gpu::run_gpu_evaluation(hero, villain, board, mode, hi_lo)
            }
        }
    }

    pub fn run_range_evaluation(&self, hero_range: &Range, villain_range: &Range, board: &Board, mode: &EvalMode, hi_lo: bool) -> EquityResult {
        match self {
            Backend::Cpu => evaluate_range_vs_range_internal(hero_range.clone(), villain_range.clone(), board.clone(), mode.clone(), hi_lo),
            Backend::Auto => {
                if !hi_lo {
                    if let Some(res) = crate::gpu::run_gpu_range_evaluation(hero_range, villain_range, board, mode) {
                        return res;
                    }
                }
                evaluate_range_vs_range_internal(hero_range.clone(), villain_range.clone(), board.clone(), mode.clone(), hi_lo)
            }
            _ => {
                if !hi_lo {
                    if let Some(res) = crate::gpu::run_gpu_range_evaluation(hero_range, villain_range, board, mode) {
                        return res;
                    }
                }
                evaluate_range_vs_range_internal(hero_range.clone(), villain_range.clone(), board.clone(), mode.clone(), hi_lo)
            }
        }
    }

    pub fn run_range_evaluation_batch(
        &self,
        cases: &[(Range, Range, Board, EvalMode)],
        hi_lo: bool,
    ) -> Vec<EquityResult> {
        if hi_lo {
            return cases.iter().map(|(h, v, b, m)| {
                evaluate_range_vs_range_internal(h.clone(), v.clone(), b.clone(), m.clone(), hi_lo)
            }).collect();
        }

        match self {
            Backend::Cpu => cases.iter().map(|(h, v, b, m)| {
                evaluate_range_vs_range_internal(h.clone(), v.clone(), b.clone(), m.clone(), hi_lo)
            }).collect(),
            Backend::Auto | Backend::Metal | Backend::Cuda | Backend::Vulkan => {
                let mut all_results = Vec::with_capacity(cases.len());
                for chunk in cases.chunks(256) {
                    let gpu_results = crate::gpu::run_gpu_range_evaluation_batch(chunk);
                    for (i, res) in gpu_results.into_iter().enumerate() {
                        if let Some(r) = res {
                            all_results.push(r);
                        } else {
                            let (h, v, b, m) = &chunk[i];
                            all_results.push(evaluate_range_vs_range_internal(h.clone(), v.clone(), b.clone(), m.clone(), hi_lo));
                        }
                    }
                }
                all_results
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityResult {
    pub win: f64,
    pub tie: f64,
    pub loss: f64,
    pub win_low: Option<f64>,
    pub tie_low: Option<f64>,
    pub loss_low: Option<f64>,
    pub trial_count: u64,
    pub mode: EvalMode,
    pub confidence_interval: Option<(f64, f64)>,
}

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
                    let cards = [
                        hand.0[h1], hand.0[h2],
                        board[i], board[j], board[k]
                    ];
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
                    let cards = [
                        hand.0[h1], hand.0[h2],
                        board[i], board[j], board[k]
                    ];
                    if let Some(low_rank) = crate::eval::evaluate_5_cards_low(&cards) {
                        if let Some(best) = best_low {
                            let mut cur_vals = [0u32; 5];
                            let mut best_vals = [0u32; 5];
                            for n in 0..5 {
                                cur_vals[n] = if low_rank[n] == Rank::Ace { 1 } else { low_rank[n] as u32 };
                                best_vals[n] = if best[n] == Rank::Ace { 1 } else { best[n] as u32 };
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
                win_low: if hi_lo && low_count > 0 { Some(hero_wins_low as f64 / low_count as f64) } else { None },
                tie_low: if hi_lo && low_count > 0 { Some(ties_low as f64 / low_count as f64) } else { None },
                loss_low: if hi_lo && low_count > 0 { Some(villain_wins_low as f64 / low_count as f64) } else { None },
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
                win_low: if hi_lo && low_count > 0 { Some(hero_wins_low as f64 / low_count as f64) } else { None },
                tie_low: if hi_lo && low_count > 0 { Some(ties_low as f64 / low_count as f64) } else { None },
                loss_low: if hi_lo && low_count > 0 { Some(villain_wins_low as f64 / low_count as f64) } else { None },
                trial_count: samples,
                mode: EvalMode::MonteCarlo { samples, seed },
                confidence_interval: None,
            }
        }
        _ => {
            let samples = 10000;
            let seed = 42;
            evaluate_hand_vs_hand(hero, villain, board, EvalMode::MonteCarlo { samples, seed }, hi_lo)
        }
    }
}

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
        if villain_hand.0.iter().any(|c| hero.0.contains(c) || board.0.contains(c)) {
            continue;
        }

        let res = evaluate_hand_vs_hand(hero.clone(), villain_hand.clone(), board.clone(), mode.clone(), hi_lo);
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
        win_low: if hi_lo && total_weight_low > 0.0 { Some(total_win_low / total_weight_low) } else { None },
        tie_low: if hi_lo && total_weight_low > 0.0 { Some(total_tie_low / total_weight_low) } else { None },
        loss_low: if hi_lo && total_weight_low > 0.0 { Some(total_loss_low / total_weight_low) } else { None },
        trial_count: total_trials,
        mode,
        confidence_interval: None,
    }
}

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
            if villain_hand.0.iter().any(|c| hero_hand.0.contains(c) || board.0.contains(c)) {
                continue;
            }

            let res = evaluate_hand_vs_hand(hero_hand.clone(), villain_hand.clone(), board.clone(), mode.clone(), hi_lo);
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
        win_low: if hi_lo && total_weight_low > 0.0 { Some(total_win_low / total_weight_low) } else { None },
        tie_low: if hi_lo && total_weight_low > 0.0 { Some(total_tie_low / total_weight_low) } else { None },
        loss_low: if hi_lo && total_weight_low > 0.0 { Some(total_loss_low / total_weight_low) } else { None },
        trial_count: total_trials,
        mode,
        confidence_interval: None,
    }
}

pub fn random_hand(dead_cards: &[Card], rng_seed: Option<u64>) -> Hand {
    let mut deck = Vec::new();
    for s in Suit::all() {
        for r in Rank::all() {
            let card = Card::new(r, s);
            if !dead_cards.contains(&card) {
                deck.push(card);
            }
        }
    }

    let mut rng = if let Some(seed) = rng_seed {
        Pcg64::seed_from_u64(seed)
    } else {
        Pcg64::from_entropy()
    };

    deck.shuffle(&mut rng);
    Hand::new([deck[0], deck[1], deck[2], deck[3]])
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

        let cpu_res = evaluate_hand_vs_hand(hero.clone(), villain.clone(), board.clone(), EvalMode::Exhaustive, false);
        let gpu_res = crate::gpu::run_gpu_evaluation(&hero, &villain, &board, &EvalMode::Exhaustive, false);

        if let Some(gpu_res) = gpu_res {
            assert!((cpu_res.win - gpu_res.win).abs() < 1e-6);
            assert!((cpu_res.tie - gpu_res.tie).abs() < 1e-6);
            assert!((cpu_res.loss - gpu_res.loss).abs() < 1e-6);
        } else {
            println!("GPU not available, skipping test");
        }
    }

    #[test]
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

        let gpu_res = crate::gpu::run_gpu_evaluation(&hero, &villain, &board, &EvalMode::Exhaustive, false);
        if let Some(res) = gpu_res {
            assert!(res.win + res.tie + res.loss > 0.0);
        }
    }
}
