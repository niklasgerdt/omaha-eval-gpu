use crate::types::*;
use crate::eval::{HandRank, evaluate_5_cards, evaluate_5_cards_low};
use crate::gpu;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64;

pub fn evaluate_omaha_hand(hand: &Hand, board: &[Card]) -> HandRank {
    let mut best_rank = HandRank::HighCard(Rank::Two, Rank::Two, Rank::Two, Rank::Two, Rank::Two);

    if board.len() < 3 {
        return best_rank;
    }

    let hand_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for &(h1, h2) in &hand_indices {
        for i in 0..board.len() {
            for j in (i + 1)..board.len() {
                for k in (j + 1)..board.len() {
                    let cards = [hand.0[h1], hand.0[h2], board[i], board[j], board[k]];
                    let rank = evaluate_5_cards(&cards);
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

    let hand_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    for &(h1, h2) in &hand_indices {
        for i in 0..board.len() {
            for j in (i + 1)..board.len() {
                for k in (j + 1)..board.len() {
                    let cards = [hand.0[h1], hand.0[h2], board[i], board[j], board[k]];
                    if let Some(low_rank) = evaluate_5_cards_low(&cards) {
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

pub fn evaluate_hand_vs_hand(hero: Hand, villain: Hand, board: Board, mode: EvalMode, hi_lo: bool) -> EquityResult {
    let mode = match mode {
        EvalMode::Auto => {
            if board.0.len() >= 3 {
                EvalMode::Exhaustive
            } else {
                EvalMode::MonteCarlo { samples: 100_000, seed: 42 }
            }
        }
        _ => mode,
    };

    match mode {
        EvalMode::Exhaustive => {
            let deck: Vec<Card> = full_deck()
                .into_iter()
                .filter(|c| !hero.0.contains(c) && !villain.0.contains(c) && !board.0.contains(c))
                .collect();

            let mut win = 0.0;
            let mut tie = 0.0;
            let mut loss = 0.0;
            let mut win_low = 0.0;
            let mut tie_low = 0.0;
            let mut loss_low = 0.0;
            let mut trials = 0;
            let mut trials_low = 0;

            let missing = 5 - board.0.len();
            if missing == 0 {
                let (h_win, v_win, h_win_low, v_win_low) = compare_hands(&hero, &villain, &board.0, hi_lo);
                trials = 1;
                if h_win > v_win { win = 1.0; } else if v_win > h_win { loss = 1.0; } else { tie = 1.0; }
                if hi_lo {
                    if let (Some(hl), Some(vl)) = (h_win_low, v_win_low) {
                        trials_low = 1;
                        if hl < vl { win_low = 1.0; } else if vl < hl { loss_low = 1.0; } else { tie_low = 1.0; }
                    }
                }
            } else {
                let n = deck.len();
                match missing {
                    1 => {
                        for i in 0..n {
                            let mut b = board.0.clone();
                            b.push(deck[i]);
                            let (h_win, v_win, h_win_low, v_win_low) = compare_hands(&hero, &villain, &b, hi_lo);
                            trials += 1;
                            if h_win > v_win { win += 1.0; } else if v_win > h_win { loss += 1.0; } else { tie += 1.0; }
                            if hi_lo {
                                if let (Some(hl), Some(vl)) = (h_win_low, v_win_low) {
                                    trials_low += 1;
                                    if hl < vl { win_low += 1.0; } else if vl < hl { loss_low += 1.0; } else { tie_low += 1.0; }
                                }
                            }
                        }
                    }
                    2 => {
                        for i in 0..n {
                            for j in i + 1..n {
                                let mut b = board.0.clone();
                                b.push(deck[i]);
                                b.push(deck[j]);
                                let (h_win, v_win, h_win_low, v_win_low) = compare_hands(&hero, &villain, &b, hi_lo);
                                trials += 1;
                                if h_win > v_win { win += 1.0; } else if v_win > h_win { loss += 1.0; } else { tie += 1.0; }
                                if hi_lo {
                                    if let (Some(hl), Some(vl)) = (h_win_low, v_win_low) {
                                        trials_low += 1;
                                        if hl < vl { win_low += 1.0; } else if vl < hl { loss_low += 1.0; } else { tie_low += 1.0; }
                                    }
                                }
                            }
                        }
                    }
                    _ => unreachable!("Exhaustive only for missing <= 2"),
                }
            }

            EquityResult {
                win: win / trials as f64,
                tie: tie / trials as f64,
                loss: loss / trials as f64,
                win_low: if trials_low > 0 { Some(win_low / trials_low as f64) } else { None },
                tie_low: if trials_low > 0 { Some(tie_low / trials_low as f64) } else { None },
                loss_low: if trials_low > 0 { Some(loss_low / trials_low as f64) } else { None },
                trial_count: trials,
                mode: EvalMode::Exhaustive,
                confidence_interval: None,
            }
        }
        EvalMode::MonteCarlo { samples, seed } => {
            let mut rng = Pcg64::seed_from_u64(seed);
            let deck: Vec<Card> = full_deck()
                .into_iter()
                .filter(|c| !hero.0.contains(c) && !villain.0.contains(c) && !board.0.contains(c))
                .collect();

            let mut win = 0.0;
            let mut tie = 0.0;
            let mut loss = 0.0;
            let mut win_low = 0.0;
            let mut tie_low = 0.0;
            let mut loss_low = 0.0;
            let mut trials_low = 0;

            let missing = 5 - board.0.len();

            for _ in 0..samples {
                let chosen = deck.choose_multiple(&mut rng, missing).cloned().collect::<Vec<_>>();
                let mut b = board.0.clone();
                b.extend(chosen);
                let (h_win, v_win, h_win_low, v_win_low) = compare_hands(&hero, &villain, &b, hi_lo);
                if h_win > v_win { win += 1.0; } else if v_win > h_win { loss += 1.0; } else { tie += 1.0; }
                if hi_lo {
                    if let (Some(hl), Some(vl)) = (h_win_low, v_win_low) {
                        trials_low += 1;
                        if hl < vl { win_low += 1.0; } else if vl < hl { loss_low += 1.0; } else { tie_low += 1.0; }
                    }
                }
            }

            EquityResult {
                win: win / samples as f64,
                tie: tie / samples as f64,
                loss: loss / samples as f64,
                win_low: if trials_low > 0 { Some(win_low / trials_low as f64) } else { None },
                tie_low: if trials_low > 0 { Some(tie_low / trials_low as f64) } else { None },
                loss_low: if trials_low > 0 { Some(loss_low / trials_low as f64) } else { None },
                trial_count: samples,
                mode: EvalMode::MonteCarlo { samples, seed },
                confidence_interval: None,
            }
        }
    }
}

fn compare_hands(hero: &Hand, villain: &Hand, board: &[Card], hi_lo: bool) -> (HandRank, HandRank, Option<[Rank; 5]>, Option<[Rank; 5]>) {
    let hr = evaluate_omaha_hand(hero, board);
    let vr = evaluate_omaha_hand(villain, board);
    let mut hl = None;
    let mut vl = None;
    if hi_lo {
        hl = evaluate_omaha_hand_low(hero, board);
        vl = evaluate_omaha_hand_low(villain, board);
    }
    (hr, vr, hl, vl)
}

pub fn evaluate_hand_vs_range(hero: Hand, villain_range: Range, board: Board, mode: EvalMode, hi_lo: bool) -> EquityResult {
    let mut total_win = 0.0;
    let mut total_tie = 0.0;
    let mut total_loss = 0.0;
    let mut total_win_low = 0.0;
    let mut total_tie_low = 0.0;
    let mut total_loss_low = 0.0;
    let mut total_weight = 0.0;
    let mut total_weight_low = 0.0;

    for (v_hand, weight) in villain_range.hands {
        if v_hand.0.iter().any(|c| hero.0.contains(c) || board.0.contains(c)) {
            continue;
        }

        let res = evaluate_hand_vs_hand(hero.clone(), v_hand, board.clone(), mode.clone(), hi_lo);
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
    }

    EquityResult {
        win: total_win / total_weight,
        tie: total_tie / total_weight,
        loss: total_loss / total_weight,
        win_low: if total_weight_low > 0.0 { Some(total_win_low / total_weight_low) } else { None },
        tie_low: if total_weight_low > 0.0 { Some(total_tie_low / total_weight_low) } else { None },
        loss_low: if total_weight_low > 0.0 { Some(total_loss_low / total_weight_low) } else { None },
        trial_count: 0,
        mode,
        confidence_interval: None,
    }
}

pub fn evaluate_range_vs_range(hero_range: Range, villain_range: Range, board: Board, mode: EvalMode, hi_lo: bool, backend: Backend) -> EquityResult {
    if backend != Backend::Cpu && backend.is_available() {
        if let Some(res) = gpu::run_gpu_range_evaluation(&hero_range, &villain_range, &board, &mode) {
            return res;
        }
    }
    evaluate_range_vs_range_internal(hero_range, villain_range, board, mode, hi_lo)
}

fn evaluate_range_vs_range_internal(hero_range: Range, villain_range: Range, board: Board, mode: EvalMode, hi_lo: bool) -> EquityResult {
    let mut total_win = 0.0;
    let mut total_tie = 0.0;
    let mut total_loss = 0.0;
    let mut total_win_low = 0.0;
    let mut total_tie_low = 0.0;
    let mut total_loss_low = 0.0;
    let mut total_weight = 0.0;
    let mut total_weight_low = 0.0;

    for (h_hand, h_weight) in &hero_range.hands {
        if h_hand.0.iter().any(|c| board.0.contains(c)) {
            continue;
        }
        let res = evaluate_hand_vs_range(h_hand.clone(), villain_range.clone(), board.clone(), mode.clone(), hi_lo);
        total_win += res.win * h_weight;
        total_tie += res.tie * h_weight;
        total_loss += res.loss * h_weight;
        if let (Some(wl), Some(tl), Some(ll)) = (res.win_low, res.tie_low, res.loss_low) {
            total_win_low += wl * h_weight;
            total_tie_low += tl * h_weight;
            total_loss_low += ll * h_weight;
            total_weight_low += h_weight;
        }
        total_weight += h_weight;
    }

    EquityResult {
        win: total_win / total_weight,
        tie: total_tie / total_weight,
        loss: total_loss / total_weight,
        win_low: if total_weight_low > 0.0 { Some(total_win_low / total_weight_low) } else { None },
        tie_low: if total_weight_low > 0.0 { Some(total_tie_low / total_weight_low) } else { None },
        loss_low: if total_weight_low > 0.0 { Some(total_loss_low / total_weight_low) } else { None },
        trial_count: 0,
        mode,
        confidence_interval: None,
    }
}

pub fn random_hand(dead_cards: &[Card], rng_seed: Option<u64>) -> Hand {
    let mut rng = if let Some(s) = rng_seed { Pcg64::seed_from_u64(s) } else { Pcg64::from_entropy() };
    let mut deck = full_deck();
    deck.retain(|c| !dead_cards.contains(c));
    let chosen = deck.choose_multiple(&mut rng, 4).cloned().collect::<Vec<_>>();
    let mut arr = [Card::new(Rank::Two, crate::types::card::Suit::Spades); 4];
    arr.copy_from_slice(&chosen);
    Hand::new(arr)
}

pub fn random_hands(count: usize, dead_cards: &[Card], rng_seed: Option<u64>) -> Vec<Hand> {
    let mut rng = if let Some(s) = rng_seed { Pcg64::seed_from_u64(s) } else { Pcg64::from_entropy() };
    let mut deck = full_deck();
    deck.retain(|c| !dead_cards.contains(c));
    let mut hands = Vec::with_capacity(count);
    for _ in 0..count {
        deck.shuffle(&mut rng);
        let mut arr = [Card::new(Rank::Two, crate::types::card::Suit::Spades); 4];
        arr.copy_from_slice(&deck[0..4]);
        hands.push(Hand::new(arr));
    }
    hands
}
