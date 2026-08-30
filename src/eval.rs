use crate::{Card, Rank};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandRank {
    HighCard(Rank, Rank, Rank, Rank, Rank),
    OnePair(Rank, Rank, Rank, Rank),
    TwoPair(Rank, Rank, Rank),
    ThreeOfAKind(Rank, Rank, Rank),
    Straight(Rank),
    Flush(Rank, Rank, Rank, Rank, Rank),
    FullHouse(Rank, Rank),
    FourOfAKind(Rank, Rank),
    StraightFlush(Rank),
}

pub fn evaluate_5_cards(cards: &[Card; 5]) -> HandRank {
    let mut ranks = [Rank::Two; 5];
    for i in 0..5 {
        ranks[i] = cards[i].rank;
    }
    ranks.sort_by(|a, b| b.cmp(a));

    let is_flush = cards[0].suit == cards[1].suit &&
                   cards[0].suit == cards[2].suit &&
                   cards[0].suit == cards[3].suit &&
                   cards[0].suit == cards[4].suit;
    
    let is_straight = {
        let mut unique_len = 1;
        for i in 1..5 {
            if ranks[i] != ranks[i-1] {
                unique_len += 1;
            }
        }
        if unique_len == 5 {
            if (ranks[0] as u32) - (ranks[4] as u32) == 4 {
                Some(ranks[0])
            } else if ranks[0] == Rank::Ace && ranks[1] == Rank::Five {
                Some(Rank::Five) // A-5 straight
            } else {
                None
            }
        } else {
            None
        }
    };

    if let Some(high_rank) = is_straight {
        if is_flush {
            return HandRank::StraightFlush(high_rank);
        }
    }

    let mut counts = [(Rank::Two, 0u8); 5];
    let mut num_unique = 0;

    for &r in &ranks {
        let mut found = false;
        for i in 0..num_unique {
            if counts[i].0 == r {
                counts[i].1 += 1;
                found = true;
                break;
            }
        }
        if !found {
            counts[num_unique] = (r, 1);
            num_unique += 1;
        }
    }

    // Sort by count descending, then by rank descending
    let sorted_counts = &mut counts[0..num_unique];
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

    if sorted_counts[0].1 == 4 {
        return HandRank::FourOfAKind(sorted_counts[0].0, sorted_counts[1].0);
    }
    if sorted_counts[0].1 == 3 && sorted_counts[1].1 >= 2 {
        return HandRank::FullHouse(sorted_counts[0].0, sorted_counts[1].0);
    }
    
    if is_flush {
        return HandRank::Flush(ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]);
    }
    if let Some(high_rank) = is_straight {
        return HandRank::Straight(high_rank);
    }

    match sorted_counts[0].1 {
        3 => HandRank::ThreeOfAKind(sorted_counts[0].0, sorted_counts[1].0, sorted_counts[2].0),
        2 if sorted_counts.len() > 1 && sorted_counts[1].1 == 2 => HandRank::TwoPair(sorted_counts[0].0, sorted_counts[1].0, sorted_counts[2].0),
        2 => HandRank::OnePair(sorted_counts[0].0, sorted_counts[1].0, sorted_counts[2].0, sorted_counts[3].0),
        _ => HandRank::HighCard(ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]),
    }
}

pub fn evaluate_5_cards_low(cards: &[Card; 5]) -> Option<[Rank; 5]> {
    let mut unique_ranks = cards.iter().map(|c| c.rank).collect::<Vec<_>>();
    unique_ranks.sort_by(|a, b| {
        let val_a = if *a == Rank::Ace { 1 } else { *a as u32 };
        let val_b = if *b == Rank::Ace { 1 } else { *b as u32 };
        val_a.cmp(&val_b)
    });
    unique_ranks.dedup();

    // Ace is 1, so filter ranks > 8
    let low_ranks: Vec<Rank> = unique_ranks.into_iter().filter(|&r| {
        let val = if r == Rank::Ace { 1 } else { r as u32 };
        val <= 8
    }).collect();

    if low_ranks.len() < 5 {
        return None;
    }

    // Return the best 5 low ranks (lowest 5), but traditionally low hands are compared from highest rank down
    // The "best" low hand is the one with the lowest high rank.
    // So we take the 5 lowest ranks, and sort them descending for comparison.
    let mut best_low = [low_ranks[0], low_ranks[1], low_ranks[2], low_ranks[3], low_ranks[4]];
    best_low.sort_by(|a, b| {
        let val_a = if *a == Rank::Ace { 1 } else { *a as u32 };
        let val_b = if *b == Rank::Ace { 1 } else { *b as u32 };
        val_b.cmp(&val_a)
    });

    Some(best_low)
}
