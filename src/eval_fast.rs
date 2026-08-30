use crate::{Card, Rank, Suit};

/// A fast 5-card evaluator using a lookup table (simplified for now).
/// In a real implementation, this would use a precomputed table.
/// For the purpose of this task, we'll use a slightly more efficient version
/// of the existing logic that can be easily ported to WGSL.
pub fn evaluate_5_cards_fast(cards: &[u8; 5]) -> u32 {
    // This is a placeholder for a real lookup-table evaluator.
    // For now, it converts card indices to the existing HandRank.
    let mut actual_cards = [Card::new(Rank::Two, Suit::Spades); 5];
    for i in 0..5 {
        actual_cards[i] = card_index_to_card(cards[i]);
    }
    
    // We return a u32 score where higher is better.
    let rank = crate::eval::evaluate_5_cards(&actual_cards);
    hand_rank_to_score(rank)
}

fn card_index_to_card(index: u8) -> Card {
    let rank = match index % 13 {
        0 => Rank::Two,
        1 => Rank::Three,
        2 => Rank::Four,
        3 => Rank::Five,
        4 => Rank::Six,
        5 => Rank::Seven,
        6 => Rank::Eight,
        7 => Rank::Nine,
        8 => Rank::Ten,
        9 => Rank::Jack,
        10 => Rank::Queen,
        11 => Rank::King,
        12 => Rank::Ace,
        _ => unreachable!(),
    };
    let suit = match index / 13 {
        0 => Suit::Spades,
        1 => Suit::Hearts,
        2 => Suit::Diamonds,
        3 => Suit::Clubs,
        _ => unreachable!(),
    };
    Card::new(rank, suit)
}

pub fn card_to_index(card: Card) -> u8 {
    let r = card.rank as u8 - 2;
    let s = card.suit as u8;
    s * 13 + r
}

fn hand_rank_to_score(rank: crate::eval::HandRank) -> u32 {
    use crate::eval::HandRank::*;
    match rank {
        StraightFlush(r) => 8000000 + (r as u32),
        FourOfAKind(r1, r2) => 7000000 + (r1 as u32) * 15 + (r2 as u32),
        FullHouse(r1, r2) => 6000000 + (r1 as u32) * 15 + (r2 as u32),
        Flush(r1, r2, r3, r4, r5) => 5000000 + (r1 as u32) * 15u32.pow(4) + (r2 as u32) * 15u32.pow(3) + (r3 as u32) * 15u32.pow(2) + (r4 as u32) * 15 + (r5 as u32),
        Straight(r) => 4000000 + (r as u32),
        ThreeOfAKind(r1, r2, r3) => 3000000 + (r1 as u32) * 15u32.pow(2) + (r2 as u32) * 15 + (r3 as u32),
        TwoPair(r1, r2, r3) => 2000000 + (r1 as u32) * 15u32.pow(2) + (r2 as u32) * 15 + (r3 as u32),
        OnePair(r1, r2, r3, r4) => 1000000 + (r1 as u32) * 15u32.pow(3) + (r2 as u32) * 15u32.pow(2) + (r3 as u32) * 15 + (r4 as u32),
        HighCard(r1, r2, r3, r4, r5) => (r1 as u32) * 15u32.pow(4) + (r2 as u32) * 15u32.pow(3) + (r3 as u32) * 15u32.pow(2) + (r4 as u32) * 15 + (r5 as u32),
    }
}
