use crate::types::card::{Card, Hand, Rank, full_deck};
use serde::{Deserialize, Serialize};

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

            if part.len() == 8 {
                let mut cards = [Card::new(Rank::Two, crate::types::card::Suit::Spades); 4];
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

            if part.len() == 2 || part.len() == 4 {
                let pattern_ranks: Vec<Rank> = part
                    .chars()
                    .map(|c| Rank::from_char(c).ok_or(format!("Invalid rank: {}", c)))
                    .collect::<Result<Vec<_>, _>>()?;

                let deck: Vec<Card> = full_deck()
                    .into_iter()
                    .filter(|c| !dead_cards.contains(c))
                    .collect();

                let n = deck.len();
                if n < 4 { continue; }

                let mut combos = Vec::new();
                for i in 0..n {
                    for j in i + 1..n {
                        for k in j + 1..n {
                            for l in k + 1..n {
                                let h = [deck[i], deck[j], deck[k], deck[l]];
                                
                                let mut match_found = true;
                                if pattern_ranks.len() == 2 && pattern_ranks[0] == pattern_ranks[1] {
                                    let target = pattern_ranks[0];
                                    let count = h.iter().filter(|c| c.rank == target).count();
                                    if count < 2 { match_found = false; }
                                } else {
                                    for &pr in &pattern_ranks {
                                        if !h.iter().any(|c| c.rank == pr) {
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
                        let mut arr = [Card::new(Rank::Two, crate::types::card::Suit::Spades); 4];
                        arr.copy_from_slice(&cards);
                        hands.push((Hand::new(arr), 1.0));
                    }
                    continue;
                }
            }

            return Err(format!("Invalid range part: {}", part));
        }

        Ok(Range { hands })
    }
}
