use serde::{Deserialize, Serialize};
use std::fmt;

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
        (*self as u8).cmp(&(*other as u8))
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

    pub fn to_char(self) -> char {
        match self {
            Suit::Spades => 's',
            Suit::Hearts => 'h',
            Suit::Diamonds => 'd',
            Suit::Clubs => 'c',
        }
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
        if s.len() != 2 { return None; }
        let mut chars = s.chars();
        let r = Rank::from_char(chars.next()?)?;
        let s = Suit::from_char(chars.next()?)?;
        Some(Card::new(r, s))
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank.to_char(), self.suit.to_char())
    }
}

pub fn full_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for s in Suit::all() {
        for r in Rank::all() {
            deck.push(Card::new(r, s));
        }
    }
    deck
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
