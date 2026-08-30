struct GpuCaseInput {
    hero_hands: array<array<u32, 4>, 128>,
    villain_hands: array<array<u32, 4>, 128>,
    hero_weights: array<f32, 128>,
    villain_weights: array<f32, 128>,
    hero_count: u32,
    villain_count: u32,
    board: array<u32, 5>,
    board_len: u32,
    mode: u32,
    samples: u32,
    seed: u32,
    padding: u32,
}

struct GpuInput {
    cases: array<GpuCaseInput, 256>,
}

@group(0) @binding(0) var<storage, read> input: GpuInput;
@group(0) @binding(1) var<storage, read_write> results: array<atomic<u32>, 1024>;

fn get_card_rank(card: u32) -> u32 {
    return card % 13u;
}

fn get_card_suit(card: u32) -> u32 {
    return card / 13u;
}

fn evaluate_5_cards(c0: u32, c1: u32, c2: u32, c3: u32, c4: u32) -> u32 {
    var ranks = array<u32, 5>(c0, c1, c2, c3, c4);
    for (var i = 0u; i < 4u; i++) {
        for (var j = i + 1u; j < 5u; j++) {
            if (get_card_rank(ranks[j]) > get_card_rank(ranks[i])) {
                let temp = ranks[i];
                ranks[i] = ranks[j];
                ranks[j] = temp;
            }
        }
    }

    let is_flush = get_card_suit(ranks[0]) == get_card_suit(ranks[1]) &&
                   get_card_suit(ranks[0]) == get_card_suit(ranks[2]) &&
                   get_card_suit(ranks[0]) == get_card_suit(ranks[3]) &&
                   get_card_suit(ranks[0]) == get_card_suit(ranks[4]);

    var is_straight = false;
    var straight_high = 0u;

    let r0 = get_card_rank(ranks[0]);
    let r1 = get_card_rank(ranks[1]);
    let r2 = get_card_rank(ranks[2]);
    let r3 = get_card_rank(ranks[3]);
    let r4 = get_card_rank(ranks[4]);

    if (r0 == r1 + 1 && r1 == r2 + 1 && r2 == r3 + 1 && r3 == r4 + 1) {
        is_straight = true;
        straight_high = r0;
    } else if (r0 == 12u && r1 == 3u && r2 == 2u && r3 == 1u && r4 == 0u) {
        is_straight = true;
        straight_high = 3u;
    }

    if (is_straight && is_flush) {
        return 8000000u + straight_high;
    }

    var counts = array<u32, 13>(0,0,0,0,0,0,0,0,0,0,0,0,0);
    counts[get_card_rank(ranks[0])]++;
    counts[get_card_rank(ranks[1])]++;
    counts[get_card_rank(ranks[2])]++;
    counts[get_card_rank(ranks[3])]++;
    counts[get_card_rank(ranks[4])]++;

    var four = 13u;
    var three = 13u;
    var pairs = array<u32, 2>(13u, 13u);
    var pair_count = 0u;

    for (var i = 12i; i >= 0i; i--) {
        let u_i = u32(i);
        if (counts[u_i] == 4u) { four = u_i; }
        else if (counts[u_i] == 3u) { three = u_i; }
        else if (counts[u_i] == 2u) {
            if (pair_count < 2u) {
                pairs[pair_count] = u_i;
                pair_count++;
            }
        }
    }

    if (four != 13u) {
        var kicker = 0u;
        for(var i=0u; i<5u; i++) { if(get_card_rank(ranks[i]) != four) { kicker = get_card_rank(ranks[i]); break; } }
        return 7000000u + four * 15u + kicker;
    }
    if (three != 13u && pair_count >= 1u) {
        return 6000000u + three * 15u + pairs[0];
    }
    if (is_flush) {
        return 5000000u + r0 * 50625u + r1 * 3375u + r2 * 225u + r3 * 15u + r4;
    }
    if (is_straight) {
        return 4000000u + straight_high;
    }
    if (three != 13u) {
        var k1 = 13u; var k2 = 13u;
        for(var i=0u; i<5u; i++) { 
            let r = get_card_rank(ranks[i]);
            if(r != three) { if(k1 == 13u) { k1 = r; } else if(k2 == 13u) { k2 = r; } }
        }
        return 3000000u + three * 225u + k1 * 15u + k2;
    }
    if (pair_count >= 2u) {
        var kicker = 0u;
        for(var i=0u; i<5u; i++) { 
            let r = get_card_rank(ranks[i]);
            if(r != pairs[0] && r != pairs[1]) { kicker = r; break; }
        }
        return 2000000u + pairs[0] * 225u + pairs[1] * 15u + kicker;
    }
    if (pair_count == 1u) {
        var k = array<u32, 3>(0u, 0u, 0u);
        var kc = 0u;
        for(var i=0u; i<5u; i++) { 
            let r = get_card_rank(ranks[i]);
            if(r != pairs[0]) { k[kc] = r; kc++; }
        }
        return 1000000u + pairs[0] * 3375u + k[0] * 225u + k[1] * 15u + k[2];
    }

    return r0 * 50625u + r1 * 3375u + r2 * 225u + r3 * 15u + r4;
}

fn evaluate_omaha(h0: u32, h1: u32, h2: u32, h3: u32, b0: u32, b1: u32, b2: u32, b3: u32, b4: u32) -> u32 {
    var best = 0u;
    best = max(best, evaluate_5_cards(h0, h1, b0, b1, b2));
    best = max(best, evaluate_5_cards(h0, h1, b0, b1, b3));
    best = max(best, evaluate_5_cards(h0, h1, b0, b1, b4));
    best = max(best, evaluate_5_cards(h0, h1, b0, b2, b3));
    best = max(best, evaluate_5_cards(h0, h1, b0, b2, b4));
    best = max(best, evaluate_5_cards(h0, h1, b0, b3, b4));
    best = max(best, evaluate_5_cards(h0, h1, b1, b2, b3));
    best = max(best, evaluate_5_cards(h0, h1, b1, b2, b4));
    best = max(best, evaluate_5_cards(h0, h1, b1, b3, b4));
    best = max(best, evaluate_5_cards(h0, h1, b2, b3, b4));
    
    best = max(best, evaluate_5_cards(h0, h2, b0, b1, b2));
    best = max(best, evaluate_5_cards(h0, h2, b0, b1, b3));
    best = max(best, evaluate_5_cards(h0, h2, b0, b1, b4));
    best = max(best, evaluate_5_cards(h0, h2, b0, b2, b3));
    best = max(best, evaluate_5_cards(h0, h2, b0, b2, b4));
    best = max(best, evaluate_5_cards(h0, h2, b0, b3, b4));
    best = max(best, evaluate_5_cards(h0, h2, b1, b2, b3));
    best = max(best, evaluate_5_cards(h0, h2, b1, b2, b4));
    best = max(best, evaluate_5_cards(h0, h2, b1, b3, b4));
    best = max(best, evaluate_5_cards(h0, h2, b2, b3, b4));

    best = max(best, evaluate_5_cards(h0, h3, b0, b1, b2));
    best = max(best, evaluate_5_cards(h0, h3, b0, b1, b3));
    best = max(best, evaluate_5_cards(h0, h3, b0, b1, b4));
    best = max(best, evaluate_5_cards(h0, h3, b0, b2, b3));
    best = max(best, evaluate_5_cards(h0, h3, b0, b2, b4));
    best = max(best, evaluate_5_cards(h0, h3, b0, b3, b4));
    best = max(best, evaluate_5_cards(h0, h3, b1, b2, b3));
    best = max(best, evaluate_5_cards(h0, h3, b1, b2, b4));
    best = max(best, evaluate_5_cards(h0, h3, b1, b3, b4));
    best = max(best, evaluate_5_cards(h0, h3, b2, b3, b4));

    best = max(best, evaluate_5_cards(h1, h2, b0, b1, b2));
    best = max(best, evaluate_5_cards(h1, h2, b0, b1, b3));
    best = max(best, evaluate_5_cards(h1, h2, b0, b1, b4));
    best = max(best, evaluate_5_cards(h1, h2, b0, b2, b3));
    best = max(best, evaluate_5_cards(h1, h2, b0, b2, b4));
    best = max(best, evaluate_5_cards(h1, h2, b0, b3, b4));
    best = max(best, evaluate_5_cards(h1, h2, b1, b2, b3));
    best = max(best, evaluate_5_cards(h1, h2, b1, b2, b4));
    best = max(best, evaluate_5_cards(h1, h2, b1, b3, b4));
    best = max(best, evaluate_5_cards(h1, h2, b2, b3, b4));

    best = max(best, evaluate_5_cards(h1, h3, b0, b1, b2));
    best = max(best, evaluate_5_cards(h1, h3, b0, b1, b3));
    best = max(best, evaluate_5_cards(h1, h3, b0, b1, b4));
    best = max(best, evaluate_5_cards(h1, h3, b0, b2, b3));
    best = max(best, evaluate_5_cards(h1, h3, b0, b2, b4));
    best = max(best, evaluate_5_cards(h1, h3, b0, b3, b4));
    best = max(best, evaluate_5_cards(h1, h3, b1, b2, b3));
    best = max(best, evaluate_5_cards(h1, h3, b1, b2, b4));
    best = max(best, evaluate_5_cards(h1, h3, b1, b3, b4));
    best = max(best, evaluate_5_cards(h1, h3, b2, b3, b4));

    best = max(best, evaluate_5_cards(h2, h3, b0, b1, b2));
    best = max(best, evaluate_5_cards(h2, h3, b0, b1, b3));
    best = max(best, evaluate_5_cards(h2, h3, b0, b1, b4));
    best = max(best, evaluate_5_cards(h2, h3, b0, b2, b3));
    best = max(best, evaluate_5_cards(h2, h3, b0, b2, b4));
    best = max(best, evaluate_5_cards(h2, h3, b0, b3, b4));
    best = max(best, evaluate_5_cards(h2, h3, b1, b2, b3));
    best = max(best, evaluate_5_cards(h2, h3, b1, b2, b4));
    best = max(best, evaluate_5_cards(h2, h3, b1, b3, b4));
    best = max(best, evaluate_5_cards(h2, h3, b2, b3, b4));

    return best;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let case_idx = global_id.y;
    let pair_index = global_id.x;
    
    if (case_idx >= 256u) { return; }
    
    let hero_idx = pair_index / input.cases[case_idx].villain_count;
    let villain_idx = pair_index % input.cases[case_idx].villain_count;

    if (hero_idx >= input.cases[case_idx].hero_count) { return; }

    let hero_h0 = input.cases[case_idx].hero_hands[hero_idx][0];
    let hero_h1 = input.cases[case_idx].hero_hands[hero_idx][1];
    let hero_h2 = input.cases[case_idx].hero_hands[hero_idx][2];
    let hero_h3 = input.cases[case_idx].hero_hands[hero_idx][3];

    let villain_h0 = input.cases[case_idx].villain_hands[villain_idx][0];
    let villain_h1 = input.cases[case_idx].villain_hands[villain_idx][1];
    let villain_h2 = input.cases[case_idx].villain_hands[villain_idx][2];
    let villain_h3 = input.cases[case_idx].villain_hands[villain_idx][3];

    let weight = input.cases[case_idx].hero_weights[hero_idx] * input.cases[case_idx].villain_weights[villain_idx];

    // Check overlaps
    let b0 = input.cases[case_idx].board[0];
    let b1 = input.cases[case_idx].board[1];
    let b2 = input.cases[case_idx].board[2];
    let b3 = input.cases[case_idx].board[3];
    let b4 = input.cases[case_idx].board[4];

    if (hero_h0 == villain_h0 || hero_h0 == villain_h1 || hero_h0 == villain_h2 || hero_h0 == villain_h3) { return; }
    if (hero_h1 == villain_h0 || hero_h1 == villain_h1 || hero_h1 == villain_h2 || hero_h1 == villain_h3) { return; }
    if (hero_h2 == villain_h0 || hero_h2 == villain_h1 || hero_h2 == villain_h2 || hero_h2 == villain_h3) { return; }
    if (hero_h3 == villain_h0 || hero_h3 == villain_h1 || hero_h3 == villain_h2 || hero_h3 == villain_h3) { return; }

    if (hero_h0 == b0 || hero_h0 == b1 || hero_h0 == b2 || hero_h0 == b3 || hero_h0 == b4) {
        if (hero_h0 != 255u) { return; }
    }
    if (hero_h1 == b0 || hero_h1 == b1 || hero_h1 == b2 || hero_h1 == b3 || hero_h1 == b4) {
        if (hero_h1 != 255u) { return; }
    }
    if (hero_h2 == b0 || hero_h2 == b1 || hero_h2 == b2 || hero_h2 == b3 || hero_h2 == b4) {
        if (hero_h2 != 255u) { return; }
    }
    if (hero_h3 == b0 || hero_h3 == b1 || hero_h3 == b2 || hero_h3 == b3 || hero_h3 == b4) {
        if (hero_h3 != 255u) { return; }
    }

    if (villain_h0 == b0 || villain_h0 == b1 || villain_h0 == b2 || villain_h0 == b3 || villain_h0 == b4) {
        if (villain_h0 != 255u) { return; }
    }
    if (villain_h1 == b0 || villain_h1 == b1 || villain_h1 == b2 || villain_h1 == b3 || villain_h1 == b4) {
        if (villain_h1 != 255u) { return; }
    }
    if (villain_h2 == b0 || villain_h2 == b1 || villain_h2 == b2 || villain_h2 == b3 || villain_h2 == b4) {
        if (villain_h2 != 255u) { return; }
    }
    if (villain_h3 == b0 || villain_h3 == b1 || villain_h3 == b2 || villain_h3 == b3 || villain_h3 == b4) {
        if (villain_h3 != 255u) { return; }
    }

    var win = 0.0;
    var tie = 0.0;
    var loss = 0.0;

    if (input.cases[case_idx].board_len == 5u) {
        let hero_score = evaluate_omaha(hero_h0, hero_h1, hero_h2, hero_h3, b0, b1, b2, b3, b4);
        let villain_score = evaluate_omaha(villain_h0, villain_h1, villain_h2, villain_h3, b0, b1, b2, b3, b4);
        if (hero_score > villain_score) { win = 1.0; }
        else if (hero_score == villain_score) { tie = 1.0; }
        else { loss = 1.0; }
    } else {
        // GPU evaluation for boards with < 5 cards is currently disabled in shader
        // to prevent stack issues and ensure performance. 
        // CPU fallback should be used.
        win = 0.0;
        tie = 0.0;
        loss = 0.0;
    }

    let scale = 1000u;
    let offset = case_idx * 4u;
    if (win > 0.0) { atomicAdd(&results[offset + 0u], u32(weight * win * f32(scale))); }
    if (tie > 0.0) { atomicAdd(&results[offset + 1u], u32(weight * tie * f32(scale))); }
    if (loss > 0.0) { atomicAdd(&results[offset + 2u], u32(weight * loss * f32(scale))); }
    atomicAdd(&results[offset + 3u], u32(weight * f32(scale)));
}

fn pcg_hash(input_seed: u32) -> u32 {
    let state = input_seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random_u32(state: ptr<function, u32>) -> u32 {
    *state = pcg_hash(*state);
    return *state;
}
