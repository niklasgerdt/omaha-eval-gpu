use plo_eval_gpu::{Card, Board, Range, EvalMode, Backend, evaluate_range_vs_range};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::collections::HashMap;
use clap::Parser;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the validation checklist (JSONL or plain text format)
    #[arg(short, long)]
    input: String,

    /// Backend to use (auto, cpu, cuda, vulkan, metal)
    #[arg(short, long, default_value = "auto")]
    backend: String,

    /// Evaluation mode (auto, exhaustive, monte-carlo)
    #[arg(short, long, default_value = "auto")]
    mode: String,

    /// Number of samples for Monte Carlo
    #[arg(short, long, default_value_t = 100000)]
    samples: u64,

    /// Tolerance for equity difference (0.1 = 10%)
    #[arg(short, long, default_value_t = 0.1)]
    tolerance: f64,

    /// Output log file
    #[arg(short, long, default_value = "test_results.log")]
    output: String,

    /// Path to ps-eval binary for comparison
    #[arg(long)]
    ps_eval: Option<String>,
}

#[derive(Debug, Clone)]
struct ParsedCase {
    hero: String,
    villain: String,
    board: String,
    expected_equity: f64,
}

fn parse_line(line: &str) -> Option<ParsedCase> {
    if line.trim().is_empty() { return None; }
    
    let line = if let Some((_, rest)) = line.split_once(':') {
        rest.trim()
    } else {
        line.trim()
    };

    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 { return None; }

    if parts.len() == 3 {
        let equity = parts[2].parse::<f64>().ok()?;
        Some(ParsedCase {
            hero: parts[0].to_string(),
            villain: parts[1].to_string(),
            board: "".to_string(),
            expected_equity: equity,
        })
    } else {
        let equity = parts[3].parse::<f64>().ok()?;
        Some(ParsedCase {
            hero: parts[0].to_string(),
            villain: parts[1].to_string(),
            board: parts[2].to_string(),
            expected_equity: equity,
        })
    }
}

fn main() {
    let args = Args::parse();

    let backend = match args.backend.to_lowercase().as_str() {
        "cpu" => Backend::Cpu,
        "cuda" => Backend::Cuda,
        "vulkan" => Backend::Vulkan,
        "metal" => Backend::Metal,
        _ => Backend::Auto,
    };

    let eval_mode = match args.mode.to_lowercase().as_str() {
        "exhaustive" => EvalMode::Exhaustive,
        "monte-carlo" | "mc" => EvalMode::MonteCarlo { samples: args.samples, seed: 42 },
        _ => EvalMode::Auto,
    };

    let file = File::open(&args.input).expect("Failed to open input file");
    let reader = BufReader::new(file);

    let mut total = 0;
    let mut passed = 0;
    let mut skipped = 0;

    #[derive(Default)]
    struct Stats {
        count: u64,
        passed: u64,
        total_time: std::time::Duration,
    }

    let mut stats_by_type: HashMap<String, Stats> = HashMap::new();
    let mut ps_eval_stats: Stats = Stats::default();
    let mut ps_comparison_our_time = std::time::Duration::default();
    let mut ps_comparison_count = 0;

    println!("Starting validation bench...");
    println!("Input: {}", args.input);
    println!("Backend: {:?}", backend);
    println!("Mode: {:?}", eval_mode);
    println!("Tolerance: {}", args.tolerance);
    println!("--------------------------------------------------");

    let bench_start = std::time::Instant::now();

    let lines: Vec<(usize, String)> = reader.lines().enumerate()
        .map(|(i, l)| (i, l.expect("Failed to read line")))
        .collect();

    let parsed_cases: Vec<(usize, ParsedCase, plo_eval_gpu::Range, plo_eval_gpu::Range, plo_eval_gpu::Board)> = lines.into_iter().filter_map(|(idx, line)| {
        let case = parse_line(&line)?;
        let hero_range = plo_eval_gpu::Range::from_shorthand(&case.hero, &[]).ok()?;
        let board_cards: Vec<Card> = case.board.as_bytes().chunks(2)
            .filter_map(|c| Card::from_str(std::str::from_utf8(c).unwrap()))
            .collect();
        let board = Board::new(board_cards.clone());
        let villain_range = plo_eval_gpu::Range::from_shorthand(&case.villain, &board_cards).ok()?;
        Some((idx, case, hero_range, villain_range, board))
    }).collect();

    let results: Vec<(usize, ParsedCase, Option<f64>, std::time::Duration, Option<std::time::Duration>)> = if args.backend.to_lowercase() == "cpu" {
        parsed_cases.into_par_iter().map(|(line_idx, case, hero_range, villain_range, board)| {
            if case.expected_equity.is_nan() {
                return (line_idx, case.clone(), None, std::time::Duration::default(), None);
            }
            let start = std::time::Instant::now();
            let res = evaluate_range_vs_range(hero_range, villain_range, board, eval_mode.clone(), false, backend.clone());
            let actual_equity = (res.win + 0.5 * res.tie) * 100.0;
            let duration = start.elapsed();
            (line_idx, case.clone(), Some(actual_equity), duration, None)
        }).collect()
    } else {
        // GPU/Auto Backend: Batching in parallel
        let chunks: Vec<_> = parsed_cases.chunks(256).collect();
        chunks.into_par_iter().flat_map(|chunk| {
            let start = std::time::Instant::now();
            let batch_input: Vec<_> = chunk.iter().map(|(_, _, h, v, b)| (h.clone(), v.clone(), b.clone(), eval_mode.clone())).collect();
            let equities = backend.run_range_evaluation_batch(&batch_input, false);
            let duration = start.elapsed();
            let per_case_duration = duration / chunk.len() as u32;

            let mut chunk_results = Vec::with_capacity(chunk.len());
            for (i, res) in equities.into_iter().enumerate() {
                let (line_idx, case, _, _, _) = &chunk[i];
                if case.expected_equity.is_nan() {
                    chunk_results.push((*line_idx, case.clone(), None, std::time::Duration::default(), None));
                } else {
                    let actual_equity = (res.win + 0.5 * res.tie) * 100.0;
                    chunk_results.push((*line_idx, case.clone(), Some(actual_equity), per_case_duration, None));
                }
            }
            chunk_results
        }).collect()
    };

    for (line_idx, case, actual_equity, duration, ps_duration) in results {
        if actual_equity.is_none() {
            if case.expected_equity.is_nan() {
                println!("Skipping case {}: NaN equity", line_idx + 1);
            } else {
                println!("Skipping case {}: Invalid range(s)", line_idx + 1);
            }
            skipped += 1;
            continue;
        }

        let actual_equity = actual_equity.unwrap();
        let diff = (actual_equity - case.expected_equity).abs();
        let is_ok = diff <= args.tolerance * 100.0;

        if is_ok {
            passed += 1;
        } else {
            println!("FAILED Case {}: Hero={}, Villain={}, Board={}", line_idx + 1, case.hero, case.villain, case.board);
            println!("  Expected: {:.4}, Got: {:.4}, Diff: {:.4}", case.expected_equity, actual_equity, diff);
        }

        let board_cards: Vec<Card> = case.board.as_bytes().chunks(2)
            .filter_map(|c| Card::from_str(std::str::from_utf8(c).unwrap()))
            .collect();
        let board = Board::new(board_cards);

        let query_type = match board.0.len() {
            0 => "Pre-flop",
            3 => "Flop",
            4 => "Turn",
            5 => "River",
            _ => "Other",
        }.to_string();

        let s = stats_by_type.entry(query_type).or_default();
        s.count += 1;
        if is_ok {
            s.passed += 1;
        }
        s.total_time += duration;

        if let Some(ps_dur) = ps_duration {
            ps_eval_stats.count += 1;
            ps_eval_stats.total_time += ps_dur;
            ps_comparison_our_time += duration;
            ps_comparison_count += 1;
        }

        total += 1;
    }

    let total_duration = bench_start.elapsed();
    let mut report = String::new();
    use std::fmt::Write as _;

    writeln!(report, "--- Validation Run: {} ---", chrono::Local::now().to_rfc3339()).ok();
    writeln!(report, "Input: {}, Backend: {:?}, Mode: {:?}", args.input, backend, eval_mode).ok();
    writeln!(report, "Total cases: {}, Passed: {}, Skipped: {}", total, passed, skipped).ok();
    if total > 0 {
        writeln!(report, "Overall Pass rate: {:.2}%", (passed as f64 / total as f64) * 100.0).ok();
    }
    writeln!(report, "Total time: {:?}", total_duration).ok();

    writeln!(report, "\nBreakdown by query type:").ok();
    let mut keys: Vec<_> = stats_by_type.keys().collect();
    keys.sort();
    for kt in keys {
        let s = &stats_by_type[kt];
        let avg = if s.count > 0 { s.total_time / s.count as u32 } else { std::time::Duration::default() };
        let pass_rate = if s.count > 0 { (s.passed as f64 / s.count as f64) * 100.0 } else { 0.0 };
        writeln!(report, "  {:<10}: Count={:<4}, Pass Rate={:>6.2}%, Avg Time={:?}", kt, s.count, pass_rate, avg).ok();
    }

    if ps_eval_stats.count > 0 {
        let avg_ps = ps_eval_stats.total_time / ps_eval_stats.count as u32;
        writeln!(report, "\nComparison with ps-eval ({} cases):", ps_eval_stats.count).ok();
        writeln!(report, "  ps-eval Avg Time: {:?}", avg_ps).ok();
        
        let avg_our_comp = if ps_comparison_count > 0 { ps_comparison_our_time / ps_comparison_count as u32 } else { std::time::Duration::default() };
        writeln!(report, "  Internal Avg Time (comp): {:?}", avg_our_comp).ok();
        
        let total_our_time: std::time::Duration = stats_by_type.values().map(|s| s.total_time).sum();
        let avg_our = if total > 0 { total_our_time / total as u32 } else { std::time::Duration::default() };
        writeln!(report, "  Internal Overall Avg Time: {:?}", avg_our).ok();
    }
    writeln!(report, "--------------------------------------------------\n").ok();

    // Prepend to log file
    let old_content = std::fs::read_to_string(&args.output).unwrap_or_default();
    let mut new_log = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&args.output)
        .expect("Failed to open output log file");
    new_log.write_all(report.as_bytes()).expect("Failed to write report");
    new_log.write_all(old_content.as_bytes()).expect("Failed to write old content");

    println!("--------------------------------------------------");
    println!("Validation finished in {:?}", total_duration);
    println!("Total cases: {}, Passed: {}, Skipped: {}", total, passed, skipped);
    if total > 0 {
        println!("Pass rate: {:.2}%", (passed as f64 / total as f64) * 100.0);
    }
    println!("Average time per case: {:?}", total_duration / (total + skipped).max(1) as u32);
}
