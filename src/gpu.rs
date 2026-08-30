use crate::{Hand, Board, EvalMode, EquityResult};
use crate::eval_fast::card_to_index;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::borrow::Cow;
use once_cell::sync::Lazy;
use pollster::block_on;

use std::sync::Mutex;

static GPU_CONTEXT: Lazy<Option<GpuContext>> = Lazy::new(|| block_on(init_gpu()));
static GPU_LOCK: Mutex<()> = Mutex::new(());

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    results_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
}

async fn init_gpu() -> Option<GpuContext> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Omaha Evaluator"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("omaha.wgsl"))),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
    });

    let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Results Buffer"),
        size: 256 * 4 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: 256 * 4 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Global Input Buffer"),
        size: std::mem::size_of::<GpuInput>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Some(GpuContext { device, queue, pipeline, results_buffer, staging_buffer, input_buffer })
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuCaseInput {
    hero_hands: [[u32; 4]; 128],
    villain_hands: [[u32; 4]; 128],
    hero_weights: [f32; 128],
    villain_weights: [f32; 128],
    hero_count: u32,
    villain_count: u32,
    board: [u32; 5],
    board_len: u32,
    mode: u32,
    samples: u32,
    seed: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuInput {
    cases: [GpuCaseInput; 256],
}

pub fn run_gpu_range_evaluation_batch(
    cases: &[(crate::Range, crate::Range, Board, EvalMode)],
) -> Vec<Option<EquityResult>> {
    let ctx = match GPU_CONTEXT.as_ref() {
        Some(c) => c,
        None => return vec![None; cases.len()],
    };

    let mut gpu_input_struct = GpuInput::zeroed();

    let mut trial_counts = Vec::with_capacity(256);
    let mut modes = Vec::with_capacity(256);

    for (idx, (hero_range, villain_range, board, mode)) in cases.iter().take(256).enumerate() {
        let board_len = board.0.len();
        
        let hero_count = hero_range.hands.len().min(128);
        let villain_count = villain_range.hands.len().min(128);
        trial_counts.push((hero_count * villain_count) as u64);
        modes.push(mode.clone());

        let effective_mode = match mode {
            EvalMode::Auto => {
                if board_len >= 3 {
                    EvalMode::Exhaustive
                } else {
                    EvalMode::MonteCarlo { samples: 10000, seed: 42 }
                }
            }
            _ => mode.clone(),
        };

        let case_input = &mut gpu_input_struct.cases[idx];
        case_input.hero_count = hero_count as u32;
        case_input.villain_count = villain_count as u32;
        case_input.board = {
            let mut b = [255u32; 5];
            for i in 0..board.0.len().min(5) {
                b[i] = card_to_index(board.0[i]) as u32;
            }
            b
        };
        case_input.board_len = board_len as u32;
        case_input.mode = match effective_mode {
            EvalMode::MonteCarlo { .. } => 1,
            _ => 0,
        };
        case_input.samples = match effective_mode {
            EvalMode::MonteCarlo { samples, .. } => samples as u32,
            _ => 0,
        };
        case_input.seed = match effective_mode {
            EvalMode::MonteCarlo { seed, .. } => seed as u32,
            _ => 0,
        };

        for i in 0..hero_count {
            let (hand, weight) = &hero_range.hands[i];
            for j in 0..4 {
                case_input.hero_hands[i][j] = card_to_index(hand.0[j]) as u32;
            }
            case_input.hero_weights[i] = *weight as f32;
        }
        for i in 0..villain_count {
            let (hand, weight) = &villain_range.hands[i];
            for j in 0..4 {
                case_input.villain_hands[i][j] = card_to_index(hand.0[j]) as u32;
            }
            case_input.villain_weights[i] = *weight as f32;
        }
    }

    let _lock = GPU_LOCK.lock().unwrap();

    ctx.device.poll(wgpu::Maintain::Wait);
    ctx.queue.write_buffer(&ctx.input_buffer, 0, bytemuck::bytes_of(&gpu_input_struct));
    ctx.queue.write_buffer(&ctx.results_buffer, 0, bytemuck::cast_slice(&[0u32; 1024]));

    let bind_group_layout = ctx.pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ctx.input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: ctx.results_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        
        let mut max_pairs = 1u32;
        for i in 0..cases.len().min(256) {
            let case = &gpu_input_struct.cases[i];
            let pairs = (case.hero_count * case.villain_count) as u32;
            max_pairs = max_pairs.max(pairs);
        }
        let workgroups_x = (max_pairs + 63) / 64;
        let workgroups_y = cases.len().min(256) as u32;
        cpass.dispatch_workgroups(workgroups_x.max(1), workgroups_y, 1);
    }

    encoder.copy_buffer_to_buffer(&ctx.results_buffer, 0, &ctx.staging_buffer, 0, 1024 * 4);
    ctx.queue.submit(Some(encoder.finish()));

    let (sender, receiver) = std::sync::mpsc::channel();
    let buffer_slice = ctx.staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);

    let mut results_out = vec![None; cases.len()];
    if let Ok(Ok(())) = receiver.recv() {
        let data = buffer_slice.get_mapped_range();
        let mut results = vec![0u32; 1024];
        results.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        ctx.staging_buffer.unmap();

        for i in 0..cases.len().min(256) {
            let offset = i * 4;
            let win = results[offset + 0];
            let tie = results[offset + 1];
            let loss = results[offset + 2];
            let total_weight = results[offset + 3];
            
            let total = total_weight as f64;
            if total == 0.0 {
                results_out[i] = Some(EquityResult {
                    win: 0.0,
                    tie: 0.0,
                    loss: 0.0,
                    win_low: None,
                    tie_low: None,
                    loss_low: None,
                    trial_count: 0,
                    mode: modes[i].clone(),
                    confidence_interval: None,
                });
            } else {
                results_out[i] = Some(EquityResult {
                    win: win as f64 / total,
                    tie: tie as f64 / total,
                    loss: loss as f64 / total,
                    win_low: None,
                    tie_low: None,
                    loss_low: None,
                    trial_count: trial_counts[i],
                    mode: modes[i].clone(),
                    confidence_interval: None,
                });
            }
        }
    }
    results_out
}

pub fn run_gpu_range_evaluation(
    hero_range: &crate::Range,
    villain_range: &crate::Range,
    board: &Board,
    mode: &EvalMode,
) -> Option<EquityResult> {
    let cases = vec![(hero_range.clone(), villain_range.clone(), board.clone(), mode.clone())];
    let mut results = run_gpu_range_evaluation_batch(&cases);
    results.pop().unwrap()
}

pub fn run_gpu_evaluation(
    hero: &Hand,
    villain: &Hand,
    board: &Board,
    mode: &EvalMode,
    hi_lo: bool,
) -> Option<EquityResult> {
    if hi_lo {
        // Hi/Lo not yet implemented in GPU
        return None;
    }
    
    let hero_range = crate::Range { hands: vec![(hero.clone(), 1.0)] };
    let villain_range = crate::Range { hands: vec![(villain.clone(), 1.0)] };
    run_gpu_range_evaluation(&hero_range, &villain_range, board, mode)
}
