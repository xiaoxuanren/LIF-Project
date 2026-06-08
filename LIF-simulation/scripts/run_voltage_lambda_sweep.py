import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lif_inference.voltage_augmented_learned_lif_connectivity import (
    VoltageAugmentedPerNeuronLIF,
    build_ground_truth,
    build_train_val_voltage_datasets,
    compute_neighbor_indices,
    evaluate_connectivity,
    evaluate_event_windows,
    load_all_recordings_with_voltage,
    resolve_session_dt,
    split_recording_boundaries,
    train_epoch_events,
)
from lif_inference.voltage_training_orchestration import (
    train_voltage_model_with_early_stopping,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_lambdas(raw_values):
    """Parse a comma-separated lambda list into floating-point values.

    Args:
        raw_values: Comma-separated string such as ``"0,0.25,1"``.

    Returns:
        A list of floating-point lambda values in the order supplied by the user.
    """
    return [float(value.strip()) for value in raw_values.split(',') if value.strip()]


def lambda_tag(value):
    """Convert one lambda value into a filesystem-friendly tag.

    Args:
        value: Numeric lambda value.

    Returns:
        A short string safe to embed in filenames.
    """
    text = f'{value:g}'
    return text.replace('-', 'm').replace('.', 'p')


def output_stem(session_name, k, epochs, batch_size, candidate_mode, spatial_frac, max_delay):
    """Build the shared filename stem used for sweep artifacts.

    Args:
        session_name: Session name used as the base label.
        k: Number of candidate neighbors per neuron.
        epochs: Number of training epochs per sweep run.
        batch_size: Mini-batch size used during training.
        candidate_mode: Candidate-edge generation mode.
        spatial_frac: Fraction of candidates drawn from spatial neighbors.
        max_delay: Maximum delay bin used by the learned-LIF model.

    Returns:
        A compact filename stem shared by the sweep outputs.
    """
    spatial_pct = int(round(float(spatial_frac) * 100.0))
    return f'{session_name}_k{k}_e{epochs}_b{batch_size}_{candidate_mode}_sf{spatial_pct}_lag{max_delay}'


def prepare_shared_context(args):
    """Load shared data, candidates, and datasets once for the full lambda sweep.

    Args:
        args: Parsed CLI namespace describing the session, candidate settings, and
            event-window configuration used for every lambda value in the sweep.

    Returns:
        A dictionary containing the shared loaded data, candidate sets, ground truth,
        validation metadata, and train/validation dataloaders reused across runs.
    """
    print(f"\n{'=' * 70}")
    print('VOLTAGE-LAMBDA SWEEP FOR VOLTAGE-AUGMENTED LEARNED LIF')
    print(f'Session: {os.path.basename(args.session)}')
    print(f'Lambdas: {args.lambdas}')
    print(f'K={args.k}, epochs={args.epochs}, batch={args.batch}, lr={args.lr}, max_delay={args.max_delay}')
    print(f'Dt: {args.dt:g} ms ({args.dt_source})')
    print(f'Device: {args.device}')
    print(f"{'=' * 70}")

    print('\nLoading shared data...')
    # Load candidates and event windows once so every lambda sees the same data slice.
    data = load_all_recordings_with_voltage(
        args.session,
        dt=args.dt,
        mask_pre_ms=args.mask_pre_ms,
        mask_post_ms=args.mask_post_ms,
        peak_threshold_mv=args.peak_threshold_mv,
    )
    print(f'  Loaded {data["n_recordings"]} recordings, total duration: {data["total_duration"] / 1000:.0f}s')

    spike_matrix = data['spike_matrix']
    voltage_matrix = data['voltage_matrix']
    voltage_mask = data['voltage_mask']
    boundaries = data['boundaries']
    n_neurons = data['n_neurons']
    connections = data['connections']
    positions = data['neuron_positions']

    print(f'  Neurons: {n_neurons}, Connections: {len(connections)}')
    print(f'  Spike matrix: {list(spike_matrix.shape)} ({int(spike_matrix.sum())} total spikes)')
    print(f'  Voltage matrix: {list(voltage_matrix.shape)} (valid fraction={float(voltage_mask.mean()):.3f})')

    candidate_train_boundaries, _ = split_recording_boundaries(boundaries, args.val_fraction)
    candidate_max_lag = args.max_delay if args.candidate_max_lag is None else args.candidate_max_lag
    neighbor_indices, k_actual, candidate_info = compute_neighbor_indices(
        positions,
        args.k,
        spike_matrix=spike_matrix,
        mode=args.candidate_mode,
        boundaries=candidate_train_boundaries,
        spatial_frac=args.candidate_spatial_frac,
        temporal_min_lag=args.candidate_min_lag,
        temporal_max_lag=candidate_max_lag,
    )
    true_weights, true_binary = build_ground_truth(connections, n_neurons)

    total_in_k = sum(true_binary[j, neighbor_indices[j]].sum() for j in range(n_neurons))
    total_true = int(true_binary.sum())
    print(f'  Candidate mode: {candidate_info["mode"]}')
    if candidate_info['mode'] == 'hybrid':
        print(
            '  Candidate mix: '
            f'{candidate_info["n_spatial"]} spatial + {candidate_info["n_temporal"]} temporal '
            f'(lag {candidate_info["temporal_min_lag"]}-{candidate_info["temporal_max_lag"]} bins)'
        )
        print(f'  Mean temporal-only candidates per neuron: {candidate_info["mean_temporal_only"]:.1f}')
    print(f'  K={k_actual}, coverage: {total_in_k}/{total_true} ({total_in_k / max(total_true, 1):.1%})')

    print('\nBuilding shared event-window datasets...')
    train_ds, val_ds, validation_strategy = build_train_val_voltage_datasets(
        spike_matrix,
        voltage_matrix,
        voltage_mask,
        neighbor_indices,
        np.arange(n_neurons),
        pre_context=args.pre_context,
        post_context=args.post_context,
        warmup=args.warmup,
        neg_ratio=args.neg_ratio,
        neg_min_distance=args.neg_min_dist,
        boundaries=boundaries,
        val_fraction=args.val_fraction,
        rng_seed=42,
    )
    print(f'  Validation strategy: {validation_strategy}')
    print(f'  Train windows: {len(train_ds)} ({train_ds.n_pos} pos, {train_ds.n_neg} neg)')
    print(f'  Val windows:   {len(val_ds)} ({val_ds.n_pos} pos, {val_ds.n_neg} neg)')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    return {
        'data': data,
        'n_neurons': n_neurons,
        'connections': connections,
        'positions': positions,
        'neighbor_indices': neighbor_indices,
        'k_actual': k_actual,
        'candidate_info': candidate_info,
        'true_weights': true_weights,
        'true_binary': true_binary,
        'validation_strategy': validation_strategy,
        'train_loader': train_loader,
        'val_loader': val_loader,
    }


def train_single_lambda(args, shared, voltage_lambda, output_dir, base_stem):
    """Train and export one voltage-augmented model for a single lambda value.

    Args:
        args: Parsed CLI namespace controlling training hyperparameters and outputs.
        shared: Dictionary returned by `prepare_shared_context` with fixed data and loaders.
        voltage_lambda: Voltage-loss weight to evaluate in this run.
        output_dir: Directory where checkpoints and connectivity outputs are written.
        base_stem: Shared artifact stem used to keep run names aligned across lambdas.

    Returns:
        A dictionary summarizing the run configuration, saved artifact paths, elapsed
        time, best epoch, and held-out validation/connectivity metrics.
    """
    session_name = os.path.basename(args.session)
    run_tag = f'{base_stem}_vl{lambda_tag(voltage_lambda)}'
    output_name = f'{session_name}_{run_tag}'
    device = args.device

    print(f"\n{'-' * 70}")
    print(f'Starting voltage_lambda={voltage_lambda:g} ({output_name})')
    print(f"{'-' * 70}")

    # Only the voltage-loss weight changes across runs; the shared context stays fixed.
    model = VoltageAugmentedPerNeuronLIF(
        n_neurons=shared['n_neurons'],
        K=shared['k_actual'],
        max_delay=args.max_delay,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=max(2, args.epochs // 3),
    )

    def log_progress(epoch, n_epochs, train_stats, val_stats, conn_results, model,
                     elapsed_seconds):
        alpha_val = torch.sigmoid(model.alpha_logit).item()
        print(
            f'  Epoch {epoch:2d}/{n_epochs}: '
            f'train={train_stats["loss"]:.4f} '
            f'val={val_stats["loss"]:.4f} '
            f'(spike={val_stats["spike_loss"]:.4f} voltage={val_stats["voltage_loss"]:.4f}) '
            f'conn_AUC={conn_results["auc"]:.4f} alpha={alpha_val:.3f} '
            f'thresh={model.threshold.item():.3f} ({elapsed_seconds:.0f}s)'
        )

    training_results = train_voltage_model_with_early_stopping(
        model,
        shared['train_loader'],
        shared['val_loader'],
        optimizer,
        scheduler,
        train_epoch_fn=train_epoch_events,
        evaluate_windows_fn=evaluate_event_windows,
        evaluate_connectivity_fn=evaluate_connectivity,
        neighbor_indices=shared['neighbor_indices'],
        true_binary=shared['true_binary'],
        true_weights=shared['true_weights'],
        device=device,
        warmup=args.warmup,
        pos_weight=args.pos_weight,
        l1_lambda=args.l1,
        voltage_lambda=voltage_lambda,
        n_epochs=args.epochs,
        patience=None,
        log_fn=None,
        progress_fn=log_progress,
    )
    best_epoch = training_results['best_epoch']
    train_history = training_results['train_history']
    val_history = training_results['val_history']
    conn_aucs = training_results['conn_aucs']
    val_window_results = training_results['val_window_results']
    all_results, _, _, _, _, conn_matrix = evaluate_connectivity(
        model,
        shared['neighbor_indices'],
        shared['true_binary'],
        shared['true_weights'],
    )

    elapsed = training_results['elapsed_seconds']
    print(
        f'Completed voltage_lambda={voltage_lambda:g}: '
        f'AUC={all_results["auc"]:.4f}, AP={all_results["ap"]:.4f}, '
        f'F1={all_results["f1"]:.4f}, val_loss={val_window_results["loss"]:.4f} '
        f'in {elapsed / 60.0:.1f} min'
    )

    checkpoint_path = os.path.join(output_dir, f'voltage_augmented_learned_lif_{output_name}.pt')
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'K': shared['k_actual'],
            'T': int(shared['data']['spike_matrix'].shape[1]),
            'dt': float(args.dt),
            'max_delay': args.max_delay,
            'n_neurons': shared['n_neurons'],
            'session_name': session_name,
            'output_name': output_name,
            'validation_strategy': shared['validation_strategy'],
            'candidate_info': shared['candidate_info'],
            'threshold_mode': model.threshold_mode,
            'neighbor_indices': shared['neighbor_indices'],
            'connectivity_matrix': conn_matrix,
            'results_window_val': val_window_results,
            'results_all': all_results,
            'train_history': train_history,
            'val_history': val_history,
            'connectivity_aucs': conn_aucs,
            'recording_summaries': shared['data']['recording_summaries'],
            'window_config': {
                'pre_context': int(args.pre_context),
                'post_context': int(args.post_context),
                'warmup': int(args.warmup),
                'neg_ratio': float(args.neg_ratio),
                'neg_min_distance': int(args.neg_min_dist),
                'val_fraction': float(args.val_fraction),
                'rng_seed': 42,
            },
            'data_config': {
                'use_all_recordings': True,
                'recording_idx': 0,
                'subsample_T': None,
            },
            'voltage_cleaning': {
                'mask_pre_ms': args.mask_pre_ms,
                'mask_post_ms': args.mask_post_ms,
                'peak_threshold_mv': args.peak_threshold_mv,
                'voltage_lambda': voltage_lambda,
            },
        },
        checkpoint_path,
    )

    connectivity_path = os.path.join(output_dir, f'connectivity_{output_name}.npz')
    np.savez_compressed(
        connectivity_path,
        connectivity_matrix=conn_matrix,
        threshold=all_results.get('threshold', 0.5),
        neighbor_indices=shared['neighbor_indices'],
        neuron_positions=shared['positions'],
        true_weights=shared['true_weights'],
    )

    return {
        'lambda': float(voltage_lambda),
        'output_name': output_name,
        'checkpoint_path': checkpoint_path,
        'connectivity_path': connectivity_path,
        'elapsed_sec': elapsed,
        'best_epoch': int(best_epoch),
        'epochs': int(args.epochs),
        'results_all': all_results,
        'results_window_val': val_window_results,
    }


def save_summary_files(records, output_dir, prefix):
    """Write machine-readable JSON and CSV summaries for the completed sweep.

    Args:
        records: Per-lambda run summaries produced by the sweep.
        output_dir: Directory where the JSON and CSV files are written.
        prefix: Shared filename prefix used for both summary files.

    Returns:
        A tuple containing the JSON summary path and CSV summary path.
    """
    json_path = os.path.join(output_dir, f'{prefix}.json')
    csv_path = os.path.join(output_dir, f'{prefix}.csv')

    # Export machine-readable summaries so later notebook analysis does not need to parse logs.
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(records, handle, indent=2)

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'lambda', 'output_name', 'best_epoch', 'epochs', 'elapsed_sec',
            'val_loss', 'val_spike_loss', 'val_voltage_loss', 'val_l1_loss',
            'auc', 'ap', 'f1', 'precision', 'recall', 'weight_corr',
            'sign_accuracy', 'threshold',
        ])
        for record in records:
            val_metrics = record['results_window_val']
            conn_metrics = record['results_all']
            writer.writerow([
                record['lambda'],
                record['output_name'],
                record['best_epoch'],
                record['epochs'],
                f'{record["elapsed_sec"]:.3f}',
                f'{val_metrics["loss"]:.6f}',
                f'{val_metrics["spike_loss"]:.6f}',
                f'{val_metrics["voltage_loss"]:.6f}',
                f'{val_metrics["l1_loss"]:.6f}',
                f'{conn_metrics["auc"]:.6f}',
                f'{conn_metrics["ap"]:.6f}',
                f'{conn_metrics["f1"]:.6f}',
                f'{conn_metrics["precision"]:.6f}',
                f'{conn_metrics["recall"]:.6f}',
                f'{conn_metrics["weight_corr"]:.6f}',
                f'{conn_metrics["sign_accuracy"]:.6f}',
                f'{conn_metrics["threshold"]:.6f}',
            ])

    return json_path, csv_path


def save_sweep_plot(records, output_dir, prefix):
    """Render the aggregate metric curves across all swept lambda values.

    Args:
        records: Per-lambda run summaries produced by the sweep.
        output_dir: Directory where the plot is written.
        prefix: Filename prefix used for the saved PNG.

    Returns:
        The path to the saved sweep plot PNG.
    """
    lambdas = np.array([record['lambda'] for record in records], dtype=np.float32)
    order = np.argsort(lambdas)
    # Sort numerically before plotting because the CLI accepts arbitrary comma-separated order.
    ordered = [records[index] for index in order]
    lambdas = lambdas[order]

    val_loss = np.array([record['results_window_val']['loss'] for record in ordered], dtype=np.float32)
    spike_loss = np.array([record['results_window_val']['spike_loss'] for record in ordered], dtype=np.float32)
    voltage_loss = np.array([record['results_window_val']['voltage_loss'] for record in ordered], dtype=np.float32)
    auc = np.array([record['results_all']['auc'] for record in ordered], dtype=np.float32)
    ap = np.array([record['results_all']['ap'] for record in ordered], dtype=np.float32)
    f1 = np.array([record['results_all']['f1'] for record in ordered], dtype=np.float32)
    weight_corr = np.array([record['results_all']['weight_corr'] for record in ordered], dtype=np.float32)
    sign_accuracy = np.array([record['results_all']['sign_accuracy'] for record in ordered], dtype=np.float32)
    threshold = np.array([record['results_all']['threshold'] for record in ordered], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle('Voltage-Lambda Sweep on LIF Data', fontsize=15, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(lambdas, val_loss, marker='o', linewidth=2, label='Val total')
    ax.plot(lambdas, spike_loss, marker='s', linewidth=2, label='Val spike')
    ax.plot(lambdas, voltage_loss, marker='^', linewidth=2, label='Val voltage')
    ax.set_xlabel('Voltage loss weight (lambda)')
    ax.set_ylabel('Loss')
    ax.set_title('Held-Out Validation Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(lambdas, auc, marker='o', linewidth=2, label='AUC')
    ax.plot(lambdas, ap, marker='s', linewidth=2, label='AP')
    ax.plot(lambdas, f1, marker='^', linewidth=2, label='F1')
    ax.set_xlabel('Voltage loss weight (lambda)')
    ax.set_ylabel('Connectivity metric')
    ax.set_title('Connectivity Recovery')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(lambdas, weight_corr, marker='o', linewidth=2, label='Weight corr')
    ax.plot(lambdas, sign_accuracy, marker='s', linewidth=2, label='Sign accuracy')
    ax.set_xlabel('Voltage loss weight (lambda)')
    ax.set_ylabel('Metric')
    ax.set_title('Recovered Weight Quality')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(lambdas, threshold, marker='o', linewidth=2, color='tab:purple', label='Threshold')
    best_auc_idx = int(np.argmax(auc))
    best_lambda = float(lambdas[best_auc_idx])
    ax.axvline(best_lambda, color='tab:red', linestyle='--', linewidth=1.5,
               label=f'Best AUC lambda={best_lambda:g}')
    ax.set_xlabel('Voltage loss weight (lambda)')
    ax.set_ylabel('Decision threshold')
    ax.set_title('Selected Connectivity Threshold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = os.path.join(output_dir, f'{prefix}.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main():
    """Parse CLI arguments, run the lambda sweep, and save aggregate artifacts.

    Args:
        None.

    Returns:
        None. The function trains one model per lambda value and writes the sweep
        summaries, checkpoints, and plot outputs to disk.
    """
    parser = argparse.ArgumentParser(description='Shared-data sweep over voltage_lambda values')
    parser.add_argument('--session', type=str, required=True)
    parser.add_argument('--lambdas', type=str, default='0,0.25,0.5,1,4')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-delay', type=int, default=8)
    parser.add_argument('--l1', type=float, default=0.01)
    parser.add_argument('--pos-weight', type=float, default=5.0)
    parser.add_argument('--dt', type=float, default=None,
                        help='Optional spike/voltage bin width override in ms. Defaults to the session metadata value.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--candidate-mode', type=str, default='hybrid', choices=['spatial', 'hybrid'])
    parser.add_argument('--candidate-spatial-frac', type=float, default=0.8)
    parser.add_argument('--candidate-min-lag', type=int, default=1)
    parser.add_argument('--candidate-max-lag', type=int, default=None)
    parser.add_argument('--pre-context', type=int, default=50)
    parser.add_argument('--post-context', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=100,
                        help='Leading bins simulated per window but excluded from the loss, giving slow membrane/adaptation state time to settle before the scored region')
    parser.add_argument('--neg-ratio', type=float, default=1.0)
    parser.add_argument('--neg-min-dist', type=int, default=100)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--mask-pre-ms', type=float, default=0.0,
                        help='Voltage masked before each spike; default 0.0 keeps the pre-spike depolarization ramp as a supervised timing target')
    parser.add_argument('--mask-post-ms', type=float, default=2.0)
    parser.add_argument('--peak-threshold-mv', type=float, default=15.0)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    args.dt, args.dt_source = resolve_session_dt(args.session, args.dt)

    output_dir = REPO_ROOT / 'voltage_augmented_learned_lif_outputs'
    os.makedirs(output_dir, exist_ok=True)

    lambda_values = parse_lambdas(args.lambdas)
    args.lambdas = ', '.join(f'{value:g}' for value in lambda_values)
    base_stem = output_stem(
        os.path.basename(args.session),
        args.k,
        args.epochs,
        args.batch,
        args.candidate_mode,
        args.candidate_spatial_frac,
        args.max_delay,
    )
    summary_prefix = f'voltage_lambda_sweep_{base_stem}'

    shared = prepare_shared_context(args)

    records = []
    sweep_start = time.time()
    # Reuse the shared candidate set and train/val split across all lambda values for a fair sweep.
    for value in lambda_values:
        records.append(train_single_lambda(args, shared, value, output_dir, base_stem))

    total_minutes = (time.time() - sweep_start) / 60.0
    json_path, csv_path = save_summary_files(records, output_dir, summary_prefix)
    plot_path = save_sweep_plot(records, output_dir, summary_prefix)

    print(f"\n{'=' * 70}")
    print('SWEEP COMPLETE')
    print(f'Total sweep time: {total_minutes:.1f} min')
    print(f'Summary JSON: {json_path}')
    print(f'Summary CSV:  {csv_path}')
    print(f'Plot PNG:     {plot_path}')
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()