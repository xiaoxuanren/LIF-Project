r"""Tune the interval-jitter surrogate window (``jitter_bins``) on a saved sim session.

For each candidate ``jitter_bins`` it fits interval-jitter null models with the
voltage-augmented surrogate machinery and reports, against the real trained model:

  * power_edge_collapse = median(null score @ true-edge candidates)
                          / median(observed score @ true-edge candidates)
                          -- want SMALL (true edges must fully collapse under the null;
                          a window that under-destroys edges inflates the null at true
                          edges and silently raises the cutoff / kills recall).
  * calibration_ratio   = (null non-edge p95) / (observed non-edge p95)
                          -- want ~1.0 (the null FP floor matches the real one).
  * surrogate_cutoff / true_precision / true_recall / true_FDR / FP: the actual global
    surrogate_fdr selection at ``--target-fdr`` scored against ground truth.

Selection rule (sim-only, uses ground truth): among windows whose power_edge_collapse
stays small (full edge-collapse), pick the one whose true_FDR is closest to target_fdr
(equivalently calibration_ratio closest to 1) -- the smallest window with full
edge-collapse, NOT simply the highest null floor. In production (real data, no ground
truth) you apply the chosen window + target_fdr directly without the oracle/true columns.

NOTE on target_fdr: it must be a point on this model's achievable frontier. The voltage
model's scores overlap, so sub-1% FDR is unreachable at usable recall; pick target_fdr
from the oracle frontier (printed below), not the spike-only model's 0.5%.

``--reuse-observed <connectivity_npz>`` reuses a saved model's scores/candidates/
excluded-bins instead of retraining (fits only the nulls) -- exact + fast for re-tuning
an already-fitted model.

Example (PowerShell):
  ..\.venv\Scripts\python.exe scripts\run_jitter_sweep.py `
      --session "D:\...\20260523_084407" --jitter-bins-list 10,12,14,16,20,25 `
      --target-fdr 0.11 --reuse-observed voltage_augmented_learned_lif_outputs\connectivity_20260523_084407_voltage_all20_K100_jitter25.npz
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from lif_inference.connectivity_metrics import (
    compute_binary_classification_metrics,
    flatten_candidate_scores,
    select_connectivity_threshold,
)
from lif_inference.shared_data import validate_jitter_null
from lif_inference.voltage_augmented_learned_lif_connectivity import (
    VoltageAugmentedPerNeuronLIF,
    build_ground_truth,
    build_train_val_voltage_datasets,
    compute_neighbor_indices,
    estimate_surrogate_connectivity_score_sets,
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


def parse_windows(raw_values):
    """Parse a comma-separated jitter-window list into ints."""
    return [int(value.strip()) for value in raw_values.split(',') if value.strip()]


def flatten_labels(true_binary, neighbor_indices, n_neurons):
    """Flatten ground-truth labels in the canonical evaluate_connectivity order."""
    return np.concatenate([
        true_binary[i, neighbor_indices[i]].astype(np.int32)
        for i in range(n_neurons)
    ])


def print_frontier(obs_scores, labels, signed_scores, true_sign, targets=(0.005, 0.05, 0.11, 0.20, 0.30)):
    """Print the oracle precision/recall/true-FDR frontier and ranking metrics."""
    auc = roc_auc_score(labels, obs_scores)
    ap = average_precision_score(labels, obs_scores)
    inh_pos = (true_sign < 0).astype(int)
    inh_auprc = (average_precision_score(inh_pos, -signed_scores)
                 if inh_pos.sum() > 0 else float('nan'))
    print(f"\n  Ranking (window-invariant): AUC={auc:.4f} AP={ap:.4f} "
          f"inhibitory-AUPRC={inh_auprc:.4f}")
    order = np.argsort(-obs_scores)
    ls = labels[order]
    ss = obs_scores[order]
    precision = np.cumsum(ls) / np.arange(1, len(ls) + 1)
    recall = np.cumsum(ls) / labels.sum()
    fdr = 1.0 - precision
    print(f"  ORACLE FRONTIER  {'target':>8}{'reach':>7}{'cutoff':>9}{'prec':>8}{'recall':>8}{'trueFDR':>9}")
    for target in targets:
        feasible = np.where(fdr <= target)[0]
        if feasible.size == 0:
            print(f"  {'':>17}{target:>8.3f}{'NO':>7}{'--':>9}{'--':>8}{'--':>8}{'--':>9}")
        else:
            k = int(feasible[-1])
            print(f"  {'':>17}{target:>8.3f}{'yes':>7}{ss[k]:>9.4f}{precision[k]:>8.4f}"
                  f"{recall[k]:>8.4f}{fdr[k]:>9.4f}")
    return {'auc': float(auc), 'ap': float(ap), 'inh_auprc': float(inh_auprc)}


def true_metrics_at_cutoff(obs_scores, labels, cutoff):
    """Confusion metrics scoring observed candidates against ground truth at a cutoff."""
    if not np.isfinite(cutoff):
        return {'precision': 0.0, 'recall': 0.0, 'tp': 0, 'fp': 0, 'fn': int(labels.sum()), 'fdr': 0.0}
    predicted = (obs_scores >= cutoff).astype(np.int32)
    m = compute_binary_classification_metrics(labels, predicted)
    tp, fp = int(m['tp']), int(m['fp'])
    m['fdr'] = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
    return m


def evaluate_null(null_sets, obs_scores, labels, score_neuron_ids, target_fdr):
    """Run global surrogate_fdr selection at target_fdr; return cutoff + true metrics."""
    info = select_connectivity_threshold(
        labels, obs_scores, mode='surrogate_fdr',
        surrogate_score_sets=null_sets, surrogate_fdr=target_fdr,
        score_neuron_ids=score_neuron_ids)
    cutoff = float(info['threshold'])
    tm = true_metrics_at_cutoff(obs_scores, labels, cutoff)
    return cutoff, tm


def main():
    """Run the interval-jitter window sweep and print frontier + power/calibration/true-FDR."""
    parser = argparse.ArgumentParser(description='Sweep interval-jitter jitter_bins on a sim session')
    parser.add_argument('--session', type=str, required=True)
    parser.add_argument('--jitter-bins-list', type=str, default='10,12,14,16,20,25')
    parser.add_argument('--target-fdr', type=float, default=0.11,
                        help='Surrogate-FDR target; must be reachable on this model frontier')
    parser.add_argument('--reuse-observed', type=str, default=None,
                        help='Saved connectivity npz to reuse (scores/candidates/excluded-bins); skips training')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-delay', type=int, default=8)
    parser.add_argument('--l1', type=float, default=0.01)
    parser.add_argument('--pos-weight', type=float, default=5.0)
    parser.add_argument('--voltage-lambda', type=float, default=1.0)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--candidate-mode', type=str, default='hybrid', choices=['spatial', 'hybrid'])
    parser.add_argument('--candidate-spatial-frac', type=float, default=0.8)
    parser.add_argument('--candidate-min-lag', type=int, default=1)
    parser.add_argument('--candidate-max-lag', type=int, default=None)
    parser.add_argument('--pre-context', type=int, default=50)
    parser.add_argument('--post-context', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--neg-ratio', type=float, default=1.0)
    parser.add_argument('--neg-min-dist', type=int, default=100)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--mask-pre-ms', type=float, default=0.0)
    parser.add_argument('--mask-post-ms', type=float, default=2.0)
    parser.add_argument('--peak-threshold-mv', type=float, default=15.0)
    parser.add_argument('--n-surrogates', type=int, default=4)
    parser.add_argument('--surrogate-epochs', type=int, default=1)
    parser.add_argument('--surrogate-patience', type=int, default=1)
    parser.add_argument('--surrogate-seed', type=int, default=1234)
    parser.add_argument('--nonedge-percentile', type=float, default=95.0)
    parser.add_argument('--power-max', type=float, default=0.5)
    parser.add_argument('--skip-circular', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    args.dt, dt_source = resolve_session_dt(args.session, args.dt)

    print(f"\n{'=' * 78}")
    print('INTERVAL-JITTER WINDOW SWEEP (voltage-augmented learned LIF)')
    print(f'Session: {os.path.basename(args.session)} | target_fdr={args.target_fdr:g}')
    print(f'Windows: {args.jitter_bins_list} | max_delay={args.max_delay} | dt={args.dt:g} ms ({dt_source})')
    print(f"{'=' * 78}\n  Loading data...")

    data = load_all_recordings_with_voltage(
        args.session, dt=args.dt, mask_pre_ms=args.mask_pre_ms,
        mask_post_ms=args.mask_post_ms, peak_threshold_mv=args.peak_threshold_mv)
    spike_matrix = data['spike_matrix']
    voltage_matrix = data['voltage_matrix']
    voltage_mask = data['voltage_mask']
    boundaries = data['boundaries']
    n_neurons = data['n_neurons']
    connections = data['connections']
    positions = data['neuron_positions']
    true_weights, true_binary = build_ground_truth(connections, n_neurons)
    print(f'  Neurons={n_neurons} recordings={data["n_recordings"]} '
          f'duration={data["total_duration"] / 1000:.0f}s spikes={int(spike_matrix.sum())}')

    if args.reuse_observed:
        z = np.load(args.reuse_observed, allow_pickle=True)
        conn_matrix = z['connectivity_matrix']
        neighbor_indices = [np.asarray(a, int) for a in z['neighbor_indices']]
        excluded_bins = z['excluded_bins'] if 'excluded_bins' in z.files else None
        k_actual = int(max(len(r) for r in neighbor_indices))
        threshold_mode = 'adaptive'
        print(f'  Reusing observed model: {os.path.basename(args.reuse_observed)} '
              f'(K={k_actual}, excluded_bins={0 if excluded_bins is None else len(excluded_bins)})')
    else:
        cand_train_boundaries, _ = split_recording_boundaries(boundaries, args.val_fraction)
        cand_max_lag = args.max_delay if args.candidate_max_lag is None else args.candidate_max_lag
        neighbor_indices, k_actual, _ = compute_neighbor_indices(
            positions, args.k, spike_matrix=spike_matrix, mode=args.candidate_mode,
            boundaries=cand_train_boundaries, spatial_frac=args.candidate_spatial_frac,
            temporal_min_lag=args.candidate_min_lag, temporal_max_lag=cand_max_lag)
        excluded_bins = None
        train_ds, val_ds, _ = build_train_val_voltage_datasets(
            spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
            np.arange(n_neurons), pre_context=args.pre_context, post_context=args.post_context,
            warmup=args.warmup, neg_ratio=args.neg_ratio, neg_min_distance=args.neg_min_dist,
            boundaries=boundaries, val_fraction=args.val_fraction, rng_seed=42)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        print(f'  K={k_actual} | training real model...')
        model = VoltageAugmentedPerNeuronLIF(
            n_neurons=n_neurons, K=k_actual, max_delay=args.max_delay).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(2, args.epochs // 3))

        def _progress(epoch, n_epochs, train_stats, val_stats, conn_results, model, elapsed_seconds):
            print(f'    epoch {epoch:2d}/{n_epochs}: val={val_stats["loss"]:.4f} '
                  f'conn_AUC={conn_results["auc"]:.4f} ({elapsed_seconds:.0f}s)')

        train_voltage_model_with_early_stopping(
            model, train_loader, val_loader, optimizer, scheduler,
            train_epoch_fn=train_epoch_events, evaluate_windows_fn=evaluate_event_windows,
            evaluate_connectivity_fn=evaluate_connectivity, neighbor_indices=neighbor_indices,
            true_binary=true_binary, true_weights=true_weights, device=args.device,
            warmup=args.warmup, pos_weight=args.pos_weight, l1_lambda=args.l1,
            voltage_lambda=args.voltage_lambda, n_epochs=args.epochs, patience=None,
            log_fn=None, progress_fn=_progress)
        conn_matrix = model.get_connectivity_matrix(neighbor_indices)
        threshold_mode = model.threshold_mode

    # Observed flattened scores/labels in canonical candidate order.
    obs_scores = flatten_candidate_scores(conn_matrix, neighbor_indices,
                                          neuron_ids=np.arange(n_neurons), absolute=True)
    signed = np.concatenate([conn_matrix[i, neighbor_indices[i]] for i in range(n_neurons)])
    labels = flatten_labels(true_binary, neighbor_indices, n_neurons)
    true_sign = np.concatenate([np.sign(true_weights[i, neighbor_indices[i]]) for i in range(n_neurons)])
    score_neuron_ids = np.concatenate([
        np.full(len(neighbor_indices[i]), i, dtype=np.int32) for i in range(n_neurons)])

    rank = print_frontier(obs_scores, labels, signed, true_sign)

    def fit_null(surrogate_null, jitter_bins):
        return estimate_surrogate_connectivity_score_sets(
            spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
            n_neurons=n_neurons, K_actual=k_actual, max_delay=args.max_delay,
            threshold_mode=threshold_mode, lr=args.lr, batch_size=args.batch, warmup=args.warmup,
            pos_weight=args.pos_weight, l1_lambda=args.l1, voltage_lambda=args.voltage_lambda,
            pre_context=args.pre_context, post_context=args.post_context,
            neg_ratio=args.neg_ratio, neg_min_distance=args.neg_min_dist,
            boundaries=boundaries, excluded_bins=excluded_bins, val_fraction=args.val_fraction,
            device=args.device, n_surrogates=args.n_surrogates, surrogate_epochs=args.surrogate_epochs,
            surrogate_patience=args.surrogate_patience, surrogate_seed=args.surrogate_seed,
            surrogate_null=surrogate_null, jitter_bins=jitter_bins)

    windows = parse_windows(args.jitter_bins_list)
    print(f'\n  Fitting nulls (target_fdr={args.target_fdr:g}) for windows {windows}...')
    rows = []
    null_by_window = {}
    for jb in windows:
        t0 = time.time()
        nulls = fit_null('interval_jitter', jb)
        null_by_window[jb] = nulls
        pc = validate_jitter_null(obs_scores, labels, {jb: nulls},
                                  nonedge_percentile=args.nonedge_percentile)[0]
        cutoff, tm = evaluate_null(nulls, obs_scores, labels, score_neuron_ids, args.target_fdr)
        rows.append({'null': f'jitter_{jb}', 'jitter_bins': jb, 'power': pc['power'],
                     'calibration': pc['calibration'], 'cutoff': cutoff,
                     'precision': tm['precision'], 'recall': tm['recall'],
                     'fdr': tm['fdr'], 'fp': int(tm['fp'])})
        print(f'    jitter_{jb}: power={pc["power"]:.3f} cal={pc["calibration"]:.3f} '
              f'cutoff={cutoff:.4f} prec={tm["precision"]:.3f} recall={tm["recall"]:.3f} '
              f'trueFDR={tm["fdr"]:.3f} FP={int(tm["fp"])} ({time.time() - t0:.0f}s)')

    circ_row = None
    if not args.skip_circular:
        nulls = fit_null('circular_shift', 0)
        pc = validate_jitter_null(obs_scores, labels, {0: nulls},
                                  nonedge_percentile=args.nonedge_percentile)[0]
        cutoff, tm = evaluate_null(nulls, obs_scores, labels, score_neuron_ids, args.target_fdr)
        circ_row = {'null': 'circular_shift', 'power': pc['power'], 'calibration': pc['calibration'],
                    'cutoff': cutoff, 'precision': tm['precision'], 'recall': tm['recall'],
                    'fdr': tm['fdr'], 'fp': int(tm['fp'])}

    # ---- report ----------------------------------------------------------------
    print(f"\n{'=' * 90}\n  SWEEP (target_fdr={args.target_fdr:g}; power small=good, calibration~1=good, trueFDR~target=good)")
    print(f"{'=' * 90}")
    hdr = (f'  {"null":<14}{"power":>8}{"calib":>8}{"cutoff":>9}{"true_prec":>10}'
           f'{"true_recall":>12}{"true_FDR":>9}{"FP":>7}')
    print(hdr)
    if circ_row is not None:
        r = circ_row
        print(f'  {r["null"]:<14}{r["power"]:>8.3f}{r["calibration"]:>8.3f}{r["cutoff"]:>9.4f}'
              f'{r["precision"]:>10.3f}{r["recall"]:>12.3f}{r["fdr"]:>9.3f}{r["fp"]:>7d}')
    for r in rows:
        print(f'  {r["null"]:<14}{r["power"]:>8.3f}{r["calibration"]:>8.3f}{r["cutoff"]:>9.4f}'
              f'{r["precision"]:>10.3f}{r["recall"]:>12.3f}{r["fdr"]:>9.3f}{r["fp"]:>7d}')

    # selection: among windows with full edge-collapse (power<=power_max), true_FDR closest to target.
    passing = [r for r in rows if r['power'] <= args.power_max]
    print(f"\n{'=' * 90}")
    if passing:
        best = min(passing, key=lambda r: abs(r['fdr'] - args.target_fdr))
        print(f'  RECOMMENDED jitter_bins = {best["jitter_bins"]}: power={best["power"]:.3f} '
              f'(<= {args.power_max:g}, full edge-collapse), true_FDR={best["fdr"]:.3f} '
              f'closest to target {args.target_fdr:g} -> precision={best["precision"]:.3f} '
              f'recall={best["recall"]:.3f} FP={best["fp"]}.')
    else:
        print(f'  NO window kept power<={args.power_max:g}. Increase windows above max_delay={args.max_delay}.')
    print(f"{'=' * 90}")

    out_path = args.out or os.path.join(
        REPO_ROOT, 'voltage_augmented_learned_lif_outputs',
        f'jitter_sweep_{os.path.basename(args.session)}_tfdr{args.target_fdr:g}.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['null', 'jitter_bins', 'power_edge_collapse', 'calibration_ratio',
                    'surrogate_cutoff', 'true_precision', 'true_recall', 'true_FDR', 'FP',
                    'target_fdr', 'auc', 'ap', 'inh_auprc'])
        allrows = ([circ_row] if circ_row is not None else []) + rows
        for r in allrows:
            w.writerow([r['null'], r.get('jitter_bins', ''), f'{r["power"]:.6f}',
                        f'{r["calibration"]:.6f}', f'{r["cutoff"]:.6f}', f'{r["precision"]:.6f}',
                        f'{r["recall"]:.6f}', f'{r["fdr"]:.6f}', r['fp'], args.target_fdr,
                        f'{rank["auc"]:.6f}', f'{rank["ap"]:.6f}', f'{rank["inh_auprc"]:.6f}'])
    print(f'  Saved table: {out_path}')


if __name__ == '__main__':
    main()
