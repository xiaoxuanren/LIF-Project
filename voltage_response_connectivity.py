"""
Voltage-response connectivity baseline from saved spike trains and voltage traces.

This pipeline follows a conservative use of saved voltage traces:
1. Keep presynaptic spikes as the event source.
2. Remove synthetic spike peaks and mask short neighborhoods around each
   postsynaptic spike.
3. Normalize each neuron's remaining subthreshold voltage.
4. Score each directed pair (pre -> post) by the average change in the
   postsynaptic subthreshold voltage after presynaptic spikes relative to a
   short pre-spike baseline window.

Outputs:
  - JSON summary with metrics and preprocessing stats
  - NPZ artifact with the directed score matrix and edge selections
  - PNG summary figure
  - CSV of top directed edges

Usage:
  python voltage_response_connectivity.py --session "LIF data/20260425_110211"
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Optional

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_network_data(session_dir):
    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f'No network file found in {session_dir}')
    return np.load(net_files[0], allow_pickle=True)


def load_recordings(session_dir):
    rec_files = sorted(glob.glob(os.path.join(session_dir, 'recording[0-9][0-9][0-9].npz')))
    if not rec_files:
        raise FileNotFoundError(f'No recordings found in {session_dir}')

    recordings = []
    for rec_file in rec_files:
        data = np.load(rec_file, allow_pickle=True)
        if 'voltage_traces' not in data:
            raise KeyError(
                f'voltage_traces missing from {rec_file}. '
                'This session cannot be used by the voltage-response pipeline.'
            )
        sample_rate_ms = float(data['voltage_sample_rate'])
        recordings.append({
            'path': rec_file,
            'spike_times': data['spike_times'],
            'voltage_traces': data['voltage_traces'].astype(np.float32),
            'sample_rate_ms': sample_rate_ms,
            'duration_ms': float(data['duration']),
        })
    return recordings


def build_ground_truth(connections, n_neurons):
    true_weights = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    true_binary = np.zeros((n_neurons, n_neurons), dtype=bool)
    for conn in connections:
        pre_id = int(conn[0])
        post_id = int(conn[1])
        weight = float(conn[2])
        true_weights[post_id, pre_id] = weight
        true_binary[post_id, pre_id] = True
    np.fill_diagonal(true_binary, False)
    np.fill_diagonal(true_weights, 0.0)
    return true_weights, true_binary


def spike_times_to_sample_bins(spike_times, sample_rate_ms, n_samples):
    spikes = np.asarray(spike_times, dtype=np.float64)
    if spikes.size == 0:
        return np.empty(0, dtype=np.int32)
    bins = np.floor(spikes / float(sample_rate_ms) + 1e-9).astype(np.int32)
    bins = bins[(bins >= 0) & (bins < n_samples)]
    if bins.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(bins)


def ms_to_bins(time_ms, sample_rate_ms):
    return max(1, int(round(float(time_ms) / float(sample_rate_ms))))


def preprocess_voltage_recording(voltage_traces, spike_times, sample_rate_ms,
                                 mask_pre_ms=1.0, mask_post_ms=5.0,
                                 peak_threshold_mv=15.0):
    n_neurons, n_samples = voltage_traces.shape
    mask_pre_bins = int(round(float(mask_pre_ms) / float(sample_rate_ms)))
    mask_post_bins = int(round(float(mask_post_ms) / float(sample_rate_ms)))

    valid_mask = np.ones((n_neurons, n_samples), dtype=bool)
    if peak_threshold_mv is not None:
        valid_mask &= voltage_traces < float(peak_threshold_mv)

    for neuron_id in range(n_neurons):
        spike_bins = spike_times_to_sample_bins(
            spike_times[neuron_id], sample_rate_ms, n_samples,
        )
        for spike_bin in spike_bins:
            start = max(0, int(spike_bin) - mask_pre_bins)
            end = min(n_samples, int(spike_bin) + mask_post_bins + 1)
            valid_mask[neuron_id, start:end] = False

    cleaned = voltage_traces.astype(np.float32, copy=True)
    normalized = np.zeros_like(cleaned, dtype=np.float32)
    baseline_medians = np.zeros(n_neurons, dtype=np.float32)
    baseline_scales = np.ones(n_neurons, dtype=np.float32)
    valid_fraction = np.zeros(n_neurons, dtype=np.float32)

    for neuron_id in range(n_neurons):
        valid_values = cleaned[neuron_id, valid_mask[neuron_id]]
        if valid_values.size == 0:
            baseline = float(np.median(cleaned[neuron_id]))
            scale = float(np.std(cleaned[neuron_id]))
        else:
            baseline = float(np.median(valid_values))
            scale = float(np.std(valid_values))

        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        baseline_medians[neuron_id] = baseline
        baseline_scales[neuron_id] = scale
        valid_fraction[neuron_id] = float(valid_mask[neuron_id].mean())

        normalized_row = (cleaned[neuron_id] - baseline) / scale
        normalized_row[~valid_mask[neuron_id]] = 0.0
        normalized[neuron_id] = normalized_row

    return {
        'normalized_voltage': normalized,
        'valid_mask': valid_mask.astype(np.float32),
        'baseline_medians': baseline_medians,
        'baseline_scales': baseline_scales,
        'valid_fraction': valid_fraction,
    }


def build_prefix_sum(matrix):
    cumulative = np.cumsum(matrix, axis=1, dtype=np.float64)
    return np.pad(cumulative, ((0, 0), (1, 0)), mode='constant')


def window_sum(prefix, starts, ends):
    return prefix[:, ends] - prefix[:, starts]


def accumulate_voltage_response_scores(recordings, lag_min_ms=1.0, lag_max_ms=8.0,
                                       baseline_ms=8.0, mask_pre_ms=1.0,
                                       mask_post_ms=5.0, peak_threshold_mv=15.0):
    n_neurons = len(recordings[0]['spike_times'])
    score_sum = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    score_count = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    recording_summaries = []

    for recording in recordings:
        voltage_traces = recording['voltage_traces']
        sample_rate_ms = recording['sample_rate_ms']
        n_samples = voltage_traces.shape[1]

        lag_min_bins = ms_to_bins(lag_min_ms, sample_rate_ms)
        lag_max_bins = ms_to_bins(lag_max_ms, sample_rate_ms)
        baseline_bins = ms_to_bins(baseline_ms, sample_rate_ms)
        if lag_max_bins < lag_min_bins:
            raise ValueError('lag_max_ms must be >= lag_min_ms')

        processed = preprocess_voltage_recording(
            voltage_traces,
            recording['spike_times'],
            sample_rate_ms,
            mask_pre_ms=mask_pre_ms,
            mask_post_ms=mask_post_ms,
            peak_threshold_mv=peak_threshold_mv,
        )
        normalized_voltage = processed['normalized_voltage']
        valid_mask = processed['valid_mask']
        prefix_voltage = build_prefix_sum(normalized_voltage)
        prefix_valid = build_prefix_sum(valid_mask)

        valid_event_total = 0
        pre_spike_total = 0

        for pre_id in range(n_neurons):
            spike_bins = spike_times_to_sample_bins(
                recording['spike_times'][pre_id], sample_rate_ms, n_samples,
            )
            if spike_bins.size == 0:
                continue

            valid_events = (
                (spike_bins - baseline_bins >= 0) &
                (spike_bins + lag_max_bins + 1 <= n_samples)
            )
            spike_bins = spike_bins[valid_events]
            if spike_bins.size == 0:
                continue

            future_starts = spike_bins + lag_min_bins
            future_ends = spike_bins + lag_max_bins + 1
            past_starts = spike_bins - baseline_bins
            past_ends = spike_bins

            future_sum = window_sum(prefix_voltage, future_starts, future_ends)
            future_count = window_sum(prefix_valid, future_starts, future_ends)
            past_sum = window_sum(prefix_voltage, past_starts, past_ends)
            past_count = window_sum(prefix_valid, past_starts, past_ends)

            valid_pairs = (future_count > 0) & (past_count > 0)
            if not np.any(valid_pairs):
                continue

            response = np.zeros_like(future_sum, dtype=np.float64)
            np.divide(future_sum, future_count, out=response, where=future_count > 0)

            past_mean = np.zeros_like(past_sum, dtype=np.float64)
            np.divide(past_sum, past_count, out=past_mean, where=past_count > 0)

            response -= past_mean
            response[~valid_pairs] = 0.0

            score_sum[:, pre_id] += response.sum(axis=1)
            score_count[:, pre_id] += valid_pairs.sum(axis=1)
            valid_event_total += int(valid_pairs.sum())
            pre_spike_total += int(spike_bins.size)

        recording_summaries.append({
            'path': recording['path'],
            'duration_ms': float(recording['duration_ms']),
            'sample_rate_ms': float(sample_rate_ms),
            'mean_valid_fraction': float(processed['valid_fraction'].mean()),
            'min_valid_fraction': float(processed['valid_fraction'].min()),
            'max_valid_fraction': float(processed['valid_fraction'].max()),
            'usable_presynaptic_spikes': int(pre_spike_total),
            'valid_pair_events': int(valid_event_total),
        })

    score_matrix = np.zeros_like(score_sum, dtype=np.float64)
    np.divide(score_sum, score_count, out=score_matrix, where=score_count > 0)
    np.fill_diagonal(score_matrix, 0.0)
    np.fill_diagonal(score_count, 0.0)

    return {
        'score_matrix': score_matrix.astype(np.float32),
        'valid_event_count': score_count.astype(np.int32),
        'recording_summaries': recording_summaries,
    }


def describe_values(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'mean_abs': 0.0,
            'median': 0.0,
            'median_abs': 0.0,
            'p10': 0.0,
            'p90': 0.0,
        }

    abs_values = np.abs(values)
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'mean_abs': float(abs_values.mean()),
        'median': float(np.median(values)),
        'median_abs': float(np.median(abs_values)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
    }


def sign_accuracy(pred_scores, true_weights):
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    true_weights = np.asarray(true_weights, dtype=np.float32)
    if pred_scores.size == 0:
        return 0.0
    return float(np.mean(np.sign(pred_scores) == np.sign(true_weights)))


def evaluate_score_matrix(score_matrix, true_weights, true_binary, threshold=None):
    n_neurons = score_matrix.shape[0]
    mask = ~np.eye(n_neurons, dtype=bool)

    flat_scores = score_matrix[mask].astype(np.float32)
    flat_abs_scores = np.abs(flat_scores)
    flat_true_weights = true_weights[mask].astype(np.float32)
    flat_labels = true_binary[mask].astype(np.int32)

    metrics: dict[str, Any] = {
        'n_neurons': int(n_neurons),
        'n_pairs': int(mask.sum()),
        'n_true_edges': int(flat_labels.sum()),
    }

    if len(np.unique(flat_labels)) > 1:
        metrics['auc'] = float(roc_auc_score(flat_labels, flat_abs_scores))
        metrics['ap'] = float(average_precision_score(flat_labels, flat_abs_scores))
    else:
        metrics['auc'] = 0.0
        metrics['ap'] = 0.0

    if threshold is None:
        precision_curve, recall_curve, thresholds = precision_recall_curve(flat_labels, flat_abs_scores)
        if thresholds.size > 0:
            f1_scores = 2.0 * precision_curve[:-1] * recall_curve[:-1] / (
                precision_curve[:-1] + recall_curve[:-1] + 1e-10
            )
            best_idx = int(np.argmax(f1_scores))
            threshold = float(thresholds[best_idx])
        else:
            threshold = 0.0
    metrics['threshold'] = float(threshold)

    predicted = flat_abs_scores >= float(threshold)
    tp = int(np.sum(predicted & (flat_labels == 1)))
    fp = int(np.sum(predicted & (flat_labels == 0)))
    fn = int(np.sum((~predicted) & (flat_labels == 1)))
    tn = int(np.sum((~predicted) & (flat_labels == 0)))

    metrics['tp'] = tp
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tn'] = tn
    metrics['precision'] = float(tp / (tp + fp + 1e-10))
    metrics['recall'] = float(tp / (tp + fn + 1e-10))
    metrics['f1'] = float(2.0 * tp / (2.0 * tp + fp + fn + 1e-10))
    metrics['n_predicted'] = int(predicted.sum())

    true_edge_scores = flat_scores[flat_labels == 1]
    true_edge_weights = flat_true_weights[flat_labels == 1]
    true_exc_scores = flat_scores[flat_true_weights > 0]
    true_inh_scores = flat_scores[flat_true_weights < 0]
    no_edge_scores = flat_scores[flat_labels == 0]
    predicted_scores = flat_scores[predicted]
    predicted_tp_scores = flat_scores[predicted & (flat_labels == 1)]
    predicted_tp_true_weights = flat_true_weights[predicted & (flat_labels == 1)]

    metrics['true_edge_sign_accuracy'] = sign_accuracy(true_edge_scores, true_edge_weights)
    metrics['predicted_edge_sign_accuracy'] = sign_accuracy(
        predicted_tp_scores,
        predicted_tp_true_weights,
    )
    metrics['true_excitatory_score_stats'] = describe_values(true_exc_scores)
    metrics['true_inhibitory_score_stats'] = describe_values(true_inh_scores)
    metrics['no_edge_score_stats'] = describe_values(no_edge_scores)
    metrics['predicted_score_stats'] = describe_values(predicted_scores)

    if true_edge_scores.size > 1:
        metrics['true_edge_score_weight_corr'] = float(np.corrcoef(true_edge_scores, true_edge_weights)[0, 1])
    else:
        metrics['true_edge_score_weight_corr'] = 0.0

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(flat_labels, flat_abs_scores)
    return {
        'metrics': metrics,
        'flat_scores': flat_scores,
        'flat_abs_scores': flat_abs_scores,
        'flat_true_weights': flat_true_weights,
        'flat_labels': flat_labels,
        'predicted_flat': predicted,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'pr_thresholds': pr_thresholds,
    }


def select_directed_edges(score_matrix, true_weights, true_binary,
                          top_k: Optional[int] = 400, threshold=None):
    n_neurons = score_matrix.shape[0]
    mask = ~np.eye(n_neurons, dtype=bool)
    post_ids, pre_ids = np.where(mask)
    scores = score_matrix[post_ids, pre_ids]
    abs_scores = np.abs(scores)

    if threshold is None:
        keep = np.ones_like(abs_scores, dtype=bool)
    else:
        keep = abs_scores >= float(threshold)

    candidate_indices = np.flatnonzero(keep)
    if top_k is not None:
        if top_k == 0:
            candidate_indices = np.empty(0, dtype=np.int64)
        elif len(candidate_indices) > top_k:
            ranked = np.argsort(abs_scores[candidate_indices])[::-1][:top_k]
            candidate_indices = candidate_indices[ranked]

    if candidate_indices.size > 0:
        order = np.argsort(abs_scores[candidate_indices])[::-1]
        candidate_indices = candidate_indices[order]

    return {
        'pre_ids': pre_ids[candidate_indices].astype(np.int32),
        'post_ids': post_ids[candidate_indices].astype(np.int32),
        'scores': scores[candidate_indices].astype(np.float32),
        'abs_scores': abs_scores[candidate_indices].astype(np.float32),
        'true_weights': true_weights[post_ids[candidate_indices], pre_ids[candidate_indices]].astype(np.float32),
        'true_connected': true_binary[post_ids[candidate_indices], pre_ids[candidate_indices]].astype(np.int32),
    }


def compute_plot_order(cluster_assignments, n_neurons):
    if cluster_assignments is None:
        return np.arange(n_neurons)
    return np.argsort(np.asarray(cluster_assignments))


def plot_heatmap(ax, score_matrix, cluster_assignments, title):
    n_neurons = score_matrix.shape[0]
    order = compute_plot_order(cluster_assignments, n_neurons)
    sorted_matrix = score_matrix[order][:, order]
    mask = ~np.eye(n_neurons, dtype=bool)
    vmax = float(np.percentile(np.abs(score_matrix[mask]), 99)) if np.any(mask) else 1.0
    vmax = max(vmax, 1e-3)

    im = ax.imshow(sorted_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Presynaptic neuron (cluster-sorted)')
    ax.set_ylabel('Postsynaptic neuron (cluster-sorted)')

    if cluster_assignments is not None:
        sorted_clusters = np.asarray(cluster_assignments)[order]
        boundaries = np.where(np.diff(sorted_clusters) != 0)[0] + 0.5
        for boundary in boundaries:
            ax.axhline(boundary, color='black', linewidth=0.35, alpha=0.35)
            ax.axvline(boundary, color='black', linewidth=0.35, alpha=0.35)

    return im


def plot_edge_map(ax, positions, selection, title):
    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=14, c='white', edgecolors='black', linewidths=0.45, zorder=3,
    )

    scores = selection['scores']
    max_abs = float(np.max(np.abs(scores))) if scores.size > 0 else 1.0
    max_abs = max(max_abs, 1e-6)
    draw_order = np.argsort(np.abs(scores))
    for edge_idx in draw_order:
        score = scores[edge_idx]
        pre_id = int(selection['pre_ids'][edge_idx])
        post_id = int(selection['post_ids'][edge_idx])
        color = 'crimson' if score >= 0 else 'royalblue'
        strength = abs(float(score)) / max_abs
        linewidth = 0.8 + 2.4 * strength
        alpha = 0.35 + 0.55 * strength
        ax.plot(
            [positions[pre_id, 0], positions[post_id, 0]],
            [positions[pre_id, 1], positions[post_id, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle='round',
            zorder=2,
        )

    ax.set_title(title)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_aspect('equal')
    ax.text(
        0.02, 0.98,
        'Line direction omitted on map\nHeatmap keeps direction',
        transform=ax.transAxes,
        va='top', ha='left', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )


def plot_summary_figure(session_name, score_matrix, true_weights, true_binary,
                        cluster_assignments, positions, evaluation,
                        top_edge_selection, output_path):
    metrics = evaluation['metrics']
    flat_scores = evaluation['flat_scores']
    flat_abs_scores = evaluation['flat_abs_scores']
    flat_true_weights = evaluation['flat_true_weights']
    flat_labels = evaluation['flat_labels']

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Voltage-Response Connectivity - {session_name}', fontsize=15, fontweight='bold')

    heatmap = plot_heatmap(
        axes[0, 0],
        score_matrix,
        cluster_assignments,
        'Directed score matrix (post row, pre column)',
    )
    fig.colorbar(heatmap, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Voltage-response score')

    ax = axes[0, 1]
    ax.hist(flat_scores[flat_labels == 0], bins=60, alpha=0.55, color='gray', label='No edge')
    ax.hist(flat_scores[flat_true_weights > 0], bins=60, alpha=0.55, color='crimson', label='True excitatory')
    ax.hist(flat_scores[flat_true_weights < 0], bins=60, alpha=0.55, color='royalblue', label='True inhibitory')
    ax.axvline(0.0, color='black', linewidth=1.0)
    ax.set_title('Signed score distribution by truth class')
    ax.set_xlabel('Voltage-response score')
    ax.set_ylabel('Pair count')
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.hist(flat_abs_scores[flat_labels == 1], bins=60, alpha=0.65, color='seagreen', label='True edges')
    ax.hist(flat_abs_scores[flat_labels == 0], bins=60, alpha=0.55, color='darkorange', label='No edge')
    ax.axvline(metrics['threshold'], color='black', linestyle='--', linewidth=1.5,
               label=f"Threshold={metrics['threshold']:.4f}")
    ax.set_title('Absolute score distribution')
    ax.set_xlabel('|Voltage-response score|')
    ax.set_ylabel('Pair count')
    ax.legend(fontsize=8)

    plot_edge_map(
        axes[1, 0],
        positions,
        top_edge_selection,
        f'Top directed scores (n={len(top_edge_selection["scores"])})',
    )

    ax = axes[1, 1]
    ax.plot(evaluation['recall_curve'], evaluation['precision_curve'], color='black', linewidth=2.0)
    ax.set_title('Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)

    predicted_scores = flat_scores[evaluation['predicted_flat']]
    predicted_positive = int(np.sum(predicted_scores > 0))
    predicted_negative = int(np.sum(predicted_scores < 0))

    ax = axes[1, 2]
    ax.axis('off')
    summary_lines = [
        f"AUC: {metrics['auc']:.4f}",
        f"AP: {metrics['ap']:.4f}",
        f"Threshold: {metrics['threshold']:.4f}",
        f"TP / FP / FN: {metrics['tp']} / {metrics['fp']} / {metrics['fn']}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1: {metrics['f1']:.4f}",
        f"True-edge sign accuracy: {metrics['true_edge_sign_accuracy']:.4f}",
        f"Predicted sign count (+/-): {predicted_positive} / {predicted_negative}",
        f"True-edge score/weight corr: {metrics['true_edge_score_weight_corr']:.4f}",
        '',
        'Method:',
        'Pre spikes -> post subthreshold voltage change',
        'Post spikes masked before scoring',
        'Synthetic +20 mV peaks excluded',
    ]
    ax.text(
        0.02, 0.98,
        '\n'.join(summary_lines),
        va='top', ha='left', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_edge_csv(output_path, selection):
    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'pre_id', 'post_id', 'score', 'abs_score',
            'true_connected', 'true_weight', 'sign_match_if_true',
        ])
        for edge_idx, score in enumerate(selection['scores']):
            true_weight = float(selection['true_weights'][edge_idx])
            true_connected = int(selection['true_connected'][edge_idx])
            sign_match = ''
            if true_connected:
                sign_match = int(np.sign(score) == np.sign(true_weight))
            writer.writerow([
                int(selection['pre_ids'][edge_idx]),
                int(selection['post_ids'][edge_idx]),
                float(score),
                float(selection['abs_scores'][edge_idx]),
                true_connected,
                true_weight,
                sign_match,
            ])


def select_session():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LIF data')
    sessions = sorted([path for path in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(path)])
    if not sessions:
        print('No sessions in LIF data/')
        sys.exit(1)

    print('\nSessions:')
    for idx, session_path in enumerate(sessions):
        print(f'  [{idx}] {os.path.basename(session_path)}')
    choice = input('Select (Enter=last): ').strip()
    return sessions[-1] if choice == '' else sessions[int(choice)]


def run_pipeline(session_dir, lag_min_ms=1.0, lag_max_ms=8.0, baseline_ms=8.0,
                 mask_pre_ms=1.0, mask_post_ms=5.0, peak_threshold_mv=15.0,
                 top_k_map=400, threshold=None, output_tag=None):
    session_name = os.path.basename(session_dir)
    output_tag = output_tag.strip().replace(' ', '_') if output_tag else None
    output_name = session_name if not output_tag else f'{session_name}_{output_tag}'

    print('\n' + '=' * 70)
    print('VOLTAGE-RESPONSE CONNECTIVITY')
    print(f'Session: {session_name}')
    if output_tag:
        print(f'Output tag: {output_tag}')
    print(
        f'Lag window: {lag_min_ms:.1f}-{lag_max_ms:.1f} ms | '
        f'Baseline: {baseline_ms:.1f} ms | '
        f'Mask: -{mask_pre_ms:.1f}/+{mask_post_ms:.1f} ms'
    )
    print('=' * 70)

    net_data = load_network_data(session_dir)
    recordings = load_recordings(session_dir)
    n_neurons = len(net_data['neuron_positions'])
    true_weights, true_binary = build_ground_truth(net_data['connections'], n_neurons)

    print(f'  Loaded {len(recordings)} recordings')
    print(f'  Neurons: {n_neurons}')
    print('  Accumulating directed voltage-response scores...')
    response = accumulate_voltage_response_scores(
        recordings,
        lag_min_ms=lag_min_ms,
        lag_max_ms=lag_max_ms,
        baseline_ms=baseline_ms,
        mask_pre_ms=mask_pre_ms,
        mask_post_ms=mask_post_ms,
        peak_threshold_mv=peak_threshold_mv,
    )
    score_matrix = response['score_matrix']

    evaluation = evaluate_score_matrix(
        score_matrix,
        true_weights,
        true_binary,
        threshold=threshold,
    )
    metrics = evaluation['metrics']

    top_edge_selection = select_directed_edges(
        score_matrix,
        true_weights,
        true_binary,
        top_k=top_k_map,
        threshold=None,
    )
    threshold_selection = select_directed_edges(
        score_matrix,
        true_weights,
        true_binary,
        top_k=None,
        threshold=metrics['threshold'],
    )

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voltage_connectivity_outputs')
    os.makedirs(output_dir, exist_ok=True)

    figure_path = os.path.join(output_dir, f'voltage_response_{output_name}.png')
    stats_path = os.path.join(output_dir, f'voltage_response_{output_name}.json')
    npz_path = os.path.join(output_dir, f'voltage_response_{output_name}.npz')
    csv_path = os.path.join(output_dir, f'voltage_response_top_edges_{output_name}.csv')

    plot_summary_figure(
        session_name,
        score_matrix,
        true_weights,
        true_binary,
        net_data['cluster_assignments'] if 'cluster_assignments' in net_data else None,
        net_data['neuron_positions'],
        evaluation,
        top_edge_selection,
        figure_path,
    )
    save_edge_csv(csv_path, top_edge_selection)

    summary = {
        'session_name': session_name,
        'n_recordings': int(len(recordings)),
        'n_neurons': int(n_neurons),
        'lag_min_ms': float(lag_min_ms),
        'lag_max_ms': float(lag_max_ms),
        'baseline_ms': float(baseline_ms),
        'mask_pre_ms': float(mask_pre_ms),
        'mask_post_ms': float(mask_post_ms),
        'peak_threshold_mv': None if peak_threshold_mv is None else float(peak_threshold_mv),
        'metrics': metrics,
        'recording_summaries': response['recording_summaries'],
        'top_edge_count': int(len(top_edge_selection['scores'])),
        'threshold_edge_count': int(len(threshold_selection['scores'])),
        'figure_path': figure_path,
        'npz_path': npz_path,
        'csv_path': csv_path,
    }

    with open(stats_path, 'w') as handle:
        json.dump(summary, handle, indent=2)

    np.savez_compressed(
        npz_path,
        score_matrix=score_matrix.astype(np.float32),
        valid_event_count=response['valid_event_count'].astype(np.int32),
        threshold=np.float32(metrics['threshold']),
        predicted_binary=(np.abs(score_matrix) >= metrics['threshold']).astype(np.int8),
        top_pre_ids=top_edge_selection['pre_ids'],
        top_post_ids=top_edge_selection['post_ids'],
        top_scores=top_edge_selection['scores'],
        threshold_pre_ids=threshold_selection['pre_ids'],
        threshold_post_ids=threshold_selection['post_ids'],
        threshold_scores=threshold_selection['scores'],
    )

    print(f"  AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f} F1={metrics['f1']:.4f}")
    print(f"  TP / FP / FN: {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
    print(f'  Figure saved: {figure_path}')
    print(f'  Stats saved: {stats_path}')
    print(f'  NPZ saved: {npz_path}')
    print(f'  CSV saved: {csv_path}')

    return summary


def main():
    parser = argparse.ArgumentParser(description='Voltage-response connectivity baseline')
    parser.add_argument('--session', type=str, default=None,
                        help='Path to session folder under LIF data/')
    parser.add_argument('--lag-min-ms', type=float, default=1.0,
                        help='Earliest post-spike voltage lag to score')
    parser.add_argument('--lag-max-ms', type=float, default=8.0,
                        help='Latest post-spike voltage lag to score')
    parser.add_argument('--baseline-ms', type=float, default=8.0,
                        help='Pre-spike baseline window length')
    parser.add_argument('--mask-pre-ms', type=float, default=1.0,
                        help='Voltage mask before each postsynaptic spike')
    parser.add_argument('--mask-post-ms', type=float, default=5.0,
                        help='Voltage mask after each postsynaptic spike')
    parser.add_argument('--peak-threshold-mv', type=float, default=15.0,
                        help='Voltage values at or above this are treated as spike peaks and excluded')
    parser.add_argument('--top-k-map', type=int, default=400,
                        help='Number of strongest directed edges to show in the map and CSV')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Optional absolute score threshold override')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional suffix for saved artifact names')
    args = parser.parse_args()

    session_dir = args.session if args.session else select_session()
    run_pipeline(
        session_dir=session_dir,
        lag_min_ms=args.lag_min_ms,
        lag_max_ms=args.lag_max_ms,
        baseline_ms=args.baseline_ms,
        mask_pre_ms=args.mask_pre_ms,
        mask_post_ms=args.mask_post_ms,
        peak_threshold_mv=args.peak_threshold_mv,
        top_k_map=args.top_k_map,
        threshold=args.threshold,
        output_tag=args.output_tag,
    )


if __name__ == '__main__':
    main()