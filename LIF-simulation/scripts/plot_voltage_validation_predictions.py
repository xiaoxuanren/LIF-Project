import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import torch
from torch.utils.data import DataLoader

from lif_inference.connectivity_metrics import select_connectivity_threshold
from lif_inference.event_windows import split_recording_boundaries
from lif_inference.shared_data import build_segmentwise_circular_shift_surrogates
from lif_inference.voltage_augmented_learned_lif_connectivity import (
    VoltageAugmentedPerNeuronLIF,
    build_train_val_voltage_datasets,
    load_all_recordings_with_voltage,
    load_single_recording_with_voltage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def safe_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        if all(str(key).startswith('module.') for key in state_dict.keys()):
            stripped = {str(key)[7:]: value for key, value in state_dict.items()}
            model.load_state_dict(stripped)
            return
        raise


def resolve_arg(value, fallback):
    return fallback if value is None else value


def clip_excluded_bins(excluded_bins, limit):
    if excluded_bins is None:
        return np.array([], dtype=np.int32)
    arr = np.asarray(excluded_bins, dtype=np.int32)
    return arr[(arr >= 0) & (arr < int(limit))]


def parse_single_recording_index(checkpoint):
    data_config = checkpoint.get('data_config', {})
    if 'recording_idx' in data_config:
        return int(data_config['recording_idx'])

    summaries = checkpoint.get('recording_summaries', [])
    if len(summaries) == 1:
        rec_path = summaries[0].get('path', '')
        stem = Path(rec_path).stem
        if stem.startswith('recording'):
            return int(stem.replace('recording', ''))
    return 0


def infer_session_dir(session_arg, checkpoint):
    if session_arg:
        return Path(session_arg)

    session_name = checkpoint.get('session_name')
    if not session_name:
        raise ValueError('Checkpoint is missing session_name; pass --session explicitly')

    candidate = REPO_ROOT / 'LIF data' / str(session_name)
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f'Could not infer session path for {session_name!r}; pass --session explicitly'
    )


def clip_boundaries(boundaries, limit):
    clipped = [int(boundary) for boundary in boundaries if int(boundary) <= int(limit)]
    if not clipped:
        clipped = [0]
    if clipped[-1] < int(limit):
        clipped.append(int(limit))
    return clipped


def masked_mean(values, mask):
    counts = mask.sum(axis=0).astype(np.float32)
    total = (values * mask).sum(axis=0)
    mean = np.full(values.shape[1], np.nan, dtype=np.float32)
    nonzero = counts > 0
    mean[nonzero] = (total[nonzero] / counts[nonzero]).astype(np.float32)
    return mean, counts


def compute_metrics(predicted_voltage, target_voltage, supervised_mask):
    valid_pred = predicted_voltage[supervised_mask]
    valid_target = target_voltage[supervised_mask]
    if valid_pred.size == 0:
        return {
            'n_supervised_points': 0,
            'mae': None,
            'rmse': None,
            'corr': None,
        }

    mae = float(np.mean(np.abs(valid_pred - valid_target)))
    rmse = float(np.sqrt(np.mean((valid_pred - valid_target) ** 2)))
    corr = None
    if valid_pred.size > 1:
        target_std = float(np.std(valid_target))
        pred_std = float(np.std(valid_pred))
        if target_std > 0.0 and pred_std > 0.0:
            corr_value = float(np.corrcoef(valid_target, valid_pred)[0, 1])
            if np.isfinite(corr_value):
                corr = corr_value

    return {
        'n_supervised_points': int(valid_pred.size),
        'mae': mae,
        'rmse': rmse,
        'corr': corr,
    }


def build_segment_slices(boundaries):
    arr = np.asarray(boundaries, dtype=np.int32)
    if arr.ndim != 1 or arr.size < 2:
        return []
    return [(int(start), int(end)) for start, end in zip(arr[:-1], arr[1:]) if int(end) > int(start)]


def concatenate_validation_segments(spike_matrix, segments):
    if not segments:
        return np.zeros((spike_matrix.shape[0], 0), dtype=np.float32), np.array([0], dtype=np.int32)

    pieces = []
    boundaries = [0]
    running = 0
    for start, end in segments:
        piece = spike_matrix[:, start:end].astype(np.float32, copy=False)
        pieces.append(piece)
        running += int(end - start)
        boundaries.append(running)
    return np.concatenate(pieces, axis=1), np.asarray(boundaries, dtype=np.int32)


def build_segment_warmup_mask(total_length, boundaries, warmup):
    mask = np.ones(int(total_length), dtype=bool)
    if int(total_length) <= 0:
        return mask
    warmup = max(int(warmup), 0)
    for start, end in build_segment_slices(boundaries):
        warm_end = min(int(start) + warmup, int(end))
        mask[int(start):warm_end] = False
    return mask


def build_spike_tolerance_mask(spike_matrix, tolerance_bins):
    spike_bool = np.asarray(spike_matrix) > 0.5
    tolerance_bins = max(int(tolerance_bins), 0)
    tolerance_mask = np.zeros_like(spike_bool, dtype=bool)
    for shift in range(-tolerance_bins, tolerance_bins + 1):
        if shift < 0:
            tolerance_mask[:, :shift] |= spike_bool[:, -shift:]
        elif shift > 0:
            tolerance_mask[:, shift:] |= spike_bool[:, :-shift]
        else:
            tolerance_mask |= spike_bool
    return tolerance_mask


def subsample_threshold_scores(scores, max_points, rng, tail_fraction=0.25):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    max_points = int(max_points)
    if max_points <= 0 or scores.size <= max_points:
        return scores

    tail_count = min(max(int(round(max_points * float(tail_fraction))), 1), max_points)
    order = np.argsort(scores)
    tail_idx = order[-tail_count:]
    remaining = order[:-tail_count]
    keep_random = max_points - tail_idx.size
    if keep_random > 0 and remaining.size > keep_random:
        random_idx = rng.choice(remaining, size=keep_random, replace=False)
        keep_idx = np.concatenate([tail_idx, random_idx])
    else:
        keep_idx = np.concatenate([tail_idx, remaining]) if remaining.size > 0 else tail_idx
    return scores[np.sort(keep_idx)]


def compute_tolerant_spike_metrics(actual_spikes, predicted_spike_calls, tolerance_bins):
    actual_bool = np.asarray(actual_spikes) > 0.5
    predicted_bool = np.asarray(predicted_spike_calls) > 0.5
    actual_tolerance = build_spike_tolerance_mask(actual_bool.astype(np.float32), tolerance_bins)
    predicted_tolerance = build_spike_tolerance_mask(predicted_bool.astype(np.float32), tolerance_bins)

    matched_predicted = predicted_bool & actual_tolerance
    unmatched_predicted = predicted_bool & ~actual_tolerance
    matched_actual = actual_bool & predicted_tolerance

    n_predicted = int(np.sum(predicted_bool))
    n_actual = int(np.sum(actual_bool))
    n_matched_predicted = int(np.sum(matched_predicted))
    n_matched_actual = int(np.sum(matched_actual))
    tolerant_precision = None
    tolerant_recall = None
    tolerant_f1 = None
    if n_predicted > 0:
        tolerant_precision = float(n_matched_predicted / n_predicted)
    if n_actual > 0:
        tolerant_recall = float(n_matched_actual / n_actual)
    if tolerant_precision is not None and tolerant_recall is not None:
        denom = tolerant_precision + tolerant_recall
        if denom > 0.0:
            tolerant_f1 = float(2.0 * tolerant_precision * tolerant_recall / denom)

    return {
        'n_actual_spikes': n_actual,
        'n_predicted_spikes': n_predicted,
        'n_matched_actual_spikes': n_matched_actual,
        'n_matched_predicted_spikes': n_matched_predicted,
        'n_unmatched_predicted_spikes': int(np.sum(unmatched_predicted)),
        'tolerant_precision': tolerant_precision,
        'tolerant_recall': tolerant_recall,
        'tolerant_f1': tolerant_f1,
        'matched_predicted_mask': matched_predicted,
        'unmatched_predicted_mask': unmatched_predicted,
    }


def collect_timeline_predictions(model, spike_matrix, neighbor_indices, neuron_ids, boundaries,
                                 device, batch_neurons):
    segments = build_segment_slices(boundaries)
    if not segments:
        return {
            'predicted_spike_prob': np.zeros((len(neuron_ids), 0), dtype=np.float32),
            'actual_spikes': np.zeros((len(neuron_ids), 0), dtype=np.float32),
            'boundaries': np.array([0], dtype=np.int32),
        }

    total_length = int(sum(end - start for start, end in segments))
    n_neurons = len(neuron_ids)
    predicted = np.zeros((n_neurons, total_length), dtype=np.float32)
    actual_concat, concat_boundaries = concatenate_validation_segments(spike_matrix, segments)

    offset = 0
    batch_neurons = max(int(batch_neurons), 1)
    with torch.no_grad():
        for start, end in segments:
            seg_len = int(end - start)
            segment_offset = offset
            for batch_start in range(0, n_neurons, batch_neurons):
                batch_end = min(batch_start + batch_neurons, n_neurons)
                batch_ids = np.asarray(neuron_ids[batch_start:batch_end], dtype=np.int64)
                pre_spikes = np.stack(
                    [spike_matrix[neighbor_indices[int(neuron_id)], start:end] for neuron_id in batch_ids],
                    axis=0,
                ).astype(np.float32, copy=False)
                dummy_post = np.zeros((batch_ids.shape[0], seg_len), dtype=np.float32)

                predicted_sp, _, _ = model(
                    torch.from_numpy(pre_spikes).to(device),
                    torch.from_numpy(dummy_post).to(device),
                    torch.from_numpy(batch_ids).to(device),
                    tbptt_len=seg_len,
                )
                predicted[batch_start:batch_end, segment_offset:segment_offset + seg_len] = (
                    predicted_sp.cpu().numpy().astype(np.float32)
                )
            offset += seg_len

    return {
        'predicted_spike_prob': predicted,
        'actual_spikes': actual_concat,
        'boundaries': concat_boundaries,
    }


def estimate_spike_surrogate_score_sets(model, spike_matrix, neighbor_indices, neuron_ids, boundaries,
                                        warmup, device, batch_neurons, n_surrogates,
                                        surrogate_seed=1234, min_shift_fraction=0.10,
                                        max_scores_per_set=250000):
    segments = build_segment_slices(boundaries)
    if not segments or int(n_surrogates) <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    validation_spikes, concat_boundaries = concatenate_validation_segments(spike_matrix, segments)
    warmup_mask = build_segment_warmup_mask(validation_spikes.shape[1], concat_boundaries, warmup)
    rng = np.random.default_rng(int(surrogate_seed))
    surrogate_score_sets = []

    for _ in range(int(n_surrogates)):
        surrogate_spikes, = build_segmentwise_circular_shift_surrogates(
            [validation_spikes],
            boundaries=concat_boundaries,
            rng=rng,
            min_shift_fraction=float(min_shift_fraction),
        )
        surrogate_predictions = collect_timeline_predictions(
            model,
            surrogate_spikes,
            neighbor_indices,
            neuron_ids,
            concat_boundaries,
            device=device,
            batch_neurons=batch_neurons,
        )
        surrogate_scores = surrogate_predictions['predicted_spike_prob'][:, warmup_mask].reshape(-1)
        surrogate_scores = subsample_threshold_scores(
            surrogate_scores,
            max_points=max_scores_per_set,
            rng=rng,
        )
        surrogate_score_sets.append(surrogate_scores.astype(np.float32, copy=False))

    return np.stack(surrogate_score_sets, axis=0) if surrogate_score_sets else np.zeros((0, 0), dtype=np.float32)


def select_spike_threshold(predicted_spike_prob, target_spikes, warmup, spike_threshold=None):
    if spike_threshold is not None:
        return float(spike_threshold), 'cli'

    eval_pred = predicted_spike_prob[:, warmup:].reshape(-1)
    eval_target = target_spikes[:, warmup:].reshape(-1)
    if eval_pred.size == 0 or np.unique(eval_target).size < 2:
        return 0.5, 'fallback_0p5'

    precision, recall, thresholds = precision_recall_curve(eval_target, eval_pred)
    if thresholds.size == 0:
        return 0.5, 'fallback_0p5'

    f1 = 2.0 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), 'validation_bin_f1'


def select_spike_threshold_from_scores(observed_scores, spike_threshold=None,
                                       threshold_mode='surrogate_fdr',
                                       surrogate_score_sets=None,
                                       surrogate_fdr=0.005):
    if spike_threshold is not None:
        return float(spike_threshold), 'cli', {
            'mode': 'cli',
            'threshold': float(spike_threshold),
        }

    observed_scores = np.asarray(observed_scores, dtype=np.float32).reshape(-1)
    if observed_scores.size == 0:
        return 0.5, 'fallback_0p5', {
            'mode': str(threshold_mode),
            'threshold': 0.5,
        }

    threshold_mode = str(threshold_mode).strip().lower()
    if threshold_mode != 'surrogate_fdr':
        raise ValueError(
            f'Unsupported spike_threshold_mode={threshold_mode!r}; '
            "use 'surrogate_fdr' or pass --spike-threshold explicitly"
        )

    threshold_info = select_connectivity_threshold(
        np.zeros(observed_scores.shape[0], dtype=np.int32),
        observed_scores,
        mode='surrogate_fdr',
        surrogate_score_sets=surrogate_score_sets,
        surrogate_fdr=float(surrogate_fdr),
        default_threshold=0.5,
    )
    return float(threshold_info['threshold']), 'surrogate_fdr', threshold_info


def compute_spike_metrics(predicted_spike_prob, target_spikes, warmup, spike_threshold):
    eval_pred = predicted_spike_prob[:, warmup:].reshape(-1)
    eval_target = target_spikes[:, warmup:].reshape(-1)
    if eval_pred.size == 0:
        return {
            'n_eval_bins': 0,
            'n_actual_spikes': 0,
            'actual_spike_rate': None,
            'brier': None,
            'bce': None,
            'auc': None,
            'ap': None,
            'spike_threshold': float(spike_threshold),
            'precision_at_threshold': None,
            'recall_at_threshold': None,
            'f1_at_threshold': None,
            'predicted_positive_rate': None,
            'mean_pred_on_spikes': None,
            'mean_pred_on_nonspikes': None,
        }

    clipped_pred = np.clip(eval_pred, 1e-6, 1.0 - 1e-6)
    spike_mask = eval_target > 0.5
    nonspike_mask = ~spike_mask
    brier = float(np.mean((eval_pred - eval_target) ** 2))
    bce = float(-np.mean(
        eval_target * np.log(clipped_pred) + (1.0 - eval_target) * np.log(1.0 - clipped_pred)
    ))

    auc = None
    ap = None
    if np.unique(eval_target).size > 1:
        auc_value = float(roc_auc_score(eval_target, eval_pred))
        ap_value = float(average_precision_score(eval_target, eval_pred))
        if np.isfinite(auc_value):
            auc = auc_value
        if np.isfinite(ap_value):
            ap = ap_value

    predicted_binary = eval_pred >= float(spike_threshold)
    tp = int(np.sum(predicted_binary & spike_mask))
    fp = int(np.sum(predicted_binary & nonspike_mask))
    fn = int(np.sum((~predicted_binary) & spike_mask))
    precision_at_threshold = None
    recall_at_threshold = None
    f1_at_threshold = None
    if (tp + fp) > 0:
        precision_at_threshold = float(tp / (tp + fp))
    if (tp + fn) > 0:
        recall_at_threshold = float(tp / (tp + fn))
    if precision_at_threshold is not None and recall_at_threshold is not None:
        denom = precision_at_threshold + recall_at_threshold
        if denom > 0.0:
            f1_at_threshold = float(2.0 * precision_at_threshold * recall_at_threshold / denom)

    return {
        'n_eval_bins': int(eval_pred.size),
        'n_actual_spikes': int(spike_mask.sum()),
        'actual_spike_rate': float(np.mean(eval_target)),
        'brier': brier,
        'bce': bce,
        'auc': auc,
        'ap': ap,
        'spike_threshold': float(spike_threshold),
        'precision_at_threshold': precision_at_threshold,
        'recall_at_threshold': recall_at_threshold,
        'f1_at_threshold': f1_at_threshold,
        'predicted_positive_rate': float(np.mean(predicted_binary)),
        'mean_pred_on_spikes': float(np.mean(eval_pred[spike_mask])) if np.any(spike_mask) else None,
        'mean_pred_on_nonspikes': float(np.mean(eval_pred[nonspike_mask])) if np.any(nonspike_mask) else None,
    }


def compute_window_spike_quality(predicted_spike_prob, target_spikes, warmup):
    n_windows = predicted_spike_prob.shape[0]
    quality = np.full(n_windows, np.nan, dtype=np.float32)
    mean_on_spikes = np.full(n_windows, np.nan, dtype=np.float32)
    mean_off_spikes = np.full(n_windows, np.nan, dtype=np.float32)
    spike_bce = np.full(n_windows, np.nan, dtype=np.float32)

    for idx in range(n_windows):
        pred = np.clip(predicted_spike_prob[idx, warmup:], 1e-6, 1.0 - 1e-6)
        target = target_spikes[idx, warmup:]
        if pred.size == 0:
            continue

        spike_bce[idx] = float(-np.mean(
            target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred)
        ))

        spike_mask = target > 0.5
        nonspike_mask = ~spike_mask
        if not np.any(spike_mask):
            continue

        mean_on = float(np.mean(pred[spike_mask]))
        mean_off = float(np.mean(pred[nonspike_mask])) if np.any(nonspike_mask) else 0.0
        mean_on_spikes[idx] = mean_on
        mean_off_spikes[idx] = mean_off
        quality[idx] = float(mean_on - mean_off)

    return {
        'quality': quality,
        'mean_on_spikes': mean_on_spikes,
        'mean_off_spikes': mean_off_spikes,
        'spike_bce': spike_bce,
    }


def plot_global_spike_raster(output_path, checkpoint, context, timeline_predictions,
                             spike_threshold, spike_threshold_source,
                             threshold_info, onset_tolerance_ms):
    actual_spikes = timeline_predictions['actual_spikes']
    predicted_spike_calls = timeline_predictions['predicted_spike_calls']
    boundaries = np.asarray(timeline_predictions['boundaries'], dtype=np.int32)
    matched_predicted_mask = timeline_predictions['matched_predicted_mask']
    unmatched_predicted_mask = timeline_predictions['unmatched_predicted_mask']
    tolerant_metrics = timeline_predictions['tolerant_metrics']
    neuron_ids = np.asarray(context['neuron_ids'], dtype=np.int32)
    dt = float(context['dt'])
    warmup = int(context['warmup'])

    n_neurons, total_length = actual_spikes.shape
    height = min(max(8.0, 2.0 + 0.035 * n_neurons), 26.0)
    fig, ax = plt.subplots(figsize=(18, height))
    time_axis = np.arange(total_length, dtype=np.float32) * dt

    actual_y, actual_x = np.where(actual_spikes > 0.5)
    if actual_x.size > 0:
        ax.scatter(
            time_axis[actual_x],
            2 * actual_y,
            color='tab:orange',
            marker='|',
            s=14,
            linewidths=0.8,
            rasterized=True,
            label='actual spikes',
        )

    matched_y, matched_x = np.where(matched_predicted_mask)
    if matched_x.size > 0:
        ax.scatter(
            time_axis[matched_x],
            2 * matched_y + 1,
            color='black',
            marker='|',
            s=14,
            linewidths=0.8,
            rasterized=True,
            label=f'predicted calls within +/- {onset_tolerance_ms:.1f} ms',
        )

    unmatched_y, unmatched_x = np.where(unmatched_predicted_mask)
    if unmatched_x.size > 0:
        ax.scatter(
            time_axis[unmatched_x],
            2 * unmatched_y + 1,
            color='tab:gray',
            marker='|',
            s=12,
            linewidths=0.6,
            alpha=0.7,
            rasterized=True,
            label='predicted calls outside tolerance',
        )

    for boundary in boundaries[1:-1]:
        ax.axvline(float(boundary) * dt, color='tab:blue', lw=0.8, ls='--', alpha=0.35)

    for start, end in build_segment_slices(boundaries):
        warmup_end = min(int(start) + warmup, int(end))
        if warmup_end > int(start):
            ax.axvspan(float(start) * dt, float(warmup_end) * dt, color='gray', alpha=0.08)

    tick_count = min(12, n_neurons)
    if tick_count > 0:
        tick_idx = np.unique(np.round(np.linspace(0, n_neurons - 1, num=tick_count)).astype(int))
        ax.set_yticks(2 * tick_idx + 0.5)
        ax.set_yticklabels([str(int(neuron_ids[idx])) for idx in tick_idx])

    metric_lines = [
        f'Spike threshold={spike_threshold:.4f} ({spike_threshold_source})',
        f'Surrogate FDR target={threshold_info.get("surrogate_fdr_target", None):.4f}'
        if threshold_info.get('surrogate_fdr_target', None) is not None else 'Surrogate FDR target=NA',
        f'Surrogate models={int(threshold_info.get("surrogate_n_models", 0))}'
        if threshold_info.get('surrogate_n_models', None) is not None else 'Surrogate models=NA',
        f'Onset tolerance=+/- {onset_tolerance_ms:.1f} ms',
        f'Tolerant precision={tolerant_metrics["tolerant_precision"]:.4f}'
        if tolerant_metrics['tolerant_precision'] is not None else 'Tolerant precision=NA',
        f'Tolerant recall={tolerant_metrics["tolerant_recall"]:.4f}'
        if tolerant_metrics['tolerant_recall'] is not None else 'Tolerant recall=NA',
        f'Tolerant F1={tolerant_metrics["tolerant_f1"]:.4f}'
        if tolerant_metrics['tolerant_f1'] is not None else 'Tolerant F1=NA',
    ]
    ax.text(
        0.995,
        0.99,
        '\n'.join(metric_lines),
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.78},
    )

    ax.set_xlabel('Held-out validation time (ms, concatenated recordings)')
    ax.set_ylabel('Neuron id (actual row above predicted row)')
    ax.set_title(
        'Global spike-separation raster over held-out validation time: paired rows per neuron '
        '(actual spikes above surrogate-thresholded predicted calls)'
    )
    ax.set_ylim(2 * n_neurons - 0.5, -0.5)
    ax.grid(False)
    ax.legend(loc='upper left', fontsize=8)

    fig.suptitle(
        f'Validation spike raster: {checkpoint.get("output_name", Path(output_path).stem)}\n'
        f'reconstructed={context["validation_strategy"]} | saved={context["saved_validation_strategy"]}',
        fontsize=13,
        fontweight='bold',
    )
    fig.subplots_adjust(top=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_neuron_raster_matrix(predicted_spike_calls, post_spikes, neuron_ids, is_positive, window_quality):
    positive_indices = np.where(is_positive == 1)[0]
    if positive_indices.size == 0:
        return {
            'stacked_rows': np.zeros((0, post_spikes.shape[1]), dtype=np.float32),
            'neuron_ids': np.array([], dtype=np.int32),
            'neuron_quality': np.array([], dtype=np.float32),
        }

    unique_neurons = np.unique(neuron_ids[positive_indices])
    actual_rows = []
    predicted_rows = []
    quality_rows = []
    neuron_row_ids = []

    for neuron_id in unique_neurons:
        neuron_window_idx = positive_indices[neuron_ids[positive_indices] == neuron_id]
        if neuron_window_idx.size == 0:
            continue
        actual_rows.append(post_spikes[neuron_window_idx].mean(axis=0).astype(np.float32))
        predicted_rows.append(predicted_spike_calls[neuron_window_idx].mean(axis=0).astype(np.float32))
        quality_rows.append(float(np.nanmean(window_quality[neuron_window_idx])))
        neuron_row_ids.append(int(neuron_id))

    if not actual_rows:
        return {
            'stacked_rows': np.zeros((0, post_spikes.shape[1]), dtype=np.float32),
            'neuron_ids': np.array([], dtype=np.int32),
            'neuron_quality': np.array([], dtype=np.float32),
        }

    actual_rows = np.asarray(actual_rows, dtype=np.float32)
    predicted_rows = np.asarray(predicted_rows, dtype=np.float32)
    quality_rows = np.asarray(quality_rows, dtype=np.float32)
    neuron_row_ids = np.asarray(neuron_row_ids, dtype=np.int32)
    sort_order = np.argsort(-np.nan_to_num(quality_rows, nan=-1e9))

    actual_rows = actual_rows[sort_order]
    predicted_rows = predicted_rows[sort_order]
    quality_rows = quality_rows[sort_order]
    neuron_row_ids = neuron_row_ids[sort_order]

    stacked_rows = np.zeros((actual_rows.shape[0] * 2, actual_rows.shape[1]), dtype=np.float32)
    stacked_rows[0::2] = actual_rows
    stacked_rows[1::2] = predicted_rows
    return {
        'stacked_rows': stacked_rows,
        'neuron_ids': neuron_row_ids,
        'neuron_quality': quality_rows,
    }


def build_positive_window_heatmap(predicted_spike_prob, post_spikes, is_positive, window_quality,
                                  max_rows=None):
    positive_indices = np.where(is_positive == 1)[0]
    if positive_indices.size == 0:
        return {
            'sorted_indices': np.array([], dtype=np.int32),
            'predicted_spike_prob': np.zeros((0, predicted_spike_prob.shape[1]), dtype=np.float32),
            'post_spikes': np.zeros((0, post_spikes.shape[1]), dtype=np.float32),
            'window_quality': np.array([], dtype=np.float32),
        }

    sort_scores = np.nan_to_num(window_quality[positive_indices], nan=-1e9)
    sorted_indices = positive_indices[np.argsort(-sort_scores)]
    if max_rows is not None and int(max_rows) > 0 and sorted_indices.size > int(max_rows):
        positions = np.linspace(0, sorted_indices.size - 1, num=int(max_rows)).round().astype(int)
        sorted_indices = sorted_indices[np.unique(positions)]

    return {
        'sorted_indices': sorted_indices.astype(np.int32),
        'predicted_spike_prob': predicted_spike_prob[sorted_indices],
        'post_spikes': post_spikes[sorted_indices],
        'window_quality': window_quality[sorted_indices],
    }


def reconstruct_validation_context(args, checkpoint):
    defaults = {
        'dt': 1.0,
        'pre_context': 50,
        'post_context': 10,
        'warmup': 30,
        'neg_ratio': 1.0,
        'neg_min_distance': 100,
        'val_fraction': 0.2,
        'mask_pre_ms': 1.0,
        'mask_post_ms': 2.0,
        'peak_threshold_mv': 15.0,
        'rng_seed': 42,
    }

    window_config = checkpoint.get('window_config', {})
    voltage_cleaning = checkpoint.get('voltage_cleaning', {})
    data_config = checkpoint.get('data_config', {})

    session_dir = infer_session_dir(args.session, checkpoint)
    use_all_recordings = resolve_arg(
        args.use_all_recordings,
        data_config.get('use_all_recordings', len(checkpoint.get('recording_summaries', [])) != 1),
    )
    recording_idx = int(resolve_arg(
        args.recording_idx,
        data_config.get('recording_idx', parse_single_recording_index(checkpoint)),
    ))
    dt = float(resolve_arg(args.dt, checkpoint.get('dt', defaults['dt'])))
    pre_context = int(resolve_arg(args.pre_context, window_config.get('pre_context', defaults['pre_context'])))
    post_context = int(resolve_arg(args.post_context, window_config.get('post_context', defaults['post_context'])))
    warmup = int(resolve_arg(args.warmup, window_config.get('warmup', defaults['warmup'])))
    neg_ratio = float(resolve_arg(args.neg_ratio, window_config.get('neg_ratio', defaults['neg_ratio'])))
    neg_min_distance = int(resolve_arg(
        args.neg_min_dist,
        window_config.get('neg_min_distance', defaults['neg_min_distance']),
    ))
    val_fraction = float(resolve_arg(
        args.val_fraction,
        window_config.get('val_fraction', defaults['val_fraction']),
    ))
    rng_seed = int(resolve_arg(args.rng_seed, window_config.get('rng_seed', defaults['rng_seed'])))

    mask_pre_ms = float(resolve_arg(
        args.mask_pre_ms,
        voltage_cleaning.get('mask_pre_ms', defaults['mask_pre_ms']),
    ))
    mask_post_ms = float(resolve_arg(
        args.mask_post_ms,
        voltage_cleaning.get('mask_post_ms', defaults['mask_post_ms']),
    ))
    peak_threshold_mv = float(resolve_arg(
        args.peak_threshold_mv,
        voltage_cleaning.get('peak_threshold_mv', defaults['peak_threshold_mv']),
    ))

    if use_all_recordings:
        loaded = load_all_recordings_with_voltage(
            str(session_dir),
            dt=dt,
            mask_pre_ms=mask_pre_ms,
            mask_post_ms=mask_post_ms,
            peak_threshold_mv=peak_threshold_mv,
        )
    else:
        loaded = load_single_recording_with_voltage(
            str(session_dir),
            recording_idx=recording_idx,
            dt=dt,
            mask_pre_ms=mask_pre_ms,
            mask_post_ms=mask_post_ms,
            peak_threshold_mv=peak_threshold_mv,
        )

    limit = int(resolve_arg(args.subsample_t, checkpoint.get('T', loaded['spike_matrix'].shape[1])))
    limit = min(limit, int(loaded['spike_matrix'].shape[1]))

    spike_matrix = loaded['spike_matrix'][:, :limit]
    voltage_matrix = loaded['voltage_matrix'][:, :limit]
    voltage_mask = loaded['voltage_mask'][:, :limit]
    boundaries = clip_boundaries(loaded['boundaries'], limit)
    _, validation_boundaries = split_recording_boundaries(boundaries, val_fraction=val_fraction)
    if validation_boundaries is None:
        validation_boundaries = boundaries
    excluded_bins = clip_excluded_bins(checkpoint.get('excluded_bins'), limit)
    neighbor_indices = [np.asarray(indices, dtype=np.int64) for indices in checkpoint['neighbor_indices']]
    neuron_ids = np.arange(int(checkpoint['n_neurons']))

    _, val_ds, validation_strategy = build_train_val_voltage_datasets(
        spike_matrix,
        voltage_matrix,
        voltage_mask,
        neighbor_indices,
        neuron_ids,
        pre_context=pre_context,
        post_context=post_context,
        warmup=warmup,
        neg_ratio=neg_ratio,
        neg_min_distance=neg_min_distance,
        boundaries=boundaries,
        excluded_bins=excluded_bins,
        val_fraction=val_fraction,
        rng_seed=rng_seed,
    )

    if len(val_ds) == 0:
        raise RuntimeError('Validation dataset is empty after reconstruction')

    return {
        'session_dir': str(session_dir),
        'use_all_recordings': bool(use_all_recordings),
        'recording_idx': recording_idx,
        'dt': dt,
        'pre_context': pre_context,
        'post_context': post_context,
        'warmup': warmup,
        'neg_ratio': neg_ratio,
        'neg_min_distance': neg_min_distance,
        'val_fraction': val_fraction,
        'rng_seed': rng_seed,
        'mask_pre_ms': mask_pre_ms,
        'mask_post_ms': mask_post_ms,
        'peak_threshold_mv': peak_threshold_mv,
        'limit_T': limit,
        'validation_strategy': validation_strategy,
        'saved_validation_strategy': checkpoint.get('validation_strategy'),
        'spike_matrix': spike_matrix,
        'boundaries': np.asarray(boundaries, dtype=np.int32),
        'validation_boundaries': np.asarray(validation_boundaries, dtype=np.int32),
        'excluded_bins': excluded_bins,
        'neighbor_indices': neighbor_indices,
        'neuron_ids': neuron_ids,
        'connectivity_thresholding': checkpoint.get('connectivity_thresholding', {}),
        'val_dataset': val_ds,
    }


def collect_validation_predictions(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    predicted_spike_prob = []
    predicted_voltage = []
    target_voltage = []
    target_mask = []
    post_spikes = []
    neuron_ids = []
    is_positive = []

    with torch.no_grad():
        for pre_sp, post_sp, post_v, post_vm, batch_neuron_ids, batch_is_pos in loader:
            pre_sp = pre_sp.to(device)
            post_sp_device = post_sp.to(device)
            batch_neuron_ids_device = batch_neuron_ids.to(device)
            window_len = pre_sp.shape[2]

            predicted_sp, predicted_v, _ = model(
                pre_sp,
                post_sp_device,
                batch_neuron_ids_device,
                tbptt_len=window_len,
            )

            predicted_spike_prob.append(predicted_sp.cpu().numpy().astype(np.float32))
            predicted_voltage.append(predicted_v.cpu().numpy().astype(np.float32))
            target_voltage.append(post_v.numpy().astype(np.float32))
            target_mask.append(post_vm.numpy().astype(np.float32))
            post_spikes.append(post_sp.numpy().astype(np.float32))
            neuron_ids.append(batch_neuron_ids.numpy().astype(np.int32))
            is_positive.append(batch_is_pos.numpy().astype(np.int32))

    return {
        'predicted_spike_prob': np.concatenate(predicted_spike_prob, axis=0),
        'predicted_voltage': np.concatenate(predicted_voltage, axis=0),
        'target_voltage': np.concatenate(target_voltage, axis=0),
        'target_mask': np.concatenate(target_mask, axis=0),
        'post_spikes': np.concatenate(post_spikes, axis=0),
        'neuron_ids': np.concatenate(neuron_ids, axis=0),
        'is_positive': np.concatenate(is_positive, axis=0),
    }


def select_example_indices(neuron_ids, is_positive, supervised_counts, max_windows, include_negative,
                           window_quality):
    if include_negative:
        candidate_indices = np.where(supervised_counts > 0)[0]
    else:
        candidate_indices = np.where(
            (is_positive == 1) & (supervised_counts > 0) & np.isfinite(window_quality)
        )[0]
        if candidate_indices.size == 0:
            candidate_indices = np.where((is_positive == 1) & (supervised_counts > 0))[0]
        if candidate_indices.size == 0:
            candidate_indices = np.where(supervised_counts > 0)[0]

    if candidate_indices.size == 0:
        return []

    order = candidate_indices[np.argsort(-np.nan_to_num(window_quality[candidate_indices], nan=-1e9))]
    n_select = min(int(max_windows), int(order.size))
    if n_select <= 0:
        return []

    sample_positions = np.unique(np.round(np.linspace(0, order.size - 1, num=n_select)).astype(int))
    selected = [int(order[pos]) for pos in sample_positions]
    if len(selected) < n_select:
        for idx in order:
            idx = int(idx)
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= n_select:
                break
    return selected


def plot_validation_summary(output_path, checkpoint, context, predictions, max_windows, include_negative,
                            spike_threshold=None, spike_threshold_source=None,
                            spike_threshold_info=None, max_heatmap_rows=None):
    warmup = int(context['warmup'])
    predicted_spike_prob = predictions['predicted_spike_prob']
    predicted_voltage = predictions['predicted_voltage']
    target_voltage = predictions['target_voltage']
    target_mask = predictions['target_mask']
    post_spikes = predictions['post_spikes']
    neuron_ids = predictions['neuron_ids']
    is_positive = predictions['is_positive']

    supervised_mask = target_mask > 0.5
    supervised_mask[:, :warmup] = False
    supervised_counts = supervised_mask.sum(axis=1)
    per_window_mae = np.full(predicted_voltage.shape[0], np.nan, dtype=np.float32)
    valid_window_mask = supervised_counts > 0
    if np.any(valid_window_mask):
        diffs = np.abs(predicted_voltage - target_voltage)
        per_window_mae[valid_window_mask] = (
            (diffs * supervised_mask).sum(axis=1)[valid_window_mask] / supervised_counts[valid_window_mask]
        ).astype(np.float32)

    if spike_threshold is None or spike_threshold_source is None:
        spike_threshold, threshold_source = select_spike_threshold(
            predicted_spike_prob,
            post_spikes,
            warmup,
            spike_threshold=spike_threshold,
        )
    else:
        threshold_source = str(spike_threshold_source)
    if spike_threshold_info is None:
        spike_threshold_info = {
            'mode': threshold_source,
            'threshold': float(spike_threshold),
        }
    predicted_spike_calls = (predicted_spike_prob >= float(spike_threshold)).astype(np.float32)
    metrics = compute_metrics(predicted_voltage, target_voltage, supervised_mask)
    spike_metrics = compute_spike_metrics(
        predicted_spike_prob,
        post_spikes,
        warmup,
        spike_threshold=spike_threshold,
    )
    spike_metrics['threshold_source'] = threshold_source
    spike_metrics['threshold_info'] = spike_threshold_info
    spike_quality = compute_window_spike_quality(predicted_spike_prob, post_spikes, warmup)
    window_spike_quality = spike_quality['quality']

    positive_mask = (is_positive == 1)[:, None] & supervised_mask
    mean_pred_positive, positive_counts = masked_mean(predicted_voltage, positive_mask.astype(np.float32))
    mean_target_positive, _ = masked_mean(target_voltage, positive_mask.astype(np.float32))
    positive_windows = is_positive == 1
    mean_pred_spike = np.full(predicted_spike_prob.shape[1], np.nan, dtype=np.float32)
    mean_actual_spike = np.full(post_spikes.shape[1], np.nan, dtype=np.float32)
    mean_predicted_calls = np.full(predicted_spike_calls.shape[1], np.nan, dtype=np.float32)
    if np.any(positive_windows):
        mean_pred_spike = predicted_spike_prob[positive_windows].mean(axis=0).astype(np.float32)
        mean_actual_spike = post_spikes[positive_windows].mean(axis=0).astype(np.float32)
        mean_predicted_calls = predicted_spike_calls[positive_windows].mean(axis=0).astype(np.float32)
    window_t = np.arange(predicted_voltage.shape[1]) * float(context['dt']) - (
        warmup + int(context['pre_context'])
    ) * float(context['dt'])

    neuron_raster = build_neuron_raster_matrix(
        predicted_spike_calls,
        post_spikes,
        neuron_ids,
        is_positive,
        window_spike_quality,
    )
    heatmap_data = build_positive_window_heatmap(
        predicted_spike_prob,
        post_spikes,
        is_positive,
        window_spike_quality,
        max_rows=max_heatmap_rows,
    )

    selected_indices = select_example_indices(
        neuron_ids,
        is_positive,
        supervised_counts,
        max_windows=max_windows,
        include_negative=include_negative,
        window_quality=window_spike_quality,
    )
    n_examples = len(selected_indices)
    n_example_rows = max(1, int(np.ceil(max(n_examples, 1) / 2.0)))

    fig = plt.figure(figsize=(16, 13 + 3.6 * n_example_rows))
    grid = fig.add_gridspec(
        3 + n_example_rows,
        2,
        height_ratios=[3.3, 1.8, 4.8] + [2.3] * n_example_rows,
        hspace=0.35,
        wspace=0.25,
    )

    ax_raster = fig.add_subplot(grid[0, :])
    raster_rows = neuron_raster['stacked_rows']
    if raster_rows.size > 0:
        raster_im = ax_raster.imshow(
            raster_rows,
            aspect='auto',
            interpolation='nearest',
            cmap='magma',
            vmin=0.0,
            vmax=1.0,
            extent=[
                window_t[0] - 0.5 * float(context['dt']),
                window_t[-1] + 0.5 * float(context['dt']),
                raster_rows.shape[0] - 0.5,
                -0.5,
            ],
        )
        tick_count = min(10, neuron_raster['neuron_ids'].size)
        if tick_count > 0:
            tick_idx = np.unique(np.round(
                np.linspace(0, neuron_raster['neuron_ids'].size - 1, num=tick_count)
            ).astype(int))
            ax_raster.set_yticks(2 * tick_idx + 0.5)
            ax_raster.set_yticklabels([str(int(neuron_raster['neuron_ids'][idx])) for idx in tick_idx])
        cbar = fig.colorbar(raster_im, ax=ax_raster, fraction=0.02, pad=0.01)
        cbar.set_label('Fraction of positive windows with spike/call')
    ax_raster.axvspan(
        window_t[0] - 0.5 * float(context['dt']),
        window_t[warmup - 1] + 0.5 * float(context['dt']),
        color='white',
        alpha=0.06,
    )
    ax_raster.axvline(0.0, color='tab:cyan', ls='--', lw=1.0)
    ax_raster.set_xlabel('Time relative to selected postsynaptic spike (ms)')
    ax_raster.set_ylabel('Neuron id (pair rows)')
    ax_raster.set_title(
        'Positive-window averaged spike profile by neuron: alternating rows are mean actual '
        'spike rate and mean thresholded predicted spike-call rate'
    )
    ax_raster.text(
        0.995,
        0.99,
        f'Predicted-call threshold={spike_threshold:.4f} ({threshold_source})\n'
        'Within each neuron pair: top row=actual, bottom row=predicted',
        transform=ax_raster.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75},
    )

    ax_mean = fig.add_subplot(grid[1, :])
    ax_mean.axvspan(
        window_t[0] - 0.5 * float(context['dt']),
        window_t[warmup - 1] + 0.5 * float(context['dt']),
        color='gray',
        alpha=0.12,
        label='warmup',
    )
    if np.any(positive_windows):
        ax_mean.plot(
            window_t,
            mean_pred_spike,
            color='tab:purple',
            lw=1.6,
            label='mean predicted spike probability',
        )
        ax_mean.plot(
            window_t,
            mean_predicted_calls,
            color='black',
            lw=1.2,
            ls='--',
            label='mean thresholded predicted spike calls',
        )
        ax_mean.step(
            window_t,
            mean_actual_spike,
            color='tab:orange',
            where='mid',
            lw=1.4,
            label='mean actual postsynaptic spike target',
        )
    metric_lines = [
        f"Spike AUC={spike_metrics['auc']:.4f}" if spike_metrics['auc'] is not None else 'Spike AUC=NA',
        f"Spike AP={spike_metrics['ap']:.4f}" if spike_metrics['ap'] is not None else 'Spike AP=NA',
        f"F1@thr={spike_metrics['f1_at_threshold']:.4f}" if spike_metrics['f1_at_threshold'] is not None else 'F1@thr=NA',
        f"Brier={spike_metrics['brier']:.4f}" if spike_metrics['brier'] is not None else 'Brier=NA',
        f"p(spike) on actual spike bins={spike_metrics['mean_pred_on_spikes']:.4f}" if spike_metrics['mean_pred_on_spikes'] is not None else 'p(spike) on actual spike bins=NA',
        f"p(spike) on non-spike bins={spike_metrics['mean_pred_on_nonspikes']:.4f}" if spike_metrics['mean_pred_on_nonspikes'] is not None else 'p(spike) on non-spike bins=NA',
    ]
    ax_mean.text(
        0.98,
        0.98,
        '\n'.join(metric_lines),
        transform=ax_mean.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75},
    )
    ax_mean.axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    ax_mean.set_xlabel('Time relative to selected postsynaptic spike (ms)')
    ax_mean.set_ylabel('Spike probability / rate')
    ax_mean.set_ylim(-0.05, 1.05)
    ax_mean.set_title('Event-aligned average view for positive validation windows')
    ax_mean.grid(True, alpha=0.2)
    ax_mean.legend(loc='upper left', fontsize=8)

    ax_heatmap = fig.add_subplot(grid[2, :])
    heatmap_prob = heatmap_data['predicted_spike_prob']
    if heatmap_prob.size > 0:
        heatmap_im = ax_heatmap.imshow(
            heatmap_prob,
            aspect='auto',
            interpolation='nearest',
            cmap='viridis',
            vmin=0.0,
            vmax=1.0,
            extent=[
                window_t[0] - 0.5 * float(context['dt']),
                window_t[-1] + 0.5 * float(context['dt']),
                heatmap_prob.shape[0] - 0.5,
                -0.5,
            ],
        )
        spike_y, spike_x = np.where(heatmap_data['post_spikes'] > 0.5)
        if spike_x.size > 0:
            ax_heatmap.scatter(
                window_t[spike_x],
                spike_y,
                s=2,
                c='white',
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )
        cbar = fig.colorbar(heatmap_im, ax=ax_heatmap, fraction=0.02, pad=0.01)
        cbar.set_label('Predicted spike probability')
        ax_heatmap.set_yticks([0, max(heatmap_prob.shape[0] - 1, 0)])
        ax_heatmap.set_yticklabels(['best quality', 'worst quality'])
    ax_mean.axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    ax_heatmap.axvspan(
        window_t[0] - 0.5 * float(context['dt']),
        window_t[warmup - 1] + 0.5 * float(context['dt']),
        color='gray',
        alpha=0.12,
    )
    ax_heatmap.axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    ax_heatmap.set_xlabel('Time relative to selected postsynaptic spike (ms)')
    ax_heatmap.set_ylabel('Positive validation windows')
    ax_heatmap.set_title(
        'Positive-window spike-probability heatmap sorted by spike-prediction quality '
        '(white dots mark actual spikes)'
    )

    for plot_idx, window_idx in enumerate(selected_indices):
        row = 3 + plot_idx // 2
        col = plot_idx % 2
        panel_grid = grid[row, col].subgridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.05)
        ax_spike_window = fig.add_subplot(panel_grid[0, 0])
        ax = fig.add_subplot(panel_grid[1, 0], sharex=ax_spike_window)

        ax_spike_window.axvspan(
            window_t[0] - 0.5 * float(context['dt']),
            window_t[warmup - 1] + 0.5 * float(context['dt']),
            color='gray',
            alpha=0.12,
        )
        ax_spike_window.plot(
            window_t,
            predicted_spike_prob[window_idx],
            color='tab:purple',
            lw=1.1,
            label='predicted spike prob',
        )
        ax_spike_window.step(
            window_t,
            post_spikes[window_idx],
            where='mid',
            color='tab:orange',
            lw=1.0,
            alpha=0.9,
            label='actual spikes',
        )
        predicted_call_bins = predicted_spike_calls[window_idx] > 0.5
        if np.any(predicted_call_bins):
            ax_spike_window.scatter(
                window_t[predicted_call_bins],
                np.full(int(np.sum(predicted_call_bins)), 0.96, dtype=np.float32),
                color='black',
                marker='v',
                s=18,
                label='thresholded predicted calls',
            )
        ax_spike_window.axhline(spike_threshold, color='black', lw=0.9, ls='--', alpha=0.6)
        ax_spike_window.axvline(0.0, color='tab:blue', ls='--', lw=1.0)
        ax_spike_window.set_ylim(-0.05, 1.05)
        ax_spike_window.set_ylabel('Spike')
        ax_spike_window.tick_params(axis='x', labelbottom=False)

        ax.axvspan(
            window_t[0] - 0.5 * float(context['dt']),
            window_t[warmup - 1] + 0.5 * float(context['dt']),
            color='gray',
            alpha=0.12,
        )
        ax.plot(window_t, predicted_voltage[window_idx], color='black', lw=1.4, label='inferred')
        ax.plot(window_t, target_voltage[window_idx], color='lightgray', lw=1.0, label='stored target')
        if np.any(supervised_mask[window_idx]):
            ax.plot(
                window_t[supervised_mask[window_idx]],
                target_voltage[window_idx, supervised_mask[window_idx]],
                color='tab:blue',
                lw=1.6,
                label='actual supervised',
            )
        if np.any(~supervised_mask[window_idx]):
            ax.scatter(
                window_t[~supervised_mask[window_idx]],
                target_voltage[window_idx, ~supervised_mask[window_idx]],
                color='tab:red',
                marker='x',
                s=18,
                label='ignored bins',
            )
        spike_times = window_t[post_spikes[window_idx] > 0.5]
        for spike_time in spike_times:
            ax.axvline(spike_time, color='tab:orange', lw=1.0, alpha=0.2)
        ax.axvline(0.0, color='tab:blue', ls='--', lw=1.0)
        label_text = 'pos' if int(is_positive[window_idx]) == 1 else 'neg'
        mae_value = per_window_mae[window_idx]
        mae_text = 'NA' if not np.isfinite(mae_value) else f'{mae_value:.3f}'
        quality_value = window_spike_quality[window_idx]
        quality_text = 'NA' if not np.isfinite(quality_value) else f'{quality_value:.3f}'
        ax_spike_window.set_title(
            f'Window {window_idx} | neuron {int(neuron_ids[window_idx])} | {label_text} | '
            f'spike-quality={quality_text} | voltage-MAE={mae_text}',
            fontsize=9,
        )
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage')
        ax.grid(True, alpha=0.2)
        if plot_idx == 0:
            lines_top, labels_top = ax_spike_window.get_legend_handles_labels()
            lines_bottom, labels_bottom = ax.get_legend_handles_labels()
            ax.legend(lines_top + lines_bottom, labels_top + labels_bottom, loc='best', fontsize=8)

    fig.suptitle(
        f'Validation spike/voltage views: {checkpoint.get("output_name", Path(output_path).stem)}\n'
        f'reconstructed={context["validation_strategy"]} | saved={context["saved_validation_strategy"]}',
        fontsize=13,
        fontweight='bold',
    )
    fig.subplots_adjust(top=0.93)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'metrics': metrics,
        'spike_metrics': spike_metrics,
        'spike_threshold': float(spike_threshold),
        'spike_threshold_source': threshold_source,
        'selected_indices': [int(idx) for idx in selected_indices],
        'supervised_mask': supervised_mask,
        'supervised_counts': supervised_counts.astype(np.int32),
        'per_window_mae': per_window_mae,
        'predicted_spike_calls': predicted_spike_calls.astype(np.uint8),
        'window_spike_quality': window_spike_quality.astype(np.float32),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description='Plot inferred vs actual postsynaptic voltage on the held-out validation set '
        'for a saved voltage-augmented learned-LIF checkpoint.',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to voltage_augmented_learned_lif_*.pt checkpoint')
    parser.add_argument('--session', type=str, default=None,
                        help='Optional session directory override; defaults to LIF data/<session_name>')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for the figure, JSON summary, and NPZ export')
    parser.add_argument('--max-windows', type=int, default=6,
                        help='Maximum number of validation windows to plot as individual overlays')
    parser.add_argument('--eval-batch-size', type=int, default=128,
                        help='Batch size used when running the saved model on validation windows')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device used for forward passes')
    parser.add_argument('--include-negative-examples', action='store_true',
                        help='Allow negative validation windows in the example panels')
    parser.add_argument('--subsample-t', type=int, default=None,
                        help='Optional time-bin limit applied before rebuilding the validation split')
    parser.add_argument('--spike-threshold', type=float, default=None,
                        help='Optional threshold used to convert predicted spike probabilities into spike calls for raster and window overlays')
    parser.add_argument('--spike-threshold-mode', type=str, default='surrogate_fdr',
                        choices=['surrogate_fdr'],
                        help='Threshold rule used for inferred spike calls in the global raster')
    parser.add_argument('--spike-surrogate-fdr', type=float, default=None,
                        help='Target FDR used for label-free surrogate calibration of spike-call thresholding')
    parser.add_argument('--spike-threshold-surrogates', type=int, default=None,
                        help='Number of circular-shift surrogate score sets used to calibrate the spike-call threshold')
    parser.add_argument('--spike-threshold-surrogate-seed', type=int, default=None,
                        help='Random seed used for spike surrogate threshold calibration')
    parser.add_argument('--spike-threshold-min-shift-fraction', type=float, default=None,
                        help='Minimum within-recording circular shift fraction used for spike surrogate calibration')
    parser.add_argument('--spike-threshold-max-bins', type=int, default=250000,
                        help='Maximum number of observed or surrogate spike-probability bins used during surrogate-FDR threshold calibration')
    parser.add_argument('--timeline-batch-neurons', type=int, default=16,
                        help='Number of postsynaptic neurons processed per batch when building the continuous validation-time raster')
    parser.add_argument('--spike-onset-tolerance-ms', type=float, default=2.0,
                        help='Tolerance window around actual spike onset used when summarizing predicted-call timing in the global raster')
    parser.add_argument('--max-heatmap-rows', type=int, default=None,
                        help='Optional cap on the number of positive windows shown in the sorted spike-probability heatmap')

    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--all-recordings', dest='use_all_recordings', action='store_true',
                             help='Force all-recordings reconstruction')
    split_group.add_argument('--single-recording', dest='use_all_recordings', action='store_false',
                             help='Force single-recording reconstruction')
    parser.set_defaults(use_all_recordings=None)
    parser.add_argument('--recording-idx', type=int, default=None,
                        help='Recording index override for single-recording checkpoints')

    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--pre-context', type=int, default=None)
    parser.add_argument('--post-context', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--neg-ratio', type=float, default=None)
    parser.add_argument('--neg-min-dist', type=int, default=None)
    parser.add_argument('--val-fraction', type=float, default=None)
    parser.add_argument('--mask-pre-ms', type=float, default=None)
    parser.add_argument('--mask-post-ms', type=float, default=None)
    parser.add_argument('--peak-threshold-mv', type=float, default=None)
    parser.add_argument('--rng-seed', type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(str(checkpoint_path))

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
    if state_dict is None:
        raise KeyError('Checkpoint is missing model_state_dict/state_dict')

    threshold_mode = str(checkpoint.get('threshold_mode', 'adaptive'))
    model = VoltageAugmentedPerNeuronLIF(
        n_neurons=int(checkpoint['n_neurons']),
        K=int(checkpoint['K']),
        max_delay=int(checkpoint['max_delay']),
        threshold_mode=threshold_mode,
    )
    safe_load_state_dict(model, state_dict)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    context = reconstruct_validation_context(args, checkpoint)
    predictions = collect_validation_predictions(
        model,
        context['val_dataset'],
        batch_size=int(args.eval_batch_size),
        device=device,
    )

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    base_stem = checkpoint_path.stem
    threshold_config = context.get('connectivity_thresholding', {})

    spike_surrogate_fdr = float(resolve_arg(
        args.spike_surrogate_fdr,
        threshold_config.get('surrogate_fdr', 0.005),
    ))
    spike_threshold_surrogates = int(resolve_arg(
        args.spike_threshold_surrogates,
        threshold_config.get('n_threshold_surrogates', 4),
    ))
    spike_threshold_surrogate_seed = int(resolve_arg(
        args.spike_threshold_surrogate_seed,
        threshold_config.get('surrogate_seed', 1234),
    ))
    spike_threshold_min_shift_fraction = float(resolve_arg(
        args.spike_threshold_min_shift_fraction,
        threshold_config.get('surrogate_min_shift_fraction', 0.10),
    ))
    spike_threshold_max_bins = int(args.spike_threshold_max_bins)
    onset_tolerance_ms = float(args.spike_onset_tolerance_ms)
    onset_tolerance_bins = max(int(round(onset_tolerance_ms / float(context['dt']))), 0)

    timeline_predictions = collect_timeline_predictions(
        model,
        context['spike_matrix'],
        context['neighbor_indices'],
        context['neuron_ids'],
        context['validation_boundaries'],
        device=device,
        batch_neurons=int(args.timeline_batch_neurons),
    )
    timeline_eval_mask = build_segment_warmup_mask(
        timeline_predictions['predicted_spike_prob'].shape[1],
        timeline_predictions['boundaries'],
        context['warmup'],
    )
    surrogate_score_sets = None
    threshold_rng = np.random.default_rng(spike_threshold_surrogate_seed + 17)
    observed_spike_scores = subsample_threshold_scores(
        timeline_predictions['predicted_spike_prob'][:, timeline_eval_mask].reshape(-1),
        max_points=spike_threshold_max_bins,
        rng=threshold_rng,
    )
    if args.spike_threshold is None:
        surrogate_score_sets = estimate_spike_surrogate_score_sets(
            model,
            context['spike_matrix'],
            context['neighbor_indices'],
            context['neuron_ids'],
            context['validation_boundaries'],
            warmup=context['warmup'],
            device=device,
            batch_neurons=int(args.timeline_batch_neurons),
            n_surrogates=spike_threshold_surrogates,
            surrogate_seed=spike_threshold_surrogate_seed,
            min_shift_fraction=spike_threshold_min_shift_fraction,
            max_scores_per_set=spike_threshold_max_bins,
        )

    spike_threshold, spike_threshold_source, spike_threshold_info = select_spike_threshold_from_scores(
        observed_spike_scores,
        spike_threshold=args.spike_threshold,
        threshold_mode=args.spike_threshold_mode,
        surrogate_score_sets=surrogate_score_sets,
        surrogate_fdr=spike_surrogate_fdr,
    )
    timeline_predicted_calls = (
        timeline_predictions['predicted_spike_prob'] >= float(spike_threshold)
    ).astype(np.uint8)
    timeline_predicted_calls[:, ~timeline_eval_mask] = 0
    timeline_actual_eval = timeline_predictions['actual_spikes'].copy()
    timeline_actual_eval[:, ~timeline_eval_mask] = 0.0
    tolerant_metrics = compute_tolerant_spike_metrics(
        timeline_actual_eval,
        timeline_predicted_calls,
        tolerance_bins=onset_tolerance_bins,
    )
    timeline_predictions['predicted_spike_calls'] = timeline_predicted_calls
    timeline_predictions['matched_predicted_mask'] = tolerant_metrics.pop('matched_predicted_mask')
    timeline_predictions['unmatched_predicted_mask'] = tolerant_metrics.pop('unmatched_predicted_mask')
    timeline_predictions['tolerant_metrics'] = tolerant_metrics

    spike_raster_path = output_dir / f'{base_stem}_validation_spike_raster.png'
    plot_global_spike_raster(
        spike_raster_path,
        checkpoint,
        context,
        timeline_predictions,
        spike_threshold=spike_threshold,
        spike_threshold_source=spike_threshold_source,
        threshold_info=spike_threshold_info,
        onset_tolerance_ms=onset_tolerance_ms,
    )

    figure_path = output_dir / f'{base_stem}_validation_voltage.png'
    summary = plot_validation_summary(
        figure_path,
        checkpoint,
        context,
        predictions,
        max_windows=max(1, int(args.max_windows)),
        include_negative=bool(args.include_negative_examples),
        spike_threshold=spike_threshold,
        spike_threshold_source=spike_threshold_source,
        spike_threshold_info=spike_threshold_info,
        max_heatmap_rows=args.max_heatmap_rows,
    )

    npz_path = output_dir / f'{base_stem}_validation_voltage_predictions.npz'
    np.savez_compressed(
        npz_path,
        predicted_spike_prob=predictions['predicted_spike_prob'],
        predicted_spike_calls=summary['predicted_spike_calls'],
        predicted_voltage=predictions['predicted_voltage'],
        target_voltage=predictions['target_voltage'],
        target_mask=predictions['target_mask'],
        post_spikes=predictions['post_spikes'],
        neuron_ids=predictions['neuron_ids'],
        is_positive=predictions['is_positive'],
        supervised_mask=summary['supervised_mask'].astype(np.uint8),
        supervised_counts=summary['supervised_counts'],
        per_window_mae=summary['per_window_mae'],
        window_spike_quality=summary['window_spike_quality'],
    )

    json_path = output_dir / f'{base_stem}_validation_voltage_summary.json'
    json_payload = {
        'checkpoint': str(checkpoint_path),
        'session_dir': context['session_dir'],
        'validation_strategy_saved': context['saved_validation_strategy'],
        'validation_strategy_reconstructed': context['validation_strategy'],
        'use_all_recordings': context['use_all_recordings'],
        'recording_idx': context['recording_idx'],
        'dt': context['dt'],
        'pre_context': context['pre_context'],
        'post_context': context['post_context'],
        'warmup': context['warmup'],
        'neg_ratio': context['neg_ratio'],
        'neg_min_distance': context['neg_min_distance'],
        'val_fraction': context['val_fraction'],
        'mask_pre_ms': context['mask_pre_ms'],
        'mask_post_ms': context['mask_post_ms'],
        'peak_threshold_mv': context['peak_threshold_mv'],
        'limit_T': context['limit_T'],
        'n_val_windows': int(len(context['val_dataset'])),
        'n_positive_val_windows': int(np.sum(predictions['is_positive'] == 1)),
        'n_negative_val_windows': int(np.sum(predictions['is_positive'] == 0)),
        'voltage_metrics': summary['metrics'],
        'spike_metrics': summary['spike_metrics'],
        'spike_threshold': summary['spike_threshold'],
        'spike_threshold_source': summary['spike_threshold_source'],
        'spike_threshold_mode': args.spike_threshold_mode,
        'spike_surrogate_fdr_target': spike_surrogate_fdr,
        'spike_threshold_n_surrogates': spike_threshold_surrogates,
        'spike_threshold_surrogate_seed': spike_threshold_surrogate_seed,
        'spike_threshold_min_shift_fraction': spike_threshold_min_shift_fraction,
        'spike_threshold_max_bins': spike_threshold_max_bins,
        'spike_onset_tolerance_ms': onset_tolerance_ms,
        'timeline_spike_metrics': timeline_predictions['tolerant_metrics'],
        'selected_example_indices': summary['selected_indices'],
        'figure_path': str(figure_path),
        'spike_raster_path': str(spike_raster_path),
        'npz_path': str(npz_path),
    }
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(json_payload, handle, indent=2)

    print(json.dumps(json_payload, indent=2))


if __name__ == '__main__':
    main()