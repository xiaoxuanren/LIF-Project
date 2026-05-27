import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from lif_inference.burst_exclusion import combine_excluded_bins, detect_network_burst_windows
from lif_inference.event_windows import find_event_windows
from lif_inference.shared_data import spike_times_to_binary
from lif_inference.voltage_augmented_learned_lif_connectivity import (
    VoltageAugmentedPerNeuronLIF,
    downsample_processed_voltage,
    preprocess_voltage_recording,
    resolve_voltage_dt_factor,
)
from lif_simulation.voltage_storage import resolve_recording_voltage


SESSION_DIR = r"LIF data/20260523_084407"
SESSION_NAME = os.path.basename(SESSION_DIR.rstrip("\\/"))
PRIMARY_RECORDING_IDX = 0
DT_MS = 1.0
MASK_PRE_MS = 1.0
MASK_POST_MS = 2.0
PEAK_THRESHOLD_MV = 15.0
WARMUP = 30
PRE_CONTEXT = 50
POST_CONTEXT = 10
NEG_RATIO = 1.0
NEG_MIN_DIST = 100
OUTPUT_DIR = "voltage_augmented_learned_lif_outputs"
CHECKPOINT_NAME = (
    "voltage_augmented_learned_lif_"
    "20260523_084407_cmp1200s_voltage_postcentered_k100_exclbursts_allrec.pt"
)


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


def infer_threshold_mode(state_dict):
    if any(str(key).endswith('shared_threshold') for key in state_dict.keys()):
        return 'shared'
    return 'adaptive'


def compute_excluded_bins(spike_matrix, burst_onset_times, dt_ms):
    boundaries = [0, spike_matrix.shape[1]]

    burst_onset_bins = np.array([], dtype=np.int32)
    if burst_onset_times.size > 0:
        burst_onset_bins = np.unique((burst_onset_times / dt_ms).astype(np.int32))
        burst_onset_bins = burst_onset_bins[burst_onset_bins < spike_matrix.shape[1]]

    burst_info = detect_network_burst_windows(
        spike_matrix,
        boundaries,
        dt_ms=dt_ms,
        activity_bin_ms=100.0,
        smooth_bins=3,
        threshold_std=3.0,
        min_active_fraction=0.10,
        min_burst_duration_ms=100.0,
        merge_gap_ms=150.0,
        pad_before_ms=100.0,
        pad_after_ms=250.0,
    )
    excluded_bins = combine_excluded_bins(burst_onset_bins, burst_info['excluded_bins'])
    return boundaries, excluded_bins


def find_positive_window(spike_matrix, spike_times, boundaries, excluded_bins):
    for neuron_id in range(spike_matrix.shape[0]):
        pos_windows, _ = find_event_windows(
            spike_matrix,
            neuron_id,
            pre_context=PRE_CONTEXT,
            post_context=POST_CONTEXT,
            warmup=WARMUP,
            neg_ratio=NEG_RATIO,
            neg_min_distance=NEG_MIN_DIST,
            boundaries=boundaries,
            excluded_bins=excluded_bins,
            rng=np.random.RandomState(42 + int(neuron_id)),
        )
        if not pos_windows:
            continue

        start, end = pos_windows[0]
        spike_bin_dt = start + WARMUP + PRE_CONTEXT
        neuron_spike_times = np.asarray(spike_times[neuron_id], dtype=float)
        candidates = neuron_spike_times[
            (neuron_spike_times >= spike_bin_dt * DT_MS)
            & (neuron_spike_times < (spike_bin_dt + 1) * DT_MS)
        ]
        if candidates.size == 0:
            continue

        return {
            'neuron_id': int(neuron_id),
            'start': int(start),
            'end': int(end),
            'spike_bin_dt': int(spike_bin_dt),
            'spike_time_ms': float(candidates[0]),
        }
    return None


def load_recording_example(recording_idx):
    rec_path = os.path.join(SESSION_DIR, f'recording{recording_idx:03d}.npz')
    with np.load(rec_path, allow_pickle=True) as rec_data:
        duration = float(rec_data['duration'])
        spike_times = [np.asarray(item, dtype=float) for item in rec_data['spike_times']]
        spike_matrix = spike_times_to_binary(spike_times, duration, DT_MS)

        burst_onset_times = np.array([], dtype=float)
        if 'burst_onset_times' in rec_data.files:
            burst_onset_times = np.asarray(rec_data['burst_onset_times'], dtype=float)
            burst_onset_times = burst_onset_times[
                (burst_onset_times >= 0.0) & (burst_onset_times < duration)
            ]

        boundaries, excluded_bins = compute_excluded_bins(
            spike_matrix,
            burst_onset_times,
            DT_MS,
        )
        chosen = find_positive_window(
            spike_matrix,
            spike_times,
            boundaries,
            excluded_bins,
        )
        if chosen is None:
            raise RuntimeError(
                f'Could not find a positive post-centered window in recording {recording_idx:03d}'
            )

        voltage_bundle = resolve_recording_voltage(
            rec_data,
            recording_path=rec_path,
            load_into_memory=False,
        )
        sample_rate_ms = float(voltage_bundle['sample_rate'])
        raw_trace = np.asarray(
            voltage_bundle['traces'][chosen['neuron_id']:chosen['neuron_id'] + 1],
            dtype=np.float32,
        )

    processed_native = preprocess_voltage_recording(
        raw_trace,
        [spike_times[chosen['neuron_id']]],
        sample_rate_ms,
        mask_pre_ms=MASK_PRE_MS,
        mask_post_ms=MASK_POST_MS,
        peak_threshold_mv=PEAK_THRESHOLD_MV,
    )
    voltage_dt_factor = resolve_voltage_dt_factor(sample_rate_ms, DT_MS, rec_path)
    processed_coarse = downsample_processed_voltage(processed_native, voltage_dt_factor)

    chosen.update({
        'recording_idx': int(recording_idx),
        'recording_path': rec_path,
        'sample_rate_ms': sample_rate_ms,
        'voltage_dt_factor': int(voltage_dt_factor),
        'burst_excluded_bins': int(len(excluded_bins)),
        'raw_trace': raw_trace[0],
        'native_mask': processed_native['valid_mask'][0],
        'coarse_voltage': processed_coarse['normalized_voltage'][0],
        'coarse_mask': processed_coarse['valid_mask'][0],
        'spike_matrix': spike_matrix,
    })
    return chosen


def load_alternate_example(primary_recording_idx):
    rec_paths = sorted(glob.glob(os.path.join(SESSION_DIR, 'recording[0-9][0-9][0-9].npz')))
    for rec_path in rec_paths:
        name = os.path.splitext(os.path.basename(rec_path))[0]
        recording_idx = int(name.replace('recording', ''))
        if recording_idx == primary_recording_idx:
            continue
        try:
            return load_recording_example(recording_idx)
        except RuntimeError:
            continue
    raise RuntimeError('Could not find a second recording with a positive visualization window')


def render_mask_figures(example):
    raw_row = example['raw_trace']
    native_mask = example['native_mask']
    coarse_voltage = example['coarse_voltage']
    coarse_mask = example['coarse_mask']
    spike_matrix = example['spike_matrix']
    spike_time_ms = example['spike_time_ms']
    sample_rate_ms = example['sample_rate_ms']
    spike_bin_dt = example['spike_bin_dt']
    recording_idx = example['recording_idx']
    neuron_id = example['neuron_id']
    factor = example['voltage_dt_factor']

    spike_sample_native = int(np.floor(spike_time_ms / sample_rate_ms + 1e-9))

    native_radius_ms = 6.0
    native_radius_bins = int(round(native_radius_ms / sample_rate_ms))
    ns = max(0, spike_sample_native - native_radius_bins)
    ne = min(raw_row.shape[0], spike_sample_native + native_radius_bins + 1)
    native_indices = np.arange(ns, ne)
    t_native = native_indices * sample_rate_ms - spike_time_ms
    raw_native = raw_row[ns:ne]
    mask_native = native_mask[ns:ne]

    coarse_radius_bins = 6
    cs = max(0, spike_bin_dt - coarse_radius_bins)
    ce = min(coarse_mask.shape[0], spike_bin_dt + coarse_radius_bins + 1)
    coarse_bins = np.arange(cs, ce)
    t_coarse = (coarse_bins + 0.5) * DT_MS - spike_time_ms
    mask_coarse = coarse_mask[cs:ce]

    sample_indices = np.arange(max(0, cs * factor), min(raw_row.shape[0], ce * factor))
    sample_times = sample_indices * sample_rate_ms - spike_time_ms
    sample_valid = native_mask[sample_indices] > 0.5
    sample_slot_y = ((sample_indices % factor) + 0.5) / float(factor)

    start = example['start']
    end = example['end']
    window_len = end - start
    window_t = np.arange(window_len) * DT_MS - (WARMUP + PRE_CONTEXT) * DT_MS
    window_voltage = coarse_voltage[start:end]
    window_mask = coarse_mask[start:end]
    window_post_spikes = spike_matrix[neuron_id, start:end].astype(np.float32)

    native_path = os.path.join(
        OUTPUT_DIR,
        f'voltage_mask_example_native_{SESSION_NAME}_rec{recording_idx:03d}.png',
    )
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(11.5, 9.0),
        gridspec_kw={'height_ratios': [3.0, 1.0, 1.7]},
    )
    axs[0].plot(t_native, raw_native, color='black', lw=1.2, label='raw voltage')
    axs[0].axvspan(-MASK_PRE_MS, MASK_POST_MS, color='tab:red', alpha=0.15, label='masked neighborhood')
    masked_native = mask_native < 0.5
    if np.any(masked_native):
        axs[0].scatter(
            t_native[masked_native],
            raw_native[masked_native],
            color='tab:red',
            s=14,
            label='masked native samples',
        )
    axs[0].axvline(0.0, color='tab:blue', ls='--', lw=1.0, label='selected spike time')
    axs[0].set_ylabel('Voltage (mV)')
    axs[0].set_title(
        f'Native {sample_rate_ms:.1f} ms mask example: recording {recording_idx:03d}, '
        f'neuron {neuron_id}'
    )
    axs[0].legend(loc='best', fontsize=8)

    axs[1].step(t_native, mask_native, where='mid', color='tab:green', lw=1.5)
    axs[1].axhline(0.5, color='gray', ls='--', lw=1.0)
    axs[1].axvspan(-MASK_PRE_MS, MASK_POST_MS, color='tab:red', alpha=0.15)
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_ylabel('Native\nvalid mask')

    axs[2].bar(
        t_coarse,
        mask_coarse,
        width=0.9 * DT_MS,
        color=['tab:green' if value > 0.5 else 'tab:red' for value in mask_coarse],
        alpha=0.28,
        label='1 ms mask fraction',
    )
    axs[2].scatter(
        sample_times[sample_valid],
        sample_slot_y[sample_valid],
        color='tab:green',
        s=18,
        label='kept 0.1 ms samples',
        zorder=3,
    )
    masked_samples = ~sample_valid
    if np.any(masked_samples):
        axs[2].scatter(
            sample_times[masked_samples],
            sample_slot_y[masked_samples],
            color='tab:red',
            marker='x',
            s=20,
            label='masked 0.1 ms samples',
            zorder=3,
        )
    axs[2].axhline(0.5, color='gray', ls='--', lw=1.0, label='loss cutoff')
    axs[2].axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].set_ylabel('1 ms bins\n+ native slots')
    axs[2].set_xlabel('Time relative to spike (ms)')
    axs[2].legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(native_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    window_path = os.path.join(
        OUTPUT_DIR,
        f'voltage_mask_example_window_{SESSION_NAME}_rec{recording_idx:03d}.png',
    )
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [1.2, 3.0, 1.4]},
    )
    spike_x = window_t[window_post_spikes > 0.5]
    if spike_x.size > 0:
        axs[0].vlines(spike_x, 0.0, 1.0, color='tab:orange', lw=1.4)
    axs[0].axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    axs[0].axvspan(window_t[0] - 0.5 * DT_MS, window_t[WARMUP - 1] + 0.5 * DT_MS, color='gray', alpha=0.12)
    axs[0].set_ylim(0.0, 1.05)
    axs[0].set_ylabel('Post\nspikes')
    axs[0].set_title(
        f'Actual 1 ms event window: recording {recording_idx:03d}, neuron {neuron_id}'
    )

    valid_window = window_mask > 0.5
    axs[1].plot(window_t, window_voltage, color='lightgray', lw=1.0, label='stored target voltage')
    if np.any(valid_window):
        axs[1].plot(
            window_t[valid_window],
            window_voltage[valid_window],
            color='tab:blue',
            lw=1.6,
            label='supervised voltage samples',
        )
    if np.any(~valid_window):
        axs[1].scatter(
            window_t[~valid_window],
            window_voltage[~valid_window],
            color='tab:red',
            marker='x',
            s=22,
            label='masked / ignored bins',
        )
    axs[1].axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    axs[1].axvspan(
        window_t[0] - 0.5 * DT_MS,
        window_t[WARMUP - 1] + 0.5 * DT_MS,
        color='gray',
        alpha=0.12,
        label='warmup (excluded from loss)',
    )
    axs[1].set_ylabel('Normalized\nvoltage target')
    axs[1].legend(loc='best', fontsize=8)

    axs[2].bar(
        window_t,
        window_mask,
        width=0.9 * DT_MS,
        color=['tab:green' if value > 0.5 else 'tab:red' for value in window_mask],
        alpha=0.8,
    )
    axs[2].axhline(0.5, color='gray', ls='--', lw=1.0, label='voltage-loss cutoff')
    axs[2].axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    axs[2].axvspan(window_t[0] - 0.5 * DT_MS, window_t[WARMUP - 1] + 0.5 * DT_MS, color='gray', alpha=0.12)
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].set_ylabel('Window\nmask frac')
    axs[2].set_xlabel('Time relative to selected postsynaptic spike (ms)')
    axs[2].legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(window_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    return {
        'native_path': native_path,
        'window_path': window_path,
    }


def render_prediction_figure(example):
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    threshold_mode = infer_threshold_mode(state_dict)

    model = VoltageAugmentedPerNeuronLIF(
        n_neurons=int(checkpoint['n_neurons']),
        K=int(checkpoint['K']),
        max_delay=int(checkpoint['max_delay']),
        threshold_mode=threshold_mode,
    )
    safe_load_state_dict(model, state_dict)
    model.eval()

    post_id = int(example['neuron_id'])
    start = int(example['start'])
    end = int(example['end'])
    neighbor_indices = checkpoint['neighbor_indices']
    pre_ids = np.asarray(neighbor_indices[post_id], dtype=np.int64)

    pre_spikes = example['spike_matrix'][pre_ids, start:end].astype(np.float32)
    if pre_spikes.shape[0] < model.K:
        padded = np.zeros((model.K, pre_spikes.shape[1]), dtype=np.float32)
        padded[:pre_spikes.shape[0]] = pre_spikes
        pre_spikes = padded
    post_spikes = example['spike_matrix'][post_id, start:end].astype(np.float32)
    target_voltage = example['coarse_voltage'][start:end].astype(np.float32)
    target_mask = example['coarse_mask'][start:end].astype(np.float32)

    with torch.no_grad():
        spike_probs, predicted_voltage, _ = model(
            torch.from_numpy(pre_spikes).unsqueeze(0),
            torch.from_numpy(post_spikes).unsqueeze(0),
            torch.tensor([post_id], dtype=torch.long),
            tbptt_len=pre_spikes.shape[1],
        )

    spike_probs = spike_probs[0].cpu().numpy()
    predicted_voltage = predicted_voltage[0].cpu().numpy()
    window_t = np.arange(end - start) * DT_MS - (WARMUP + PRE_CONTEXT) * DT_MS
    supervised_mask = np.zeros_like(target_mask, dtype=bool)
    supervised_mask[WARMUP:] = target_mask[WARMUP:] > 0.5
    ignored_mask = ~supervised_mask

    valid_points = int(supervised_mask.sum())
    if valid_points > 0:
        mae = float(np.mean(np.abs(predicted_voltage[supervised_mask] - target_voltage[supervised_mask])))
    else:
        mae = float('nan')

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f'voltage_prediction_example_{SESSION_NAME}_rec{example["recording_idx"]:03d}.png',
    )
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={'height_ratios': [1.2, 3.2, 1.4]},
    )

    spike_x = window_t[post_spikes > 0.5]
    if spike_x.size > 0:
        axs[0].vlines(spike_x, 0.0, 1.0, color='tab:orange', lw=1.4, label='true post spikes')
    axs[0].plot(window_t, spike_probs, color='tab:purple', lw=1.2, label='predicted spike prob')
    axs[0].axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    axs[0].axvspan(window_t[0] - 0.5 * DT_MS, window_t[WARMUP - 1] + 0.5 * DT_MS, color='gray', alpha=0.12)
    axs[0].set_ylim(0.0, 1.05)
    axs[0].set_ylabel('Spike\noutput')
    axs[0].set_title(
        f'Trained voltage-augmented model on one real window: recording {example["recording_idx"]:03d}, '
        f'neuron {post_id}'
    )
    axs[0].legend(loc='best', fontsize=8)

    axs[1].plot(window_t, predicted_voltage, color='black', lw=1.4, label='predicted voltage')
    axs[1].plot(window_t, target_voltage, color='lightgray', lw=1.0, label='stored target voltage')
    if valid_points > 0:
        axs[1].plot(
            window_t[supervised_mask],
            target_voltage[supervised_mask],
            color='tab:blue',
            lw=1.8,
            label='supervised target voltage',
        )
    if np.any(ignored_mask):
        axs[1].scatter(
            window_t[ignored_mask],
            target_voltage[ignored_mask],
            color='tab:red',
            marker='x',
            s=22,
            label='warmup or masked bins',
        )
    axs[1].axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    axs[1].axvspan(
        window_t[0] - 0.5 * DT_MS,
        window_t[WARMUP - 1] + 0.5 * DT_MS,
        color='gray',
        alpha=0.12,
        label='warmup (excluded from loss)',
    )
    axs[1].set_ylabel('Normalized\nvoltage')
    axs[1].legend(loc='best', fontsize=8)

    axs[2].bar(
        window_t,
        target_mask,
        width=0.9 * DT_MS,
        color=['tab:green' if value > 0.5 else 'tab:red' for value in target_mask],
        alpha=0.8,
        label='target mask fraction',
    )
    axs[2].axhline(0.5, color='gray', ls='--', lw=1.0, label='loss cutoff')
    axs[2].axvline(0.0, color='tab:blue', ls='--', lw=1.0)
    axs[2].axvspan(window_t[0] - 0.5 * DT_MS, window_t[WARMUP - 1] + 0.5 * DT_MS, color='gray', alpha=0.12)
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].set_ylabel('Window\nmask frac')
    axs[2].set_xlabel('Time relative to selected postsynaptic spike (ms)')
    summary_text = (
        f'supervised bins after warmup: {valid_points}/{len(target_mask) - WARMUP}\n'
        f'MAE on supervised bins: {mae:.3f}'
    )
    axs[2].text(
        0.01,
        0.95,
        summary_text,
        transform=axs[2].transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75},
    )
    axs[2].legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(prediction_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    return {
        'prediction_path': prediction_path,
        'checkpoint_path': checkpoint_path,
        'mae_supervised': mae,
        'n_supervised_bins': valid_points,
    }


def summarize_example(example, figure_paths):
    summary = {
        'recording_idx': example['recording_idx'],
        'neuron_id': example['neuron_id'],
        'spike_time_ms': example['spike_time_ms'],
        'spike_bin_dt': example['spike_bin_dt'],
        'sample_rate_ms': example['sample_rate_ms'],
        'dt_ms': DT_MS,
        'mask_pre_ms': MASK_PRE_MS,
        'mask_post_ms': MASK_POST_MS,
        'burst_excluded_bins': example['burst_excluded_bins'],
    }
    summary.update(figure_paths)
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    primary = load_recording_example(PRIMARY_RECORDING_IDX)
    secondary = load_alternate_example(PRIMARY_RECORDING_IDX)

    primary_paths = render_mask_figures(primary)
    secondary_paths = render_mask_figures(secondary)
    prediction_info = render_prediction_figure(primary)

    summary = {
        'primary_example': summarize_example(primary, primary_paths),
        'secondary_example': summarize_example(secondary, secondary_paths),
        'prediction_example': prediction_info,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()