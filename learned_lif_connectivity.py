"""
Learned LIF Connectivity Inference

For each postsynaptic neuron, fits a differentiable LIF model:
at each timestep, computes weighted sum of delayed presynaptic inputs,
applies learnable threshold and membrane dynamics, predicts the output spike.

Each postsynaptic neuron has its OWN learnable weight vector w[K].
After training, the weight matrix directly reveals connectivity:
  |w| large = connected, |w| ≈ 0 = not connected.

Global membrane parameters (alpha, threshold, beta, reset) are shared.

Usage:
    python learned_lif_connectivity.py [--k 50] [--session path]
"""

import numpy as np
import os
import sys
import glob
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# DATA
# ============================================================================

def spike_times_to_binary(spike_times, duration_ms, dt=1.0):
    """Convert spike times to binary matrix [n_neurons, T] at dt resolution."""
    n_neurons = len(spike_times)
    n_bins = int(duration_ms / dt)
    binary = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for i in range(n_neurons):
        for t in spike_times[i]:
            idx = int(t / dt)
            if 0 <= idx < n_bins:
                binary[i, idx] = 1
    return binary


def build_ground_truth(connections, n_neurons):
    """W[post, pre] = weight, B[post, pre] = 1 if connected."""
    W = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    B = np.zeros((n_neurons, n_neurons), dtype=np.int32)
    for c in connections:
        pre, post = int(c[0]), int(c[1])
        W[post, pre] = float(c[2])
        B[post, pre] = 1
    return W, B


def compute_spatial_neighbor_indices(neuron_positions, K):
    """For each neuron, find K nearest spatial neighbors."""
    n = len(neuron_positions)
    dist = np.sqrt(((neuron_positions[:, None, :] -
                     neuron_positions[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    K_actual = min(K, n - 1)
    indices = np.argsort(dist, axis=1)[:, :K_actual]
    return indices, K_actual, dist


def build_recording_spike_indices(spike_matrix, boundaries=None, excluded_bins=None):
    """Build per-recording spike index lists for each neuron."""
    n_neurons, T = spike_matrix.shape
    if boundaries is None:
        boundaries = [0, T]

    excluded_bins_sorted = None
    if excluded_bins is not None and len(excluded_bins) > 0:
        excluded_bins_sorted = np.sort(np.asarray(excluded_bins, dtype=np.int32))

    spike_indices = []
    spike_counts = np.zeros(n_neurons, dtype=np.int32)

    for rec_idx in range(len(boundaries) - 1):
        start = boundaries[rec_idx]
        end = boundaries[rec_idx + 1]
        rec_excluded_local = None
        if excluded_bins_sorted is not None:
            left = np.searchsorted(excluded_bins_sorted, start, side='left')
            right = np.searchsorted(excluded_bins_sorted, end, side='left')
            if right > left:
                rec_excluded_local = excluded_bins_sorted[left:right] - start

        rec_spikes = []
        for neuron_id in range(n_neurons):
            spikes = np.flatnonzero(spike_matrix[neuron_id, start:end])
            if rec_excluded_local is not None and len(spikes) > 0:
                spikes = spikes[~np.isin(spikes, rec_excluded_local, assume_unique=True)]
            rec_spikes.append(spikes)
            spike_counts[neuron_id] += len(spikes)
        spike_indices.append(rec_spikes)

    return spike_indices, spike_counts


def causal_pair_score(pre_spikes, post_spikes, min_lag=1, max_lag=8):
    """
    Score a candidate edge by how often a presynaptic spike closely precedes a
    postsynaptic spike. Shorter lags receive more weight.
    """
    if len(pre_spikes) == 0 or len(post_spikes) == 0:
        return 0.0

    idx = np.searchsorted(pre_spikes, post_spikes - min_lag, side='right') - 1
    valid = idx >= 0
    if not np.any(valid):
        return 0.0

    matched_pre = pre_spikes[idx[valid]]
    lags = post_spikes[valid] - matched_pre
    valid_lags = (lags >= min_lag) & (lags <= max_lag)
    if not np.any(valid_lags):
        return 0.0

    lags = lags[valid_lags].astype(np.float32)
    return float(np.sum((max_lag - lags + 1.0) / max_lag))


def compute_temporal_candidate_scores(spike_matrix, boundaries=None,
                                      min_lag=1, max_lag=8,
                                      excluded_bins=None):
    """
    Compute causal temporal candidate scores from spike timing alone.

    Scores are accumulated within recordings only, so concatenated sessions do
    not introduce cross-recording leakage.
    """
    n_neurons = spike_matrix.shape[0]
    spike_indices_by_recording, spike_counts = build_recording_spike_indices(
        spike_matrix, boundaries=boundaries, excluded_bins=excluded_bins,
    )

    scores = np.zeros((n_neurons, n_neurons), dtype=np.float32)

    for rec_spikes in spike_indices_by_recording:
        active_neurons = [idx for idx, spikes in enumerate(rec_spikes)
                          if len(spikes) > 0]
        for post_id in active_neurons:
            post_spikes = rec_spikes[post_id]
            for pre_id in active_neurons:
                if pre_id == post_id:
                    continue
                score = causal_pair_score(
                    rec_spikes[pre_id], post_spikes,
                    min_lag=min_lag, max_lag=max_lag,
                )
                if score > 0:
                    scores[post_id, pre_id] += score

    norm = np.sqrt(np.outer(spike_counts, spike_counts)).astype(np.float32)
    valid = norm > 0
    scores[valid] = scores[valid] / norm[valid]
    np.fill_diagonal(scores, -np.inf)
    return scores


def compute_neighbor_indices(neuron_positions, K, spike_matrix=None,
                             mode='spatial', boundaries=None,
                             spatial_frac=0.7,
                             excluded_bins=None,
                             temporal_min_lag=1,
                             temporal_max_lag=8):
    """
    Build candidate presynaptic sets for each postsynaptic neuron.

    Modes:
      - spatial: pure K-nearest neighbors in physical space
      - hybrid: mix nearest spatial neighbors with top causal temporal matches
    """
    spatial_indices, K_actual, _ = compute_spatial_neighbor_indices(
        neuron_positions, K,
    )
    info = {
        'mode': mode,
        'K': K_actual,
        'spatial_frac': float(spatial_frac),
        'temporal_min_lag': int(temporal_min_lag),
        'temporal_max_lag': int(temporal_max_lag),
        'mean_temporal_only': 0.0,
    }

    if mode != 'hybrid' or spike_matrix is None or K_actual <= 1:
        return spatial_indices, K_actual, info

    spatial_frac = float(np.clip(spatial_frac, 0.0, 1.0))
    n_spatial = int(round(K_actual * spatial_frac))
    n_spatial = max(1, min(n_spatial, K_actual))
    n_temporal = K_actual - n_spatial

    if n_temporal <= 0:
        info['mode'] = 'spatial'
        return spatial_indices, K_actual, info

    temporal_scores = compute_temporal_candidate_scores(
        spike_matrix,
        boundaries=boundaries,
        min_lag=temporal_min_lag,
        max_lag=temporal_max_lag,
        excluded_bins=excluded_bins,
    )
    temporal_order = np.argsort(temporal_scores, axis=1)[:, ::-1]

    n_neurons = len(neuron_positions)
    hybrid_indices = np.zeros((n_neurons, K_actual), dtype=np.int32)
    temporal_only_counts = []

    for post_id in range(n_neurons):
        chosen = []
        chosen_set = {post_id}

        for pre_id in spatial_indices[post_id]:
            if pre_id in chosen_set:
                continue
            chosen.append(int(pre_id))
            chosen_set.add(int(pre_id))
            if len(chosen) >= n_spatial:
                break

        temporal_added = 0
        for pre_id in temporal_order[post_id]:
            pre_id = int(pre_id)
            if pre_id in chosen_set:
                continue
            if temporal_scores[post_id, pre_id] <= 0:
                break
            chosen.append(pre_id)
            chosen_set.add(pre_id)
            temporal_added += 1
            if temporal_added >= n_temporal:
                break

        if len(chosen) < K_actual:
            for pre_id in spatial_indices[post_id]:
                pre_id = int(pre_id)
                if pre_id in chosen_set:
                    continue
                chosen.append(pre_id)
                chosen_set.add(pre_id)
                if len(chosen) >= K_actual:
                    break

        if len(chosen) < K_actual:
            for pre_id in temporal_order[post_id]:
                pre_id = int(pre_id)
                if pre_id in chosen_set:
                    continue
                chosen.append(pre_id)
                chosen_set.add(pre_id)
                if len(chosen) >= K_actual:
                    break

        hybrid_indices[post_id] = np.asarray(chosen[:K_actual], dtype=np.int32)
        baseline_spatial = set(spatial_indices[post_id].tolist())
        temporal_only_counts.append(
            sum(1 for pre_id in hybrid_indices[post_id] if pre_id not in baseline_spatial)
        )

    info['n_spatial'] = n_spatial
    info['n_temporal'] = n_temporal
    info['mean_temporal_only'] = float(np.mean(temporal_only_counts))
    return hybrid_indices, K_actual, info


def load_all_recordings(session_dir, dt=1.0):
    """
    Load and concatenate ALL recordings in a session along the time axis.

    Returns:
        spike_matrix: [n_neurons, total_T] concatenated spike trains
        total_duration: total duration in ms
        boundaries: list of bin indices where each recording starts/ends
                    e.g. [0, 60000, 120000, ...] for 5 recordings of 60s each
        net_data: loaded network data dict
    """
    rec_files = sorted(glob.glob(os.path.join(session_dir, 'recording[0-9][0-9][0-9].npz')))
    if not rec_files:
        raise FileNotFoundError(f"No recordings in {session_dir}")

    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f"No network file in {session_dir}")
    net_data = np.load(net_files[0], allow_pickle=True)

    all_matrices = []
    boundaries = [0]
    total_duration = 0.0
    burst_onset_bins = []

    for rec_file in rec_files:
        data = np.load(rec_file, allow_pickle=True)
        duration = float(data['duration'])
        rec_start_bin = boundaries[-1]
        spike_matrix = spike_times_to_binary(data['spike_times'], duration, dt)
        all_matrices.append(spike_matrix)

        if 'burst_onset_times' in data.files:
            rec_burst_onsets = np.asarray(data['burst_onset_times'], dtype=float)
            rec_burst_onsets = rec_burst_onsets[
                (rec_burst_onsets >= 0.0) & (rec_burst_onsets < duration)
            ]
            if rec_burst_onsets.size > 0:
                burst_onset_bins.extend(
                    (rec_burst_onsets / dt).astype(np.int32) + rec_start_bin
                )

        total_duration += duration
        boundaries.append(boundaries[-1] + spike_matrix.shape[1])

    concatenated = np.concatenate(all_matrices, axis=1)
    burst_onset_bins = np.unique(np.asarray(burst_onset_bins, dtype=np.int32))

    return {
        'spike_matrix': concatenated,
        'total_duration': total_duration,
        'boundaries': boundaries,
        'burst_onset_bins': burst_onset_bins,
        'n_recordings': len(rec_files),
        'connections': net_data['connections'],
        'neuron_positions': net_data['neuron_positions'],
        'cluster_assignments': net_data['cluster_assignments'],
        'n_neurons': len(net_data['neuron_positions']),
    }


def merge_excluded_windows(excluded_windows, max_gap_bins=0):
    """Merge overlapping or nearby [start, end) exclusion windows."""
    if excluded_windows is None or len(excluded_windows) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    max_gap_bins = max(0, int(max_gap_bins))
    ordered = sorted(
        (int(start), int(end))
        for start, end in excluded_windows
        if int(end) > int(start)
    )
    if not ordered:
        return np.zeros((0, 2), dtype=np.int32)

    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start <= current[1] + max_gap_bins:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])

    return np.asarray(merged, dtype=np.int32)


def excluded_windows_to_bins(excluded_windows):
    """Expand exclusion windows into sorted bin indices."""
    if excluded_windows is None or len(excluded_windows) == 0:
        return np.array([], dtype=np.int32)

    ranges = [
        np.arange(int(start), int(end), dtype=np.int32)
        for start, end in excluded_windows
        if int(end) > int(start)
    ]
    if not ranges:
        return np.array([], dtype=np.int32)
    return np.unique(np.concatenate(ranges))


def combine_excluded_bins(*bin_arrays):
    """Combine multiple exclusion-bin sources into one sorted unique array."""
    valid_arrays = []
    for bins in bin_arrays:
        if bins is None:
            continue
        bins = np.asarray(bins, dtype=np.int32)
        if bins.size > 0:
            valid_arrays.append(bins)

    if not valid_arrays:
        return np.array([], dtype=np.int32)
    return np.unique(np.concatenate(valid_arrays))


def _find_true_segments(mask):
    """Return contiguous [start, end) segments where mask is True."""
    if mask.size == 0 or not np.any(mask):
        return []

    padded = np.pad(mask.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def detect_network_burst_windows(spike_matrix, recording_boundaries,
                                 dt_ms=1.0,
                                 activity_bin_ms=100,
                                 smooth_bins=3,
                                 threshold_std=3.0,
                                 min_active_fraction=0.10,
                                 min_burst_duration_ms=100,
                                 merge_gap_ms=150,
                                 pad_before_ms=100,
                                 pad_after_ms=250):
    """
    Detect network-wide burst windows from population synchrony.

    Uses the fraction of neurons active in coarse time bins, smooths the trace,
    thresholds it relative to baseline, merges nearby segments, and pads the
    resulting windows to cover pre-burst recruitment and burst tails.
    """
    n_neurons, T = spike_matrix.shape
    if recording_boundaries is None:
        recording_boundaries = [0, T]

    activity_bin_bins = max(1, int(round(activity_bin_ms / dt_ms)))
    smooth_bins = max(1, int(smooth_bins))
    min_burst_bins = max(1, int(np.ceil(min_burst_duration_ms / dt_ms)))
    merge_gap_bins = max(0, int(round(merge_gap_ms / dt_ms)))
    pad_before_bins = max(0, int(round(pad_before_ms / dt_ms)))
    pad_after_bins = max(0, int(round(pad_after_ms / dt_ms)))

    kernel = np.ones(smooth_bins, dtype=np.float32) / float(smooth_bins)
    detected_windows = []
    per_recording_counts = []
    thresholds = []

    for rec_idx in range(len(recording_boundaries) - 1):
        rec_start = int(recording_boundaries[rec_idx])
        rec_end = int(recording_boundaries[rec_idx + 1])
        rec_length = rec_end - rec_start
        if rec_length <= 0:
            per_recording_counts.append(0)
            thresholds.append(float(min_active_fraction))
            continue

        n_activity_bins = int(np.ceil(rec_length / activity_bin_bins))
        active_fraction = np.zeros(n_activity_bins, dtype=np.float32)

        for bin_idx in range(n_activity_bins):
            start = rec_start + bin_idx * activity_bin_bins
            end = min(rec_end, start + activity_bin_bins)
            active_fraction[bin_idx] = np.mean(
                np.any(spike_matrix[:, start:end] > 0, axis=1)
            )

        smoothed = np.convolve(active_fraction, kernel, mode='same')
        threshold = max(
            float(np.mean(smoothed) + threshold_std * np.std(smoothed)),
            float(min_active_fraction),
        )
        thresholds.append(threshold)

        coarse_segments = _find_true_segments(smoothed >= threshold)
        coarse_windows = []
        for start_idx, end_idx in coarse_segments:
            start = rec_start + start_idx * activity_bin_bins
            end = min(rec_end, rec_start + end_idx * activity_bin_bins)
            if end - start >= min_burst_bins:
                coarse_windows.append((start, end))

        merged = merge_excluded_windows(coarse_windows, max_gap_bins=merge_gap_bins)
        padded = [
            (
                max(rec_start, int(start) - pad_before_bins),
                min(rec_end, int(end) + pad_after_bins),
            )
            for start, end in merged
        ]
        padded = merge_excluded_windows(padded, max_gap_bins=0)

        detected_windows.extend((int(start), int(end)) for start, end in padded)
        per_recording_counts.append(len(padded))

    detected_windows = merge_excluded_windows(detected_windows, max_gap_bins=0)
    excluded_bins = excluded_windows_to_bins(detected_windows)

    return {
        'windows': detected_windows,
        'excluded_bins': excluded_bins,
        'per_recording_counts': np.asarray(per_recording_counts, dtype=np.int32),
        'thresholds': np.asarray(thresholds, dtype=np.float32),
        'activity_bin_ms': float(activity_bin_ms),
        'smooth_bins': int(smooth_bins),
        'threshold_std': float(threshold_std),
        'min_active_fraction': float(min_active_fraction),
        'min_burst_duration_ms': float(min_burst_duration_ms),
        'merge_gap_ms': float(merge_gap_ms),
        'pad_before_ms': float(pad_before_ms),
        'pad_after_ms': float(pad_after_ms),
    }


def window_overlaps_excluded_bins(start, end, excluded_bins):
    """Return True if [start, end) contains any excluded bin."""
    if excluded_bins is None or len(excluded_bins) == 0:
        return False

    idx = np.searchsorted(excluded_bins, start, side='left')
    return idx < len(excluded_bins) and int(excluded_bins[idx]) < end


def find_event_windows(spike_matrix, neuron_id, pre_context=50, post_context=10,
                       warmup=30, neg_ratio=1.0, neg_min_distance=100,
                       boundaries=None, excluded_bins=None, rng=None):
    """
    Extract positive (spike-centered) and negative (no-spike) event windows.

    Each window is [start, end) with length = warmup + pre_context + post_context.
    For positive windows, the post spike falls at position (warmup + pre_context).
    Warmup region is used to let membrane voltage settle before loss is computed.

    Args:
        spike_matrix: [n_neurons, T] binary spike matrix
        neuron_id: postsynaptic neuron index
        pre_context: bins before the event to include (for causal pre input)
        post_context: bins after the event (captures spike + reset)
        warmup: bins before pre_context (membrane settling, no loss)
        neg_ratio: n_neg = n_pos * neg_ratio
        neg_min_distance: min distance (bins) from any post spike for negatives
        boundaries: recording boundaries [0, T1, T1+T2, ...] to avoid crossing
        excluded_bins: sorted bin indices that windows must not overlap
        rng: numpy random state

    Returns:
        pos_windows: list of (start, end) tuples
        neg_windows: list of (start, end) tuples
    """
    if rng is None:
        rng = np.random.RandomState(42 + int(neuron_id))

    T = spike_matrix.shape[1]
    window_len = warmup + pre_context + post_context

    if boundaries is None:
        boundaries = [0, T]

    # All post spike times
    post_spikes = np.where(spike_matrix[neuron_id] == 1)[0]

    # Build valid time ranges that can fully contain a window
    # (window must fit entirely within one recording, no crossing boundaries)
    valid_ranges = []
    for i in range(len(boundaries) - 1):
        rec_start = boundaries[i]
        rec_end = boundaries[i + 1]
        # A window centered at t has range [t - pre_context - warmup, t + post_context)
        min_t = rec_start + warmup + pre_context
        max_t = rec_end - post_context
        if max_t > min_t:
            valid_ranges.append((min_t, max_t))

    # Positive windows: centered at each post spike that's in a valid range
    pos_windows = []
    for t in post_spikes:
        for r_start, r_end in valid_ranges:
            if r_start <= t < r_end:
                start = t - pre_context - warmup
                end = t + post_context
                if not window_overlaps_excluded_bins(start, end, excluded_bins):
                    pos_windows.append((int(start), int(end)))
                break

    n_pos = len(pos_windows)
    n_neg = int(n_pos * neg_ratio)
    neg_windows = []

    if n_neg > 0 and valid_ranges:
        spike_set = set(post_spikes.tolist())
        attempts = 0
        max_attempts = n_neg * 200

        while len(neg_windows) < n_neg and attempts < max_attempts:
            attempts += 1
            range_idx = rng.randint(len(valid_ranges))
            r_start, r_end = valid_ranges[range_idx]
            if r_end <= r_start:
                continue
            t = rng.randint(r_start, r_end)

            # Check: no post spike within neg_min_distance of t
            nearby_min = max(0, t - neg_min_distance)
            nearby_max = min(T, t + neg_min_distance)
            if spike_matrix[neuron_id, nearby_min:nearby_max].sum() == 0:
                start = t - pre_context - warmup
                end = t + post_context
                if not window_overlaps_excluded_bins(start, end, excluded_bins):
                    neg_windows.append((int(start), int(end)))

    return pos_windows, neg_windows


class NeuronDataset(Dataset):
    """Each sample = one postsynaptic neuron with K pre candidates."""

    def __init__(self, spike_matrix, neighbor_indices, true_binary, true_weights,
                 neuron_positions, neuron_ids=None):
        self.spike_matrix = spike_matrix
        self.neighbor_indices = neighbor_indices
        self.true_binary = true_binary
        self.true_weights = true_weights
        self.neuron_positions = neuron_positions
        self.neuron_ids = neuron_ids if neuron_ids is not None else \
            np.arange(len(spike_matrix))

    def __len__(self):
        return len(self.neuron_ids)

    def __getitem__(self, idx):
        post_id = self.neuron_ids[idx]
        pre_ids = self.neighbor_indices[post_id]

        pre_spikes = self.spike_matrix[pre_ids]             # [K, T]
        post_spikes = self.spike_matrix[post_id]             # [T]
        labels = self.true_binary[post_id, pre_ids].astype(np.float32)
        weights = self.true_weights[post_id, pre_ids].astype(np.float32)

        return (torch.from_numpy(pre_spikes),
                torch.from_numpy(post_spikes),
                torch.from_numpy(labels),
                torch.from_numpy(weights),
                post_id)


class EventWindowDataset(Dataset):
    """
    Event-window dataset for focused training.

    Each sample is a short window (warmup + pre_context + post_context bins)
    extracted from the full spike matrix. Windows are either:
      - Positive: centered on an actual post spike
      - Negative: centered on a time far from any post spike

    Instead of training on 60,000 mostly-zero timesteps per neuron, we train
    on ~60 carefully-chosen bins × (n_pos + n_neg) events per neuron.
    This fixes the class imbalance problem that caused the trivial "always zero"
    solution.
    """

    def __init__(self, spike_matrix, neighbor_indices, neuron_ids=None,
                 pre_context=50, post_context=10, warmup=30,
                 neg_ratio=1.0, neg_min_distance=100, boundaries=None,
                 excluded_bins=None, rng_seed=42, windows=None):
        self.spike_matrix = spike_matrix
        self.neighbor_indices = neighbor_indices
        self.pre_context = pre_context
        self.post_context = post_context
        self.warmup = warmup
        self.window_len = warmup + pre_context + post_context
        self.excluded_bins = None
        if excluded_bins is not None and len(excluded_bins) > 0:
            self.excluded_bins = np.sort(np.asarray(excluded_bins, dtype=np.int32))

        if windows is not None:
            self.windows = [
                (int(post_id), int(start), int(end), int(is_pos))
                for post_id, start, end, is_pos in windows
            ]
        else:
            if neuron_ids is None:
                neuron_ids = np.arange(spike_matrix.shape[0])

            # Extract windows for each post neuron
            self.windows = []  # list of (post_id, start, end, is_positive)
            for post_id in neuron_ids:
                rng = np.random.RandomState(rng_seed + int(post_id))
                pos_w, neg_w = find_event_windows(
                    spike_matrix, post_id,
                    pre_context=pre_context,
                    post_context=post_context,
                    warmup=warmup,
                    neg_ratio=neg_ratio,
                    neg_min_distance=neg_min_distance,
                    boundaries=boundaries,
                    excluded_bins=self.excluded_bins,
                    rng=rng,
                )
                for start, end in pos_w:
                    self.windows.append((int(post_id), start, end, 1))
                for start, end in neg_w:
                    self.windows.append((int(post_id), start, end, 0))

        self.n_pos = sum(1 for _, _, _, is_pos in self.windows if is_pos == 1)
        self.n_neg = sum(1 for _, _, _, is_pos in self.windows if is_pos == 0)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        post_id, start, end, is_pos = self.windows[idx]
        pre_ids = self.neighbor_indices[post_id]
        pre_spikes = self.spike_matrix[pre_ids, start:end].astype(np.float32)
        post_spikes = self.spike_matrix[post_id, start:end].astype(np.float32)

        return (torch.from_numpy(pre_spikes),
                torch.from_numpy(post_spikes),
                post_id,
                is_pos)


def split_event_windows(windows, val_fraction=0.2, rng_seed=42):
    """
    Split event windows into train/validation subsets while keeping every
    postsynaptic neuron in the training set.

    Stratifies by (post_id, is_positive) so each neuron keeps both positive and
    negative training windows whenever possible.
    """
    windows = list(windows)
    if not windows or val_fraction <= 0:
        return windows, []

    rng = np.random.RandomState(rng_seed)
    grouped = {}
    for window in windows:
        key = (int(window[0]), int(window[3]))
        grouped.setdefault(key, []).append(window)

    train_windows = []
    val_windows = []

    for group_windows in grouped.values():
        order = rng.permutation(len(group_windows))
        shuffled = [group_windows[i] for i in order]

        if len(shuffled) < 2:
            train_windows.extend(shuffled)
            continue

        n_val = int(round(len(shuffled) * val_fraction))
        n_val = max(1, n_val)
        n_val = min(n_val, len(shuffled) - 1)

        val_windows.extend(shuffled[:n_val])
        train_windows.extend(shuffled[n_val:])

    if not val_windows and train_windows:
        val_windows.append(train_windows.pop())

    rng.shuffle(train_windows)
    rng.shuffle(val_windows)
    return train_windows, val_windows


def split_recording_boundaries(boundaries, val_fraction=0.2):
    """Split concatenated recording boundaries into train and validation sets."""
    if boundaries is None or len(boundaries) < 3 or val_fraction <= 0:
        return boundaries, None

    n_recordings = len(boundaries) - 1
    n_val_recordings = int(round(n_recordings * val_fraction))
    n_val_recordings = max(1, n_val_recordings)
    n_val_recordings = min(n_val_recordings, n_recordings - 1)

    split_idx = n_recordings - n_val_recordings
    train_boundaries = boundaries[:split_idx + 1]
    val_boundaries = boundaries[split_idx:]
    return train_boundaries, val_boundaries


def build_train_val_event_datasets(spike_matrix, neighbor_indices, neuron_ids,
                                   pre_context=50, post_context=10, warmup=30,
                                   neg_ratio=1.0, neg_min_distance=100,
                                   boundaries=None, excluded_bins=None,
                                   val_fraction=0.2,
                                   rng_seed=42):
    """
    Build train/validation event datasets that match the per-neuron model.

    If multiple recordings are available, hold out entire recordings for the
    same neurons. Otherwise, hold out a fraction of event windows per neuron.
    """
    train_boundaries, val_boundaries = split_recording_boundaries(boundaries, val_fraction)

    if val_boundaries is not None:
        train_ds = EventWindowDataset(
            spike_matrix, neighbor_indices, neuron_ids=neuron_ids,
            pre_context=pre_context, post_context=post_context, warmup=warmup,
            neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
            boundaries=train_boundaries, excluded_bins=excluded_bins,
            rng_seed=rng_seed,
        )
        val_ds = EventWindowDataset(
            spike_matrix, neighbor_indices, neuron_ids=neuron_ids,
            pre_context=pre_context, post_context=post_context, warmup=warmup,
            neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
            boundaries=val_boundaries, excluded_bins=excluded_bins,
            rng_seed=rng_seed + 1000,
        )
        if len(train_ds) > 0 and len(val_ds) > 0:
            strategy = (
                f"held-out recordings ({len(train_boundaries) - 1} train, "
                f"{len(val_boundaries) - 1} val)"
            )
            if excluded_bins is not None and len(excluded_bins) > 0:
                strategy += f", excluding {len(excluded_bins)} bins"
            return train_ds, val_ds, strategy

    full_ds = EventWindowDataset(
        spike_matrix, neighbor_indices, neuron_ids=neuron_ids,
        pre_context=pre_context, post_context=post_context, warmup=warmup,
        neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
        boundaries=boundaries, excluded_bins=excluded_bins, rng_seed=rng_seed,
    )
    train_windows, val_windows = split_event_windows(
        full_ds.windows, val_fraction=val_fraction, rng_seed=rng_seed + 2000,
    )
    train_ds = EventWindowDataset(
        spike_matrix, neighbor_indices, warmup=warmup, windows=train_windows,
    )
    val_ds = EventWindowDataset(
        spike_matrix, neighbor_indices, warmup=warmup, windows=val_windows,
    )
    strategy = "held-out event windows"
    return train_ds, val_ds, strategy


# ============================================================================
# MODEL: Per-Neuron Differentiable LIF
# ============================================================================

class PerNeuronLIF(nn.Module):
    """
    Differentiable LIF with per-neuron learnable synaptic weights.

    Global (shared) parameters:
        - alpha: membrane leak factor
        - threshold: firing threshold
        - beta: spike sigmoid sharpness
        - reset_strength: post-spike voltage reset

    Per-neuron parameters:
        - W: [n_neurons, K] synaptic weights — THE connectivity we infer
        - delay_logits: [n_neurons, K, n_delays] per-synapse delay distributions

    At each timestep t, for postsynaptic neuron j:
        I_syn(t) = sum_i( W[j,i] * pre_i(t - delay_ji) )
        V(t) = alpha * V(t-1) + I_syn(t)
        spike_prob(t) = sigmoid( beta * (V(t) - threshold) )
        V(t) -= reset_strength * spike_prob(t)

    After training, connectivity matrix = W.
    """

    def __init__(self, n_neurons, K, max_delay=5):
        super().__init__()
        self.n_neurons = n_neurons
        self.K = K
        self.max_delay = max_delay

        # ── Per-neuron learnable weights [n_neurons, K] ──
        # Initialized at zero; positive → exc, negative → inh, ~zero → no connection
        self.W = nn.Parameter(torch.zeros(n_neurons, K))

        # ── Per-neuron delay distributions [n_neurons, K, max_delay] ──
        # Softmax over discrete delays [0, 1, ..., max_delay-1] ms
        self.delay_logits = nn.Parameter(torch.zeros(n_neurons, K, max_delay))

        # ── Global membrane parameters (shared across all neurons) ──
        self.alpha_logit = nn.Parameter(torch.tensor(3.0))   # sigmoid→ ~0.95 → tau_m ~20ms
        self.threshold = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.reset_strength = nn.Parameter(torch.tensor(2.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, pre_spikes, post_spikes, neuron_ids, tbptt_len=1000):
        """
        Forward pass with truncated backpropagation through time (TBPTT).

        Precomputes I_syn for all timesteps in one vectorized op (fast on GPU),
        then simulates membrane dynamics in chunks of tbptt_len steps.
        Gradients only flow within each chunk, not across the full 60,000 steps.

        Args:
            pre_spikes: [B, K, T] binary spike trains of K pre candidates
            post_spikes: [B, T] actual post spike train
            neuron_ids: [B] indices of the postsynaptic neurons in this batch
            tbptt_len: chunk size for truncated BPTT (default 1000 = 1 second)

        Returns:
            spike_probs: [B, T] predicted spike probability at each timestep
            voltages: [B, T] membrane voltage trace
            weights: [B, K] the learned synaptic weights for these neurons
        """
        B, K, T = pre_spikes.shape
        device = pre_spikes.device

        # Look up this batch's weights and delays
        w = self.W[neuron_ids]                           # [B, K]
        delay_logits = self.delay_logits[neuron_ids]     # [B, K, max_delay]
        delay_weights = F.softmax(delay_logits, dim=-1)  # [B, K, max_delay]

        # ── Vectorized I_syn computation (no Python loop over time) ──
        # Build delayed inputs using conv1d-style shifting
        delayed_inputs = torch.zeros(B, K, T, device=device)
        for d in range(self.max_delay):
            if d == 0:
                shifted = pre_spikes
            else:
                shifted = F.pad(pre_spikes[:, :, :-d], (d, 0))
            delayed_inputs += shifted * delay_weights[:, :, d].unsqueeze(-1)

        # Weighted sum across K pre neurons: [B, T]
        I_syn = (w.unsqueeze(-1) * delayed_inputs).sum(dim=1)

        # ── Membrane dynamics with truncated BPTT ──
        alpha = self.alpha
        beta = self.beta
        threshold = self.threshold
        reset = F.softplus(self.reset_strength)

        spike_probs_list = []
        voltages_list = []
        v = torch.zeros(B, device=device)

        for chunk_start in range(0, T, tbptt_len):
            chunk_end = min(chunk_start + tbptt_len, T)
            I_chunk = I_syn[:, chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start

            # Detach voltage at chunk boundary (truncated BPTT)
            v = v.detach()

            sp_chunk = torch.zeros(B, chunk_len, device=device)
            v_chunk = torch.zeros(B, chunk_len, device=device)

            for t in range(chunk_len):
                v = alpha * v + I_chunk[:, t]
                s = torch.sigmoid(beta * (v - threshold))
                sp_chunk[:, t] = s
                v_chunk[:, t] = v
                v = v - reset * s

            spike_probs_list.append(sp_chunk)
            voltages_list.append(v_chunk)

        spike_probs = torch.cat(spike_probs_list, dim=1)  # [B, T]
        voltages = torch.cat(voltages_list, dim=1)

        return spike_probs, voltages, w

    def get_connectivity_matrix(self, neighbor_indices):
        """
        Assemble full [n_neurons, n_neurons] connectivity matrix from learned weights.

        Returns:
            conn_matrix: [n_neurons, n_neurons] where conn_matrix[post, pre] = weight
        """
        W_np = self.W.detach().cpu().numpy()  # [n_neurons, K]
        n = self.n_neurons
        conn_matrix = np.zeros((n, n), dtype=np.float32)

        for j in range(n):
            pre_ids = neighbor_indices[j]
            conn_matrix[j, pre_ids] = W_np[j, :len(pre_ids)]

        return conn_matrix

    def get_learned_delays(self, neighbor_indices):
        """Assemble delay matrix [n_neurons, n_neurons] from learned delay distributions."""
        delay_weights = F.softmax(self.delay_logits, dim=-1).detach().cpu().numpy()
        delay_values = np.arange(self.max_delay)
        n = self.n_neurons
        delay_matrix = np.zeros((n, n), dtype=np.float32)

        for j in range(n):
            pre_ids = neighbor_indices[j]
            expected_delays = (delay_weights[j, :len(pre_ids)] * delay_values).sum(axis=-1)
            delay_matrix[j, pre_ids] = expected_delays

        return delay_matrix


# ============================================================================
# TRAINING
# ============================================================================

def compute_loss(spike_probs, post_spikes, weights, pos_weight=5.0, l1_lambda=0.01):
    """
    Loss = spike prediction BCE + L1 sparsity on weights.

    The spike prediction loss is the main signal: weights that help predict
    post spikes grow, weights that add noise shrink.
    L1 pushes unneeded weights to exactly zero.
    """
    weight_mask = torch.where(post_spikes == 1, pos_weight, 1.0)
    spike_loss = F.binary_cross_entropy(
        spike_probs.clamp(1e-7, 1 - 1e-7), post_spikes, weight=weight_mask
    )
    l1_loss = l1_lambda * weights.abs().mean()
    return spike_loss + l1_loss, spike_loss.item(), l1_loss.item()


def train_epoch(model, dataloader, optimizer, device, pos_weight, l1_lambda):
    model.train()
    total_loss = 0
    total_spike = 0
    total_l1 = 0
    n = 0

    for pre_sp, post_sp, labels, true_w, neuron_ids in dataloader:
        pre_sp = pre_sp.to(device)
        post_sp = post_sp.to(device)
        neuron_ids = neuron_ids.to(device)

        optimizer.zero_grad()
        spike_probs, voltages, weights = model(pre_sp, post_sp, neuron_ids)

        loss, sl, l1l = compute_loss(spike_probs, post_sp, weights, pos_weight, l1_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_spike += sl
        total_l1 += l1l
        n += 1

    return total_loss / n, total_spike / n, total_l1 / n


def compute_event_loss(spike_probs, post_spikes, weights, warmup,
                       pos_weight=5.0, l1_lambda=0.01):
    """
    Event-window loss: only compute BCE on non-warmup region of each window.

    The first `warmup` bins of each window let membrane voltage settle;
    no loss is computed there. Loss is only on the causal region
    [warmup : window_len] where the post spike may or may not occur.
    """
    # Mask out warmup region — loss only on post-warmup bins
    sp = spike_probs[:, warmup:]
    ps = post_spikes[:, warmup:]

    weight_mask = torch.where(ps == 1, pos_weight, 1.0)
    spike_loss = F.binary_cross_entropy(
        sp.clamp(1e-7, 1 - 1e-7), ps, weight=weight_mask
    )
    l1_loss = l1_lambda * weights.abs().mean()
    return spike_loss + l1_loss, spike_loss.item(), l1_loss.item()


def train_epoch_events(model, dataloader, optimizer, device, pos_weight,
                       l1_lambda, warmup):
    """Training epoch using event-window batches."""
    model.train()
    total_loss = 0
    total_spike = 0
    total_l1 = 0
    n = 0

    for pre_sp, post_sp, neuron_ids, is_pos in dataloader:
        pre_sp = pre_sp.to(device)
        post_sp = post_sp.to(device)
        neuron_ids = neuron_ids.to(device)

        optimizer.zero_grad()
        # For short windows (~90 bins), use full-window BPTT (no truncation needed)
        window_len = pre_sp.shape[2]
        spike_probs, voltages, weights = model(
            pre_sp, post_sp, neuron_ids, tbptt_len=window_len
        )

        loss, sl, l1l = compute_event_loss(
            spike_probs, post_sp, weights, warmup, pos_weight, l1_lambda
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_spike += sl
        total_l1 += l1l
        n += 1

    return total_loss / max(n, 1), total_spike / max(n, 1), total_l1 / max(n, 1)


@torch.no_grad()
def evaluate_event_windows(model, dataloader, device, pos_weight,
                           l1_lambda, warmup):
    """Evaluate spike-prediction loss on held-out event windows."""
    model.eval()
    total_loss = 0
    total_spike = 0
    total_l1 = 0
    n = 0

    for pre_sp, post_sp, neuron_ids, is_pos in dataloader:
        pre_sp = pre_sp.to(device)
        post_sp = post_sp.to(device)
        neuron_ids = neuron_ids.to(device)

        window_len = pre_sp.shape[2]
        spike_probs, voltages, weights = model(
            pre_sp, post_sp, neuron_ids, tbptt_len=window_len
        )
        loss, sl, l1l = compute_event_loss(
            spike_probs, post_sp, weights, warmup, pos_weight, l1_lambda
        )

        total_loss += loss.item()
        total_spike += sl
        total_l1 += l1l
        n += 1

    return {
        'loss': total_loss / max(n, 1),
        'spike_loss': total_spike / max(n, 1),
        'l1_loss': total_l1 / max(n, 1),
        'n_batches': n,
        'n_windows': len(dataloader.dataset),
    }


@torch.no_grad()
def evaluate_connectivity(model, neighbor_indices, true_binary, neuron_ids=None):
    """
    Evaluate connectivity prediction from the learned weight matrix.

    Uses |W[j,k]| as the connectivity score for each (post j, pre k) pair.
    """
    model.eval()
    conn_matrix = model.get_connectivity_matrix(neighbor_indices)
    n_neurons = model.n_neurons

    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)

    # Collect scores and labels for the evaluated neurons
    all_scores = []
    all_labels = []

    for j in neuron_ids:
        pre_ids = neighbor_indices[j]
        scores = np.abs(conn_matrix[j, pre_ids])
        labels = true_binary[j, pre_ids].astype(np.float32)
        all_scores.append(scores)
        all_labels.append(labels)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    results = {}
    if len(np.unique(labels)) > 1:
        results['auc'] = roc_auc_score(labels, scores)
        results['ap'] = average_precision_score(labels, scores)

        prec, rec, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        predicted = (scores >= best_thresh).astype(int)
        tp = np.sum((predicted == 1) & (labels == 1))
        fp = np.sum((predicted == 1) & (labels == 0))
        fn = np.sum((predicted == 0) & (labels == 1))

        results['threshold'] = float(best_thresh)
        results['precision'] = tp / (tp + fp + 1e-10)
        results['recall'] = tp / (tp + fn + 1e-10)
        results['f1'] = float(f1[best_idx])
        results['tp'] = int(tp)
        results['fp'] = int(fp)
        results['fn'] = int(fn)
    else:
        results.update({'auc': 0, 'ap': 0, 'f1': 0, 'threshold': 0.5,
                        'precision': 0, 'recall': 0, 'tp': 0, 'fp': 0, 'fn': 0})

    # Weight-true correlation (for true connections only)
    all_w_learned = []
    all_w_true = []
    for j in neuron_ids:
        pre_ids = neighbor_indices[j]
        mask = true_binary[j, pre_ids] == 1
        if mask.any():
            all_w_learned.append(conn_matrix[j, pre_ids[mask]])
            all_w_true.append(true_binary[j, pre_ids[mask]].astype(float))  # just 1s
    # Sign accuracy
    all_w_learned_full = np.concatenate(all_w_learned) if all_w_learned else np.array([])
    if len(all_w_learned_full) > 0:
        # Check if sign matches the true weight sign (exc=positive, inh=negative)
        # We'd need true weights for this, so just report weight stats
        results['mean_connected_weight'] = float(np.mean(np.abs(all_w_learned_full)))
    else:
        results['mean_connected_weight'] = 0.0

    results['n_positive'] = int(labels.sum())
    results['n_total'] = len(labels)

    return results, scores, labels, conn_matrix


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(connectivity_results, scores, labels, conn_matrix,
                 train_losses, val_losses, conn_aucs, val_window_results,
                 neuron_positions, connections, neighbor_indices,
                 model, session_name, output_name, output_dir):

    n_neurons = len(neuron_positions)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Learned LIF Connectivity — {session_name}',
                 fontsize=14, fontweight='bold')

    # ---- Training curve ----
    ax = axes[0, 0]
    ax.plot(train_losses, 'b-', alpha=0.7, label='Train loss')
    ax.plot(val_losses, 'g-', alpha=0.7, label='Val window loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training / Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(conn_aucs, 'r-', alpha=0.7, label='Conn AUC')
    ax2.set_ylabel('Connectivity AUC', color='red')
    ax2.legend(loc='center right')

    # ---- Score distribution ----
    ax = axes[0, 1]
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    ax.hist(neg_scores, bins=50, alpha=0.6, color='red',
            label=f'No conn (n={len(neg_scores)})', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, color='green',
            label=f'Connected (n={len(pos_scores)})', density=True)
    thresh = connectivity_results.get('threshold', 0.5)
    ax.axvline(thresh, color='blue', linewidth=2, label=f'Thresh={thresh:.4f}')
    ax.set_xlabel('|Learned Weight|')
    ax.set_ylabel('Density')
    ax.set_title('Weight Score Distribution')
    ax.legend(fontsize=8)

    # ---- PR curve ----
    ax = axes[0, 2]
    if connectivity_results['auc'] > 0:
        prec, rec, _ = precision_recall_curve(labels, scores)
        ax.plot(rec, prec, 'b-', linewidth=2)
        ax.fill_between(rec, prec, alpha=0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(
        f'PR Curve (AUC={connectivity_results["auc"]:.3f}, '
        f'AP={connectivity_results["ap"]:.3f})'
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ---- True connections ----
    ax = axes[1, 0]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=20, edgecolors='navy', zorder=3)
    for c in connections:
        i, j = int(c[0]), int(c[1])
        true_strength = float(np.clip(abs(float(c[2])) / 0.6, 0.0, 1.0))
        ax.plot([neuron_positions[i, 0], neuron_positions[j, 0]],
                [neuron_positions[i, 1], neuron_positions[j, 1]],
                'g-', alpha=0.18 + 0.22 * true_strength,
                linewidth=0.45 + 0.35 * true_strength)
    ax.set_title(f'True Connections (n={len(connections)})')
    ax.set_aspect('equal')

    # ---- Predicted connections ----
    ax = axes[1, 1]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=20, edgecolors='navy', zorder=3)
    predicted_strengths = np.abs(conn_matrix[np.abs(conn_matrix) >= thresh])
    max_strength = float(predicted_strengths.max()) if predicted_strengths.size else float(thresh)
    for j in range(n_neurons):
        pre_ids = neighbor_indices[j]
        for k, pre in enumerate(pre_ids):
            weight = abs(float(conn_matrix[j, pre]))
            if weight >= thresh:
                denom = max(max_strength - thresh, 1e-8)
                rel_strength = float(np.clip((weight - thresh) / denom, 0.0, 1.0))
                color = 'forestgreen'
                ax.plot([neuron_positions[pre, 0], neuron_positions[j, 0]],
                        [neuron_positions[pre, 1], neuron_positions[j, 1]],
                        color=color,
                        alpha=0.35 + 0.55 * rel_strength,
                        linewidth=0.8 + 1.6 * rel_strength)
    tp = connectivity_results.get('tp', 0)
    fp = connectivity_results.get('fp', 0)
    fn = connectivity_results.get('fn', 0)
    ax.set_title(f'Predicted (TP={tp}, FP={fp}, FN={fn})')
    ax.set_aspect('equal')

    # ---- Summary ----
    ax = axes[1, 2]
    ax.axis('off')
    alpha_val = torch.sigmoid(model.alpha_logit).item()
    tau_eff = -1.0 / np.log(alpha_val + 1e-10)
    summary = f"""
    LEARNED LIF CONNECTIVITY
    {'='*40}

    Network: {session_name}
    Neurons: {n_neurons}

    Learned Membrane Parameters:
      alpha:     {alpha_val:.4f} (tau_m ~ {tau_eff:.1f} ms)
      threshold: {model.threshold.item():.4f}
      beta:      {model.beta.item():.4f}
      reset:     {F.softplus(model.reset_strength).item():.4f}

        Held-out window validation:
            Loss:      {val_window_results.get('loss', 0):.4f}
            Spike:     {val_window_results.get('spike_loss', 0):.4f}
            L1:        {val_window_results.get('l1_loss', 0):.4f}
            Windows:   {val_window_results.get('n_windows', 0)}

        Connectivity (all fitted neurons):
            AUC:       {connectivity_results['auc']:.4f}
            AP:        {connectivity_results['ap']:.4f}
            F1:        {connectivity_results['f1']:.4f}
            Precision: {connectivity_results.get('precision', 0):.4f}
            Recall:    {connectivity_results.get('recall', 0):.4f}
      TP: {tp}  FP: {fp}  FN: {fn}

    Weight stats:
            Mean |w| (connected): {connectivity_results.get('mean_connected_weight', 0):.4f}
            Positives: {connectivity_results['n_positive']}/{connectivity_results['n_total']}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'learned_lif_{output_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {path}")
    return path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_session(session_dir, recording_idx=0, dt=1.0):
    rec_path = os.path.join(session_dir, f'recording{recording_idx:03d}.npz')
    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f"No network file in {session_dir}")
    rec_data = np.load(rec_path, allow_pickle=True)
    net_data = np.load(net_files[0], allow_pickle=True)
    burst_onset_bins = np.array([], dtype=np.int32)
    if 'burst_onset_times' in rec_data.files:
        rec_burst_onsets = np.asarray(rec_data['burst_onset_times'], dtype=float)
        duration = float(rec_data['duration'])
        rec_burst_onsets = rec_burst_onsets[
            (rec_burst_onsets >= 0.0) & (rec_burst_onsets < duration)
        ]
        burst_onset_bins = np.unique((rec_burst_onsets / dt).astype(np.int32))
    return {
        'spike_times': rec_data['spike_times'],
        'duration': float(rec_data['duration']),
        'burst_onset_bins': burst_onset_bins,
        'connections': net_data['connections'],
        'neuron_positions': net_data['neuron_positions'],
        'cluster_assignments': net_data['cluster_assignments'],
        'n_neurons': len(net_data['neuron_positions']),
    }


def plot_recording_raster_with_exclusions(spike_times, cluster_assignments,
                                          duration_ms, excluded_windows=None,
                                          title='Recording Raster',
                                          output_path=None):
    """Plot a single-recording raster and shade excluded windows."""
    if cluster_assignments is None:
        neuron_ids = list(range(len(spike_times)))
    else:
        neuron_ids = sorted(range(len(spike_times)), key=lambda idx: cluster_assignments[idx])

    fig, ax = plt.subplots(figsize=(18, 8))

    if excluded_windows is not None and len(excluded_windows) > 0:
        for window_idx, (start, end) in enumerate(excluded_windows):
            ax.axvspan(
                float(start) / 1000.0,
                float(end) / 1000.0,
                color='tab:red',
                alpha=0.18,
                linewidth=0,
                label='Excluded window' if window_idx == 0 else None,
            )

    for plot_idx, neuron_id in enumerate(neuron_ids):
        spikes = np.asarray(spike_times[neuron_id], dtype=float)
        if spikes.size == 0:
            continue
        ax.scatter(
            spikes / 1000.0,
            np.full(spikes.size, plot_idx, dtype=float),
            s=2,
            c='k',
            marker='|',
            linewidths=0.6,
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron (sorted by cluster)')
    ax.set_title(title)
    ax.set_xlim(0, float(duration_ms) / 1000.0)
    ax.set_ylim(-1, len(neuron_ids))
    if excluded_windows is not None and len(excluded_windows) > 0:
        ax.legend(loc='upper right')

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Raster saved: {output_path}")
    return fig, ax


def run_pipeline(session_dir, K=50, recording_idx=0, n_epochs=100, lr=1e-3,
                 batch_size=64, patience=20, val_fraction=0.2, dt=1.0,
                 max_delay=5, l1_lambda=0.01, pos_weight=5.0,
                 subsample_T=None, device=None, output_tag=None,
                 pre_context=50, post_context=10, warmup=30,
                 neg_ratio=1.0, neg_min_distance=100, use_all_recordings=True,
                 candidate_mode='hybrid', candidate_spatial_frac=0.8,
                 candidate_min_lag=1, candidate_max_lag=None,
                 exclude_detected_bursts=False,
                 burst_activity_bin_ms=100.0,
                 burst_smooth_bins=3,
                 burst_threshold_std=3.0,
                 burst_min_active_fraction=0.10,
                 burst_min_duration_ms=100.0,
                 burst_merge_gap_ms=150.0,
                 burst_pad_before_ms=100.0,
                 burst_pad_after_ms=250.0):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    session_name = os.path.basename(session_dir)
    output_tag = output_tag.strip().replace(' ', '_') if output_tag else None
    output_name = session_name if not output_tag else f'{session_name}_{output_tag}'
    print(f"\n{'='*70}")
    print(f"LEARNED LIF CONNECTIVITY (Event Windows + Per-Neuron Weights)")
    print(f"Session: {session_name}")
    if output_tag:
        print(f"Output tag: {output_tag}")
    print(f"K={K}, epochs={n_epochs}, lr={lr}, max_delay={max_delay}, l1={l1_lambda}")
    print(f"Window: warmup={warmup}, pre={pre_context}, post={post_context} "
          f"({warmup+pre_context+post_context} bins)")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # Load — all recordings concatenated
    print("\n  Loading data...")
    if use_all_recordings:
        data = load_all_recordings(session_dir, dt=dt)
        spike_matrix = data['spike_matrix']
        boundaries = data['boundaries']
        print(f"  Loaded {data['n_recordings']} recordings, "
              f"total duration: {data['total_duration']/1000:.0f}s")
    else:
        single = load_session(session_dir, recording_idx, dt=dt)
        spike_matrix = spike_times_to_binary(
            single['spike_times'], single['duration'], dt
        )
        boundaries = [0, spike_matrix.shape[1]]
        data = {
            'spike_matrix': spike_matrix,
            'boundaries': boundaries,
            'burst_onset_bins': single['burst_onset_bins'],
            'connections': single['connections'],
            'neuron_positions': single['neuron_positions'],
            'cluster_assignments': single['cluster_assignments'],
            'n_neurons': single['n_neurons'],
            'n_recordings': 1,
        }

    n_neurons = data['n_neurons']
    connections = data['connections']
    positions = data['neuron_positions']
    cluster_assignments = data['cluster_assignments']
    saved_burst_onset_bins = np.asarray(
        data.get('burst_onset_bins', np.array([], dtype=np.int32)),
        dtype=np.int32,
    )

    if subsample_T is not None and subsample_T < spike_matrix.shape[1]:
        print(f"  Using first {subsample_T}ms")
        spike_matrix = spike_matrix[:, :subsample_T]
        boundaries = [b for b in boundaries if b <= subsample_T]
        if boundaries[-1] < subsample_T:
            boundaries.append(subsample_T)
        saved_burst_onset_bins = saved_burst_onset_bins[saved_burst_onset_bins < subsample_T]

    detected_burst_info = {
        'windows': np.zeros((0, 2), dtype=np.int32),
        'excluded_bins': np.array([], dtype=np.int32),
        'per_recording_counts': np.zeros(max(0, len(boundaries) - 1), dtype=np.int32),
        'thresholds': np.array([], dtype=np.float32),
    }
    if exclude_detected_bursts:
        detected_burst_info = detect_network_burst_windows(
            spike_matrix,
            boundaries,
            dt_ms=dt,
            activity_bin_ms=burst_activity_bin_ms,
            smooth_bins=burst_smooth_bins,
            threshold_std=burst_threshold_std,
            min_active_fraction=burst_min_active_fraction,
            min_burst_duration_ms=burst_min_duration_ms,
            merge_gap_ms=burst_merge_gap_ms,
            pad_before_ms=burst_pad_before_ms,
            pad_after_ms=burst_pad_after_ms,
        )

    excluded_bins = combine_excluded_bins(
        saved_burst_onset_bins,
        detected_burst_info['excluded_bins'],
    )

    T = spike_matrix.shape[1]
    total_spikes = int(spike_matrix.sum())
    print(f"  Neurons: {n_neurons}, Connections: {len(connections)}")
    print(f"  Spike matrix: [{n_neurons}, {T}] ({total_spikes} total spikes)")
    print(f"  Recording boundaries: {boundaries}")
    if len(saved_burst_onset_bins) > 0:
        print(f"  Saved stimulation onsets: {len(saved_burst_onset_bins)}")
    if exclude_detected_bursts:
        print(f"  Detected burst windows: {len(detected_burst_info['windows'])}")
        print(f"  Detected excluded bins: {len(detected_burst_info['excluded_bins'])}")
        if detected_burst_info['thresholds'].size > 0:
            print(
                f"  Mean burst threshold: {np.mean(detected_burst_info['thresholds']):.3f} "
                f"active fraction per {burst_activity_bin_ms:.0f} ms bin"
            )
    if len(excluded_bins) > 0:
        print(f"  Total excluded bins: {len(excluded_bins)}")

    # Neighbors + ground truth
    candidate_train_boundaries, _ = split_recording_boundaries(boundaries, val_fraction)
    candidate_max_lag = max_delay if candidate_max_lag is None else candidate_max_lag
    neighbor_indices, K_actual, candidate_info = compute_neighbor_indices(
        positions, K,
        spike_matrix=spike_matrix,
        mode=candidate_mode,
        boundaries=candidate_train_boundaries,
        spatial_frac=candidate_spatial_frac,
        excluded_bins=excluded_bins,
        temporal_min_lag=candidate_min_lag,
        temporal_max_lag=candidate_max_lag,
    )
    true_weights, true_binary = build_ground_truth(connections, n_neurons)

    total_in_K = sum(true_binary[j, neighbor_indices[j]].sum() for j in range(n_neurons))
    total_true = int(true_binary.sum())
    print(f"  Candidate mode: {candidate_info['mode']}")
    if candidate_info['mode'] == 'hybrid':
        print(f"  Candidate mix: {candidate_info['n_spatial']} spatial + "
              f"{candidate_info['n_temporal']} temporal "
              f"(lag {candidate_info['temporal_min_lag']}-"
              f"{candidate_info['temporal_max_lag']} bins)")
        if len(excluded_bins) > 0:
            print("  Temporal candidate scoring excludes configured excluded bins")
        print(f"  Mean temporal-only candidates per neuron: "
              f"{candidate_info['mean_temporal_only']:.1f}")
    print(f"  K={K_actual}, coverage: {total_in_K}/{total_true} "
          f"({total_in_K/max(total_true,1):.1%})")

    # Build same-neuron train/validation datasets
    print(f"\n  Extracting event windows...")
    all_neuron_ids = np.arange(n_neurons)
    train_ds, val_ds, validation_strategy = build_train_val_event_datasets(
        spike_matrix, neighbor_indices, all_neuron_ids,
        pre_context=pre_context, post_context=post_context, warmup=warmup,
        neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
        boundaries=boundaries, excluded_bins=excluded_bins,
        val_fraction=val_fraction, rng_seed=42,
    )
    print(f"  Validation strategy: {validation_strategy}")
    print(f"  Train windows: {len(train_ds)} "
          f"({train_ds.n_pos} pos, {train_ds.n_neg} neg)")
    print(f"  Val windows:   {len(val_ds)} "
          f"({val_ds.n_pos} pos, {val_ds.n_neg} neg)")

    if len(train_ds) == 0:
        raise RuntimeError("No training windows extracted. "
                           "Check spike_matrix has enough spikes and window "
                           "parameters fit the recording length.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # Model
    model = PerNeuronLIF(n_neurons=n_neurons, K=K_actual, max_delay=max_delay).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} "
          f"(W: {n_neurons*K_actual:,}, delays: {n_neurons*K_actual*max_delay:,}, "
          f"global: 4)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7)

    # Train
    print(f"\n  Training with event windows...")
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    conn_aucs = []
    t0 = time.time()

    for epoch in range(n_epochs):
        loss, sl, l1l = train_epoch_events(
            model, train_loader, optimizer, device,
            pos_weight, l1_lambda, warmup,
        )
        train_losses.append(loss)

        val_window_results = evaluate_event_windows(
            model, val_loader, device, pos_weight, l1_lambda, warmup,
        )
        val_loss = val_window_results['loss']
        val_losses.append(val_loss)

        conn_results, _, _, _ = evaluate_connectivity(
            model, neighbor_indices, true_binary
        )
        conn_auc = conn_results['auc']
        conn_aucs.append(conn_auc)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            alpha = torch.sigmoid(model.alpha_logit).item()
            thresh = model.threshold.item()
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1:3d}: loss={loss:.4f} (spike={sl:.4f} l1={l1l:.4f}) "
                  f"val_loss={val_loss:.4f} conn_AUC={conn_auc:.4f} "
                  f"alpha={alpha:.3f} thresh={thresh:.3f} "
                  f"({elapsed:.0f}s)")

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Done in {time.time()-t0:.0f}s, best val loss={best_val_loss:.4f}")

    # Final held-out window eval
    val_window_results = evaluate_event_windows(
        model, val_loader, device, pos_weight, l1_lambda, warmup,
    )

    # Connectivity eval on all fitted neurons
    all_results, all_scores, all_labels, conn_matrix = evaluate_connectivity(
        model, neighbor_indices, true_binary
    )

    print(f"\n  {'='*50}")
    print(f"  HELD-OUT WINDOW VALIDATION")
    print(f"  {'='*50}")
    print(f"  Strategy:   {validation_strategy}")
    print(f"  Loss:       {val_window_results['loss']:.4f}")
    print(f"  Spike loss: {val_window_results['spike_loss']:.4f}")
    print(f"  L1 loss:    {val_window_results['l1_loss']:.4f}")
    print(f"  Windows:    {val_window_results['n_windows']}")

    print(f"\n  CONNECTIVITY RESULTS (all fitted neurons)")
    print(f"  AUC:       {all_results['auc']:.4f}")
    print(f"  AP:        {all_results['ap']:.4f}")
    print(f"  F1:        {all_results['f1']:.4f}")

    alpha = torch.sigmoid(model.alpha_logit).item()
    print(f"\n  Learned: alpha={alpha:.4f} (tau_m~{-1/np.log(alpha+1e-10):.1f}ms), "
          f"threshold={model.threshold.item():.4f}, beta={model.beta.item():.4f}")

    # Visualize
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'learned_lif_outputs')
    plot_results(all_results, all_scores, all_labels, conn_matrix,
                 train_losses, val_losses, conn_aucs, val_window_results,
                 positions, connections, neighbor_indices,
                 model, session_name, output_name, output_dir)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'learned_lif_{output_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'K': K_actual, 'T': T, 'dt': dt,
        'max_delay': max_delay,
        'n_neurons': n_neurons,
        'session_name': session_name,
        'output_name': output_name,
        'validation_strategy': validation_strategy,
        'candidate_info': candidate_info,
        'saved_burst_onset_bins': saved_burst_onset_bins,
        'detected_burst_windows': detected_burst_info['windows'],
        'excluded_bins': excluded_bins,
        'neighbor_indices': neighbor_indices,
        'connectivity_matrix': conn_matrix,
        'results_window_val': val_window_results,
        'results_val': all_results,
        'results_all': all_results,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': conn_aucs,
        'connectivity_aucs': conn_aucs,
    }, model_path)
    print(f"  Model + connectivity saved: {model_path}")

    # Save connectivity matrix as separate .npz for easy access
    conn_path = os.path.join(output_dir, f'connectivity_{output_name}.npz')
    np.savez_compressed(
        conn_path,
        connectivity_matrix=conn_matrix,
        threshold=all_results.get('threshold', 0.5),
        neighbor_indices=neighbor_indices,
        neuron_positions=positions,
        cluster_assignments=cluster_assignments,
        saved_burst_onset_bins=saved_burst_onset_bins,
        detected_burst_windows=detected_burst_info['windows'],
        excluded_bins=excluded_bins,
    )
    print(f"  Connectivity matrix saved: {conn_path}")

    return all_results, conn_matrix


# ============================================================================
# CLI
# ============================================================================

def select_session():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIF data")
    sessions = sorted([s for s in glob.glob(os.path.join(data_dir, "*"))
                       if os.path.isdir(s)])
    if not sessions:
        print("No sessions in LIF data/"); sys.exit(1)
    print("\nSessions:")
    for i, s in enumerate(sessions):
        print(f"  [{i}] {os.path.basename(s)}")
    choice = input("Select (Enter=first): ").strip()
    return sessions[0] if choice == '' else sessions[int(choice)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learned LIF Connectivity')
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional suffix for saved artifact names')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=128, help='Windows per batch')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--max-delay', type=int, default=8)
    parser.add_argument('--l1', type=float, default=0.01, help='L1 sparsity on weights')
    parser.add_argument('--pos-weight', type=float, default=5.0, help='BCE positive class weight')
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--recording', type=int, default=0, help='Single recording index (if not using all)')
    parser.add_argument('--single-recording', action='store_true',
                        help='Use only one recording instead of all')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Use only first N ms (for faster testing)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='Device (default: cpu)')
    parser.add_argument('--candidate-mode', type=str, default='hybrid',
                        choices=['spatial', 'hybrid'],
                        help='Candidate proposal mode for presynaptic neighbors')
    parser.add_argument('--candidate-spatial-frac', type=float, default=0.8,
                        help='In hybrid mode, fraction of K reserved for spatial neighbors')
    parser.add_argument('--candidate-min-lag', type=int, default=1,
                        help='In hybrid mode, minimum causal lag in bins')
    parser.add_argument('--candidate-max-lag', type=int, default=None,
                        help='In hybrid mode, maximum causal lag in bins (default: use --max-delay)')
    # Event window parameters
    parser.add_argument('--pre-context', type=int, default=50,
                        help='Bins before event (causal pre input)')
    parser.add_argument('--post-context', type=int, default=10,
                        help='Bins after event (spike + reset dynamics)')
    parser.add_argument('--warmup', type=int, default=30,
                        help='Warmup bins before loss region (membrane settling)')
    parser.add_argument('--neg-ratio', type=float, default=1.0,
                        help='Negative windows per positive window')
    parser.add_argument('--neg-min-dist', type=int, default=100,
                        help='Min distance (bins) from any post spike for negatives')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Validation fraction: held-out recordings when possible, otherwise held-out windows')
    parser.add_argument('--exclude-detected-bursts', action='store_true',
                        help='Detect network burst windows and exclude them from candidate scoring and event windows')
    parser.add_argument('--burst-activity-bin-ms', type=float, default=100.0,
                        help='Burst detection activity bin width in ms')
    parser.add_argument('--burst-smooth-bins', type=int, default=3,
                        help='Burst detection smoothing width in activity bins')
    parser.add_argument('--burst-threshold-std', type=float, default=3.0,
                        help='Burst detection threshold = mean + std_factor * std')
    parser.add_argument('--burst-min-active-frac', type=float, default=0.10,
                        help='Minimum active-neuron fraction for burst detection')
    parser.add_argument('--burst-min-duration-ms', type=float, default=100.0,
                        help='Minimum detected burst duration in ms')
    parser.add_argument('--burst-merge-gap-ms', type=float, default=150.0,
                        help='Merge nearby burst segments separated by at most this gap')
    parser.add_argument('--burst-pad-before-ms', type=float, default=100.0,
                        help='Padding before each detected burst window in ms')
    parser.add_argument('--burst-pad-after-ms', type=float, default=250.0,
                        help='Padding after each detected burst window in ms')
    args = parser.parse_args()

    session_dir = args.session if args.session else select_session()

    run_pipeline(
        session_dir, K=args.k, recording_idx=args.recording,
        n_epochs=args.epochs, lr=args.lr, batch_size=args.batch,
        patience=args.patience, dt=args.dt, max_delay=args.max_delay,
        l1_lambda=args.l1, pos_weight=args.pos_weight,
        val_fraction=args.val_fraction, output_tag=args.output_tag,
        subsample_T=args.subsample, device=args.device,
        pre_context=args.pre_context, post_context=args.post_context,
        warmup=args.warmup, neg_ratio=args.neg_ratio,
        neg_min_distance=args.neg_min_dist,
        use_all_recordings=not args.single_recording,
        candidate_mode=args.candidate_mode,
        candidate_spatial_frac=args.candidate_spatial_frac,
        candidate_min_lag=args.candidate_min_lag,
        candidate_max_lag=args.candidate_max_lag,
        exclude_detected_bursts=args.exclude_detected_bursts,
        burst_activity_bin_ms=args.burst_activity_bin_ms,
        burst_smooth_bins=args.burst_smooth_bins,
        burst_threshold_std=args.burst_threshold_std,
        burst_min_active_fraction=args.burst_min_active_frac,
        burst_min_duration_ms=args.burst_min_duration_ms,
        burst_merge_gap_ms=args.burst_merge_gap_ms,
        burst_pad_before_ms=args.burst_pad_before_ms,
        burst_pad_after_ms=args.burst_pad_after_ms,
    )
