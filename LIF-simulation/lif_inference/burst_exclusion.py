"""Burst-detection and exclusion-window helpers for learned-LIF inference."""

import numpy as np


def merge_excluded_windows(excluded_windows, max_gap_bins=0):
    """Merge overlapping or nearby ``[start, end)`` exclusion windows."""
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
    """Return contiguous ``[start, end)`` segments where a mask is true."""
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
    """Detect network-wide burst windows from population synchrony."""
    n_neurons, total_bins = spike_matrix.shape
    if recording_boundaries is None:
        recording_boundaries = [0, total_bins]

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
    """Check whether a candidate window overlaps any excluded bin."""
    if excluded_bins is None or len(excluded_bins) == 0:
        return False

    idx = np.searchsorted(excluded_bins, start, side='left')
    return idx < len(excluded_bins) and int(excluded_bins[idx]) < end