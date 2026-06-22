"""
Shared data utilities for learned-LIF inference.

Provides core spike-time conversion, ground-truth construction, and
surrogate generation helpers used by both spike-only and voltage-augmented
inference pipelines.
"""

import numpy as np


def spike_times_to_binary(spike_times, duration_ms, dt=1.0):
    """Convert per-neuron spike times into a binary spike matrix.

    Args:
        spike_times: Sequence of per-neuron spike-time arrays in milliseconds.
        duration_ms: Recording duration in milliseconds.
        dt: Target bin width in milliseconds.

    Returns:
        A binary matrix with shape ``[n_neurons, T]`` sampled at ``dt`` resolution.
    """
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
    """Build dense ground-truth weight and connectivity matrices.

    Args:
        connections: Connection table whose rows encode presynaptic id,
            postsynaptic id, and synaptic weight.
        n_neurons: Total number of neurons in the network.

    Returns:
        A tuple ``(W, B)`` where ``W[post, pre]`` stores the signed weight and
        ``B[post, pre]`` stores a binary connectivity flag.
    """
    W = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    B = np.zeros((n_neurons, n_neurons), dtype=np.int32)
    for c in connections:
        pre, post = int(c[0]), int(c[1])
        W[post, pre] = float(c[2])
        B[post, pre] = 1
    return W, B


def normalize_recording_boundaries(boundaries, total_length):
    """Validate recording boundaries and return a normalized integer array."""
    if boundaries is None:
        return np.array([0, int(total_length)], dtype=np.int32)

    arr = np.asarray(boundaries, dtype=np.int64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError('boundaries must be a one-dimensional array with at least two entries')
    if int(arr[0]) != 0 or int(arr[-1]) != int(total_length):
        raise ValueError(
            f'boundaries must start at 0 and end at total_length={int(total_length)}; '
            f'got {arr[0]}..{arr[-1]}'
        )
    if np.any(np.diff(arr) <= 0):
        raise ValueError('boundaries must be strictly increasing')
    return arr.astype(np.int32, copy=False)


def build_segmentwise_circular_shift_surrogates(arrays, boundaries=None, rng=None,
                                                min_shift_fraction=0.10):
    """Circularly shift aligned per-neuron traces within each recording segment."""
    arrays = [np.asarray(array) for array in arrays]
    if not arrays:
        return []

    base_shape = arrays[0].shape
    if len(base_shape) != 2:
        raise ValueError('surrogate shifting expects 2D [n_neurons, T] arrays')
    for array in arrays[1:]:
        if array.shape != base_shape:
            raise ValueError('all arrays must share the same shape for surrogate shifting')

    n_neurons, total_length = base_shape
    boundaries = normalize_recording_boundaries(boundaries, total_length)
    rng = np.random.default_rng() if rng is None else rng
    shifted_arrays = [np.empty_like(array) for array in arrays]

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_len = int(end - start)
        if segment_len <= 1:
            for shifted, array in zip(shifted_arrays, arrays):
                shifted[:, start:end] = array[:, start:end]
            continue

        min_shift = max(int(np.floor(segment_len * float(min_shift_fraction))), 1)
        min_shift = min(min_shift, segment_len - 1)
        shifts = rng.integers(min_shift, segment_len, size=n_neurons)

        for neuron_id, shift in enumerate(shifts):
            lo = int(start)
            hi = int(end)
            shift = int(shift)
            for shifted, array in zip(shifted_arrays, arrays):
                shifted[neuron_id, lo:hi] = np.roll(array[neuron_id, lo:hi], shift)

    return shifted_arrays


def _interval_jitter_one(spike_matrix, rng, jitter_bins, segments):
    """Resample each neuron's spikes uniformly within consecutive windows of
    width ``jitter_bins`` bins, independently per neuron, within each segment.

    Preserves per-neuron spike counts exactly; destroys structure finer than the
    window (e.g. monosynaptic lags) while preserving structure coarser than the
    window (population bursts / common input). ``segments`` is a list of
    ``(start, end)`` half-open recording spans, so spikes never cross a boundary.
    """
    sm = np.asarray(spike_matrix)
    N, T = sm.shape
    W = max(int(jitter_bins), 1)
    out = np.zeros_like(sm)
    for s, e in segments:
        for w0 in range(s, e, W):
            w1 = min(w0 + W, e)
            L = w1 - w0
            counts = sm[:, w0:w1].sum(axis=1).astype(int)
            for i in np.nonzero(counts)[0]:
                c = min(int(counts[i]), L)       # binary trains: c <= L
                out[i, w0 + rng.choice(L, size=c, replace=False)] = sm.dtype.type(1)
    return out


def build_interval_jitter_surrogates(spike_matrices, boundaries=None, rng=None,
                                     jitter_bins=25, **_ignored):
    """Interval-jitter surrogate null for connectivity-threshold calibration.

    Resamples each neuron's spikes uniformly within consecutive windows of
    ``jitter_bins`` bins (independently per neuron, within each recording
    segment). This destroys fine timing below the window (monosynaptic lags,
    which sit at ``max_delay`` ~ a handful of bins) while preserving co-activation
    above the window (population bursts, tens-to-hundreds of ms) -- i.e. it keeps
    the common-input structure that drives the voltage model's false positives,
    unlike circular shift, which rotates each neuron rigidly and destroys both.

    Signature- and return-compatible with
    :func:`build_segmentwise_circular_shift_surrogates`: takes a list of aligned
    2D ``[n_neurons, T]`` matrices, returns a list of jittered matrices (one per
    input), and absorbs extra kwargs (e.g. ``min_shift_fraction``) via
    ``**_ignored`` so callers can swap builders without changing their call site.
    Intended for binary spike matrices; per-neuron spike counts are preserved
    exactly.
    """
    spike_matrices = [np.asarray(array) for array in spike_matrices]
    if not spike_matrices:
        return []

    base_shape = spike_matrices[0].shape
    if len(base_shape) != 2:
        raise ValueError('surrogate jitter expects 2D [n_neurons, T] arrays')
    for array in spike_matrices[1:]:
        if array.shape != base_shape:
            raise ValueError('all arrays must share the same shape for surrogate jitter')

    _, total_length = base_shape
    boundaries = normalize_recording_boundaries(boundaries, total_length)
    rng = np.random.default_rng() if rng is None else rng
    segments = [(int(start), int(end))
                for start, end in zip(boundaries[:-1], boundaries[1:])]

    return [_interval_jitter_one(array, rng, jitter_bins, segments)
            for array in spike_matrices]


def validate_jitter_null(observed_scores, observed_labels,
                         null_score_sets_by_window, nonedge_percentile=95.0):
    """Diagnose interval-jitter nulls of different window widths.

    Compares each candidate null (keyed by ``jitter_bins``) against the observed
    candidate scores from a trained model. Model-agnostic: pass observed scores
    and labels from ``evaluate_connectivity`` (spike-only or voltage), and null
    score sets flattened in the same candidate order.

    Args:
        observed_scores: 1D ``|conn|`` over candidate edges.
        observed_labels: 1D 0/1 labels aligned to ``observed_scores`` (1 = true edge).
        null_score_sets_by_window: dict ``{jitter_bins: null_score_sets}`` where each
            value is a ``[n_surrogates, n_edges]`` (or 1D) array aligned to
            ``observed_scores``.
        nonedge_percentile: percentile of the non-edge score floor compared between
            null and observed (default 95).

    Returns:
        A list of per-window dicts sorted by ``jitter_bins`` with:
          * ``power`` = median(null at true-edge candidates) / median(observed at
            true-edge candidates) -- want small (true edges collapse under the null).
          * ``calibration`` = (null non-edge p95) / (observed non-edge p95) -- want
            ~1.0 (the null FP floor matches the real FP floor; <1 means the cutoff
            is still too loose, >1 means over-conservative).
        The right ``jitter_bins`` is the largest window where ``power`` is still
        small and ``calibration`` is near 1.
    """
    obs = np.asarray(observed_scores, dtype=np.float64)
    labels = np.asarray(observed_labels).astype(int)
    true_mask = labels == 1
    nonedge_mask = labels == 0
    if not np.any(true_mask) or not np.any(nonedge_mask):
        raise ValueError('validate_jitter_null needs both true-edge and non-edge candidates')

    obs_true_median = float(np.median(obs[true_mask]))
    obs_nonedge_floor = float(np.percentile(obs[nonedge_mask], nonedge_percentile))

    rows = []
    for jitter_bins in sorted(null_score_sets_by_window):
        null = np.asarray(null_score_sets_by_window[jitter_bins], dtype=np.float64)
        if null.ndim == 1:
            null = null[None, :]
        if null.shape[1] != obs.shape[0]:
            raise ValueError(
                f'null scores for jitter_bins={jitter_bins} are not aligned with '
                f'observed scores ({null.shape[1]} vs {obs.shape[0]})'
            )
        null_true_median = float(np.median(null[:, true_mask]))
        null_nonedge_floor = float(np.percentile(null[:, nonedge_mask], nonedge_percentile))
        power = null_true_median / obs_true_median if obs_true_median > 0 else float('inf')
        calibration = null_nonedge_floor / obs_nonedge_floor if obs_nonedge_floor > 0 else float('inf')
        rows.append({
            'jitter_bins': int(jitter_bins),
            'power': float(power),
            'calibration': float(calibration),
            'null_true_median': null_true_median,
            'obs_true_median': obs_true_median,
            'null_nonedge_p95': null_nonedge_floor,
            'obs_nonedge_p95': obs_nonedge_floor,
        })
    return rows
