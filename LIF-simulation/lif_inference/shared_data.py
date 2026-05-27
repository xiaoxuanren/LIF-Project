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
