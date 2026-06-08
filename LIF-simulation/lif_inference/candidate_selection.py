"""Candidate presynaptic selection helpers for learned-LIF inference."""

import numpy as np


def compute_spatial_neighbor_indices(neuron_positions, K):
    """Find the nearest spatial candidate presynaptic neurons for each neuron."""
    n = len(neuron_positions)
    dist = np.sqrt(((neuron_positions[:, None, :] -
                     neuron_positions[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    K_actual = min(K, n - 1)
    indices = np.argsort(dist, axis=1)[:, :K_actual]
    return indices, K_actual, dist


def build_recording_spike_indices(spike_matrix, boundaries=None, excluded_bins=None):
    """Build per-recording spike index lists for each neuron."""
    n_neurons, total_bins = spike_matrix.shape
    if boundaries is None:
        boundaries = [0, total_bins]

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
    """Score a candidate edge from short-latency pre-before-post spike pairs."""
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
    """Compute causal temporal candidate scores from spike timing alone."""
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
                             spatial_frac=0.8, excluded_bins=None,
                             temporal_min_lag=1, temporal_max_lag=8):
    """Build candidate presynaptic sets for each postsynaptic neuron."""
    spatial_indices, K_actual, _ = compute_spatial_neighbor_indices(
        neuron_positions, K,
    )
    mode = str(mode).strip().lower()
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