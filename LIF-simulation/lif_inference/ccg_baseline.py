"""Cross-correlogram (CCG) baseline for connectivity inference.

A fast, training-free baseline that scores each candidate edge by the excess of
short-latency presynaptic-before-postsynaptic coincidences above chance. It
produces a ``scores[post, pre]`` matrix in the same format as
``PerNeuronLIF.get_connectivity_matrix`` and reuses the package's
burst-exclusion, surrogate, thresholding, and metric utilities, so it can be
benchmarked head-to-head against the learned model.
"""

import numpy as np

from .candidate_selection import build_recording_spike_indices
from .shared_data import build_segmentwise_circular_shift_surrogates
from .connectivity_metrics import (
    flatten_candidate_scores,
    compute_binary_classification_metrics,
    select_connectivity_threshold,
)


def compute_ccg_scores(spike_matrix, boundaries=None, excluded_bins=None,
                       candidate_indices=None, min_lag=1, max_lag=4,
                       baseline_lo=15, baseline_hi=25, normalize=True):
    """Score every candidate edge by its short-latency coincidence excess.

    For each ``(post, pre)`` pair, count the presynaptic spikes that fall
    ``min_lag..max_lag`` bins *before* each postsynaptic spike (the excitatory
    monosynaptic window), subtract the chance level estimated from a far flank
    window ``baseline_lo..baseline_hi`` bins before each postsynaptic spike
    (scaled to the synaptic window width), and optionally normalize by
    ``sqrt(pre_count * post_count)``.

    Args:
        spike_matrix: Binary ``[n_neurons, T]`` spike matrix.
        boundaries: Optional recording-segment boundaries; counting never spans
            segments.
        excluded_bins: Optional bin indices to drop (e.g. burst windows).
        candidate_indices: Optional per-neuron arrays of presynaptic candidates;
            when ``None`` every co-active neuron is considered.
        min_lag: Smallest presynaptic lead (in bins) of the synaptic window.
        max_lag: Largest presynaptic lead (in bins) of the synaptic window.
        baseline_lo: Smallest presynaptic lead of the far flank baseline window.
        baseline_hi: Largest presynaptic lead of the far flank baseline window.
        normalize: Whether to divide by ``sqrt(pre_count * post_count)``.

    Returns:
        A ``[n_neurons, n_neurons]`` ``scores[post, pre]`` matrix with the
        diagonal set to ``-inf``.
    """
    n_neurons, _ = spike_matrix.shape
    spikes_by_rec, counts = build_recording_spike_indices(
        spike_matrix, boundaries=boundaries, excluded_bins=excluded_bins)
    syn_w = float(max_lag - min_lag + 1)
    base_w = float(baseline_hi - baseline_lo + 1)
    scores = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    for rec_spikes in spikes_by_rec:
        active = set(i for i, s in enumerate(rec_spikes) if len(s) > 0)
        for post_id in active:
            post_sp = rec_spikes[post_id]
            pres = candidate_indices[post_id] if candidate_indices is not None else active
            for pre_id in pres:
                pre_id = int(pre_id)
                if pre_id == post_id or pre_id not in active:
                    continue
                pre_sp = rec_spikes[pre_id]
                syn = int((np.searchsorted(pre_sp, post_sp - min_lag, 'right')
                           - np.searchsorted(pre_sp, post_sp - max_lag, 'left')).sum())
                base = int((np.searchsorted(pre_sp, post_sp - baseline_lo, 'right')
                            - np.searchsorted(pre_sp, post_sp - baseline_hi, 'left')).sum())
                scores[post_id, pre_id] += syn - (syn_w / base_w) * base
    if normalize:
        norm = np.sqrt(np.outer(counts, counts))
        valid = norm > 0
        scores[valid] = scores[valid] / norm[valid]
    np.fill_diagonal(scores, -np.inf)
    return scores


def compute_ccg_surrogate_score_sets(spike_matrix, neighbor_indices, n_surrogates=4,
                                     boundaries=None, excluded_bins=None, neuron_ids=None,
                                     min_shift_fraction=0.10, seed=1234, **ccg_kwargs):
    """Build a CCG null distribution from segment-wise circular-shift surrogates.

    Each surrogate circularly shifts every neuron independently within each
    recording segment (destroying cross-pair timing while preserving per-neuron
    rate), recomputes the CCG scores over the candidate edges, and flattens them.
    The stacked sets feed surrogate-FDR thresholding.

    Args:
        spike_matrix: Binary ``[n_neurons, T]`` spike matrix.
        neighbor_indices: Per-neuron arrays of presynaptic candidates to score.
        n_surrogates: Number of surrogate shuffles to generate.
        boundaries: Optional recording-segment boundaries.
        excluded_bins: Optional bin indices to drop.
        neuron_ids: Optional postsynaptic neuron ids to flatten over.
        min_shift_fraction: Minimum circular shift as a fraction of segment length.
        seed: Seed for the surrogate random generator.
        **ccg_kwargs: Forwarded to ``compute_ccg_scores`` (window settings, etc.).

    Returns:
        A ``[n_surrogates, n_candidate_edges]`` array of flattened surrogate
        scores (absolute valued).
    """
    rng = np.random.default_rng(seed)
    score_sets = []
    for _ in range(int(n_surrogates)):
        surrogate_matrix, = build_segmentwise_circular_shift_surrogates(
            [spike_matrix], boundaries=boundaries, rng=rng, min_shift_fraction=min_shift_fraction)
        surrogate_scores = compute_ccg_scores(
            surrogate_matrix, boundaries=boundaries, excluded_bins=excluded_bins,
            candidate_indices=neighbor_indices, **ccg_kwargs)
        score_sets.append(flatten_candidate_scores(
            surrogate_scores, neighbor_indices, neuron_ids=neuron_ids, absolute=True))
    return np.stack(score_sets, axis=0)


def run_ccg_baseline(spike_matrix, neighbor_indices, neuron_ids=None, true_binary=None,
                     boundaries=None, excluded_bins=None, threshold_mode='surrogate_fdr',
                     surrogate_fdr=0.01, n_surrogates=4, default_threshold=0.0, **ccg_kwargs):
    """Run the CCG baseline end to end and report metrics comparable to the model.

    Scores the candidate edges, flattens them, chooses a cutoff
    (surrogate-FDR by default; ``oracle_f1`` requires ``true_binary``), predicts
    edges, and optionally computes classification metrics against ground truth.

    Args:
        spike_matrix: Binary ``[n_neurons, T]`` spike matrix.
        neighbor_indices: Per-neuron arrays of presynaptic candidates to score.
        neuron_ids: Optional postsynaptic neuron ids to flatten over.
        true_binary: Optional ground-truth ``[post, pre]`` binary matrix; enables
            labels, ``oracle_f1`` thresholding, and metrics.
        boundaries: Optional recording-segment boundaries.
        excluded_bins: Optional bin indices to drop.
        threshold_mode: Thresholding mode (``'surrogate_fdr'`` or ``'oracle_f1'``).
        surrogate_fdr: Target false-discovery rate for surrogate-FDR thresholding.
        n_surrogates: Number of surrogate shuffles for the null distribution.
        default_threshold: Fallback cutoff when a mode cannot resolve one.
        **ccg_kwargs: Forwarded to ``compute_ccg_scores`` (window settings, etc.).

    Returns:
        A dictionary with the ``conn_matrix``, flattened ``scores``, chosen
        ``threshold``, ``predicted`` edges, ``threshold_info``, and (when
        ``true_binary`` is given) ``labels`` and ``metrics``.
    """
    conn_matrix = compute_ccg_scores(spike_matrix, boundaries=boundaries,
        excluded_bins=excluded_bins, candidate_indices=neighbor_indices, **ccg_kwargs)
    scores = flatten_candidate_scores(conn_matrix, neighbor_indices,
        neuron_ids=neuron_ids, absolute=True)
    labels = None
    if true_binary is not None:
        labels = (flatten_candidate_scores(true_binary, neighbor_indices,
            neuron_ids=neuron_ids, absolute=True) > 0).astype(np.int32)
    surrogate_score_sets = None
    if threshold_mode == 'surrogate_fdr':
        surrogate_score_sets = compute_ccg_surrogate_score_sets(spike_matrix, neighbor_indices,
            n_surrogates=n_surrogates, boundaries=boundaries, excluded_bins=excluded_bins,
            neuron_ids=neuron_ids, **ccg_kwargs)
    labels_for_threshold = labels if labels is not None else np.zeros(scores.shape[0], dtype=np.int32)
    threshold_info = select_connectivity_threshold(labels_for_threshold, scores,
        mode=threshold_mode, surrogate_score_sets=surrogate_score_sets,
        surrogate_fdr=surrogate_fdr, default_threshold=default_threshold)
    threshold = float(threshold_info['threshold'])
    predicted = (scores >= threshold).astype(np.int32)
    result = {'conn_matrix': conn_matrix, 'scores': scores, 'threshold': threshold,
              'predicted': predicted, 'threshold_info': threshold_info,
              'surrogate_score_sets': surrogate_score_sets}
    if labels is not None:
        result['labels'] = labels
        result['metrics'] = compute_binary_classification_metrics(labels, predicted)
    return result
