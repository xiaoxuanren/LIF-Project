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
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .connectivity_metrics import (
    evaluate_connectivity as shared_evaluate_connectivity,
)
from .event_windows import (
    EventWindowDataset as shared_EventWindowDataset,
    build_train_val_event_datasets as shared_build_train_val_event_datasets,
    find_event_windows as shared_find_event_windows,
    find_pre_centered_event_windows as shared_find_pre_centered_event_windows,
    split_event_windows as shared_split_event_windows,
    split_recording_boundaries as shared_split_recording_boundaries,
)
from .event_training import (
    compute_event_loss as shared_compute_event_loss,
    evaluate_event_windows as shared_evaluate_event_windows,
    train_epoch_events as shared_train_epoch_events,
)
from .surrogate_thresholding import (
    estimate_surrogate_connectivity_score_sets as shared_estimate_surrogate_connectivity_score_sets,
)
from .candidate_selection import (
    build_recording_spike_indices as shared_build_recording_spike_indices,
    causal_pair_score as shared_causal_pair_score,
    compute_neighbor_indices as shared_compute_neighbor_indices,
    compute_spatial_neighbor_indices as shared_compute_spatial_neighbor_indices,
    compute_temporal_candidate_scores as shared_compute_temporal_candidate_scores,
)
from .burst_exclusion import (
    _find_true_segments as shared_find_true_segments,
    combine_excluded_bins as shared_combine_excluded_bins,
    detect_network_burst_windows as shared_detect_network_burst_windows,
    excluded_windows_to_bins as shared_excluded_windows_to_bins,
    merge_excluded_windows as shared_merge_excluded_windows,
    window_overlaps_excluded_bins as shared_window_overlaps_excluded_bins,
)
from .shared_data import (
    spike_times_to_binary,
    build_ground_truth,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# ============================================================================
# CONNECTIVITY METRICS & THRESHOLDING
# ============================================================================

# Shared connectivity evaluation helpers live in connectivity_metrics.py so the
# spike-only and voltage-augmented paths can share one implementation surface.


def compute_spatial_neighbor_indices(neuron_positions, K):
    """Find the nearest spatial candidate presynaptic neurons for each neuron.

    Args:
        neuron_positions: Array of neuron coordinates with shape ``[n_neurons, 2]``.
        K: Requested number of spatial neighbors per neuron.

    Returns:
        A tuple ``(indices, K_actual, dist)`` containing the nearest-neighbor
        indices, the usable neighbor count after clipping, and the full distance
        matrix.
    """
    return shared_compute_spatial_neighbor_indices(neuron_positions, K)


def build_recording_spike_indices(spike_matrix, boundaries=None, excluded_bins=None):
    """Build per-recording spike index lists for each neuron.

    Args:
        spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
        boundaries: Optional concatenated recording boundaries.
        excluded_bins: Optional sorted bins to ignore when extracting spikes.

    Returns:
        A tuple ``(spike_indices, spike_counts)`` containing per-recording spike
        indices for each neuron and total retained spike counts per neuron.
    """
    return shared_build_recording_spike_indices(
        spike_matrix,
        boundaries=boundaries,
        excluded_bins=excluded_bins,
    )


def causal_pair_score(pre_spikes, post_spikes, min_lag=1, max_lag=8):
    """Score a candidate edge from short-latency pre-before-post spike pairs.

    Args:
        pre_spikes: Sorted presynaptic spike-bin indices.
        post_spikes: Sorted postsynaptic spike-bin indices.
        min_lag: Minimum causal lag considered valid.
        max_lag: Maximum causal lag considered valid.

    Returns:
        A nonnegative temporal-causality score where shorter valid lags contribute
        more weight.
    """
    return shared_causal_pair_score(
        pre_spikes,
        post_spikes,
        min_lag=min_lag,
        max_lag=max_lag,
    )


def compute_temporal_candidate_scores(spike_matrix, boundaries=None,
                                      min_lag=1, max_lag=8,
                                      excluded_bins=None):
    """Compute causal temporal candidate scores from spike timing alone.

    Args:
        spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
        boundaries: Optional concatenated recording boundaries.
        min_lag: Minimum causal lag considered valid.
        max_lag: Maximum causal lag considered valid.
        excluded_bins: Optional sorted bins excluded from temporal scoring.

    Returns:
        A ``[n_neurons, n_neurons]`` score matrix where larger values indicate
        stronger causal spike-timing support for a presynaptic candidate.
    """
    return shared_compute_temporal_candidate_scores(
        spike_matrix,
        boundaries=boundaries,
        min_lag=min_lag,
        max_lag=max_lag,
        excluded_bins=excluded_bins,
    )


def compute_neighbor_indices(neuron_positions, K, spike_matrix=None,
                             mode='spatial', boundaries=None,
                             spatial_frac=0.8, excluded_bins=None,
                             temporal_min_lag=1, temporal_max_lag=8):
    """Build candidate presynaptic sets for each postsynaptic neuron.

    Args:
        neuron_positions: Array of neuron coordinates with shape ``[n_neurons, 2]``.
        K: Requested number of candidate presynaptic neurons per postsynaptic neuron.
        spike_matrix: Optional binary spike matrix used for temporal candidate scoring.
        mode: Candidate-selection mode, typically ``spatial`` or ``hybrid``.
        boundaries: Optional concatenated recording boundaries.
        spatial_frac: In hybrid mode, fraction of candidates reserved for spatial neighbors.
        excluded_bins: Optional sorted bins excluded from temporal candidate scoring.
        temporal_min_lag: Minimum causal lag used by temporal scoring.
        temporal_max_lag: Maximum causal lag used by temporal scoring.

    Returns:
        A tuple ``(neighbor_indices, K_actual, info)`` containing the candidate
        matrix, the usable candidate count, and metadata describing how the
        candidate set was built.
    """
    return shared_compute_neighbor_indices(
        neuron_positions,
        K,
        spike_matrix=spike_matrix,
        mode=mode,
        boundaries=boundaries,
        spatial_frac=spatial_frac,
        excluded_bins=excluded_bins,
        temporal_min_lag=temporal_min_lag,
        temporal_max_lag=temporal_max_lag,
    )


def load_all_recordings(session_dir, dt=1.0):
    """
    Load and concatenate ALL recordings in a session along the time axis.

    Args:
        session_dir: Session directory containing recording files and one network file.
        dt: Bin width in milliseconds used to convert spike times into binary traces.

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

        # Carry stimulation onsets forward so downstream fitting can exclude obviously driven bins.
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
    """Merge overlapping or nearby ``[start, end)`` exclusion windows.

    Args:
        excluded_windows: Iterable of exclusion windows in bin coordinates.
        max_gap_bins: Maximum bin gap allowed when merging adjacent windows.

    Returns:
        A ``[n_windows, 2]`` array of merged exclusion windows.
    """
    return shared_merge_excluded_windows(excluded_windows, max_gap_bins=max_gap_bins)


def excluded_windows_to_bins(excluded_windows):
    """Expand exclusion windows into sorted bin indices.

    Args:
        excluded_windows: Iterable of exclusion windows in bin coordinates.

    Returns:
        A sorted one-dimensional array containing every excluded bin index.
    """
    return shared_excluded_windows_to_bins(excluded_windows)


def combine_excluded_bins(*bin_arrays):
    """Combine multiple exclusion-bin sources into one sorted unique array.

    Args:
        *bin_arrays: Any number of one-dimensional bin-index arrays.

    Returns:
        A sorted unique array containing the union of all supplied bin indices.
    """
    return shared_combine_excluded_bins(*bin_arrays)


def _find_true_segments(mask):
    """Return contiguous ``[start, end)`` segments where a mask is true.

    Args:
        mask: One-dimensional boolean array.

    Returns:
        A list of half-open index segments where ``mask`` remains true.
    """
    return shared_find_true_segments(mask)


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
    """Detect network-wide burst windows from population synchrony.

    Args:
        spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
        recording_boundaries: Concatenated recording boundaries.
        dt_ms: Spike-bin width in milliseconds.
        activity_bin_ms: Width of the coarse activity bins in milliseconds.
        smooth_bins: Width of the moving-average smoothing kernel in bins.
        threshold_std: Threshold scale factor used in ``mean + std_factor * std``.
        min_active_fraction: Minimum active-neuron fraction required for a burst.
        min_burst_duration_ms: Minimum detected burst duration in milliseconds.
        merge_gap_ms: Maximum allowed gap when merging nearby burst windows.
        pad_before_ms: Padding added before each detected burst window.
        pad_after_ms: Padding added after each detected burst window.

    Returns:
        A dictionary containing the detected windows, excluded bins, per-recording
        burst counts, and the thresholding parameters used to find them.
    """
    return shared_detect_network_burst_windows(
        spike_matrix,
        recording_boundaries,
        dt_ms=dt_ms,
        activity_bin_ms=activity_bin_ms,
        smooth_bins=smooth_bins,
        threshold_std=threshold_std,
        min_active_fraction=min_active_fraction,
        min_burst_duration_ms=min_burst_duration_ms,
        merge_gap_ms=merge_gap_ms,
        pad_before_ms=pad_before_ms,
        pad_after_ms=pad_after_ms,
    )


def window_overlaps_excluded_bins(start, end, excluded_bins):
    """Check whether a candidate window overlaps any excluded bin.

    Args:
        start: Window start index.
        end: Window end index.
        excluded_bins: Sorted one-dimensional array of excluded bin indices.

    Returns:
        ``True`` when the half-open window ``[start, end)`` contains at least one
        excluded bin, otherwise ``False``.
    """
    return shared_window_overlaps_excluded_bins(start, end, excluded_bins)


def find_event_windows(spike_matrix, neuron_id, pre_context=50, post_context=10,
                       warmup=30, neg_ratio=1.0, neg_min_distance=100,
                       boundaries=None, excluded_bins=None, rng=None):
    return shared_find_event_windows(
        spike_matrix,
        neuron_id,
        pre_context=pre_context,
        post_context=post_context,
        warmup=warmup,
        neg_ratio=neg_ratio,
        neg_min_distance=neg_min_distance,
        boundaries=boundaries,
        excluded_bins=excluded_bins,
        rng=rng,
    )


def find_pre_centered_event_windows(spike_matrix, neighbor_indices, neuron_id,
                                    pre_context=50, post_context=10,
                                    warmup=30, neg_ratio=1.0,
                                    pre_min_lag=1, pre_max_lag=8,
                                    boundaries=None, excluded_bins=None,
                                    rng=None, max_anchors=None):
    return shared_find_pre_centered_event_windows(
        spike_matrix,
        neighbor_indices,
        neuron_id,
        pre_context=pre_context,
        post_context=post_context,
        warmup=warmup,
        neg_ratio=neg_ratio,
        pre_min_lag=pre_min_lag,
        pre_max_lag=pre_max_lag,
        boundaries=boundaries,
        excluded_bins=excluded_bins,
        rng=rng,
        max_anchors=max_anchors,
    )


class NeuronDataset(Dataset):
    """Dataset exposing one full recording trace per postsynaptic neuron.

    Args:
        spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
        neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
        true_binary: Dense binary ground-truth connectivity matrix.
        true_weights: Dense signed ground-truth weight matrix.
        neuron_positions: Spatial coordinates for each neuron.
        neuron_ids: Optional subset of postsynaptic neurons to expose.

    Returns:
        An initialized ``NeuronDataset`` instance for the older full-trace training
        path.
    """

    def __init__(self, spike_matrix, neighbor_indices, true_binary, true_weights,
                 neuron_positions, neuron_ids=None):
        """Initialize the full-trace dataset used by the older training path.

        Args:
            spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
            neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
            true_binary: Dense binary ground-truth connectivity matrix.
            true_weights: Dense signed ground-truth weight matrix.
            neuron_positions: Spatial coordinates for each neuron.
            neuron_ids: Optional subset of postsynaptic neurons to expose.

        Returns:
            None. The constructor stores the arrays needed for indexed dataset access.
        """
        self.spike_matrix = spike_matrix
        self.neighbor_indices = neighbor_indices
        self.true_binary = true_binary
        self.true_weights = true_weights
        self.neuron_positions = neuron_positions
        self.neuron_ids = neuron_ids if neuron_ids is not None else \
            np.arange(len(spike_matrix))

    def __len__(self):
        """Return the number of postsynaptic neurons exposed by the dataset.

        Args:
            None.

        Returns:
            The number of postsynaptic neurons represented by this dataset.
        """
        return len(self.neuron_ids)

    def __getitem__(self, idx):
        """Return one postsynaptic neuron's candidate inputs and supervision.

        Args:
            idx: Dataset index selecting one postsynaptic neuron.

        Returns:
            A tuple containing candidate presynaptic spikes, postsynaptic spikes,
            binary labels, true weights, and the postsynaptic neuron id.
        """
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


class EventWindowDataset(shared_EventWindowDataset):
    """Compatibility subclass for the extracted event-window dataset."""
    pass


def split_event_windows(windows, val_fraction=0.2, rng_seed=42):
    return shared_split_event_windows(windows, val_fraction=val_fraction, rng_seed=rng_seed)


def split_recording_boundaries(boundaries, val_fraction=0.2):
    return shared_split_recording_boundaries(boundaries, val_fraction=val_fraction)


def build_train_val_event_datasets(spike_matrix, neighbor_indices, neuron_ids,
                                   pre_context=50, post_context=10, warmup=30,
                                   neg_ratio=1.0, neg_min_distance=100,
                                   boundaries=None, excluded_bins=None,
                                   val_fraction=0.2,
                                   event_anchor_mode='post',
                                   pre_event_min_lag=1,
                                   pre_event_max_lag=8,
                                   pre_event_max_anchors=None,
                                   rng_seed=42):
    return shared_build_train_val_event_datasets(
        spike_matrix,
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
        event_anchor_mode=event_anchor_mode,
        pre_event_min_lag=pre_event_min_lag,
        pre_event_max_lag=pre_event_max_lag,
        pre_event_max_anchors=pre_event_max_anchors,
        rng_seed=rng_seed,
    )


# ============================================================================
# MODEL: Per-Neuron Differentiable LIF
# ============================================================================

class PerNeuronLIF(nn.Module):
    """
    Differentiable LIF with per-neuron learnable synaptic weights.

    Global (shared) parameters:
        - alpha: membrane leak factor
        - threshold_decay: spike-triggered threshold decay factor
        - beta: spike sigmoid sharpness
        - reset_strength: post-spike voltage reset

    Per-neuron parameters:
        - W: [n_neurons, K] synaptic weights — THE connectivity we infer
        - delay_logits: [n_neurons, K, n_delays] per-synapse delay distributions
        - threshold_base: baseline firing threshold in normalized voltage units
        - threshold_increment: spike-triggered threshold increment

    At each timestep t, for postsynaptic neuron j:
        I_syn(t) = sum_i( W[j,i] * pre_i(t - delay_ji) )
        V(t) = alpha * V(t-1) + I_syn(t)
        theta_j(t) = threshold_base_j + a_j(t)
        spike_prob(t) = sigmoid( beta * (V(t) - theta_j(t)) )
        a_j(t+1) = threshold_decay * a_j(t) + threshold_increment_j * spike_prob(t)
        V(t) -= reset_strength * spike_prob(t)

    After training, connectivity matrix = W.

    Args:
        n_neurons: Number of postsynaptic neurons modeled in parallel.
        K: Number of candidate presynaptic neurons per postsynaptic neuron.
        max_delay: Number of discrete delay bins modeled per candidate synapse.

    Returns:
        An initialized ``PerNeuronLIF`` module with learnable connectivity and
        delay parameters.
    """

    def __init__(self, n_neurons, K, max_delay=5, threshold_mode='adaptive'):
        """Initialize the differentiable per-neuron LIF model.

        Args:
            n_neurons: Number of postsynaptic neurons modeled in parallel.
            K: Number of candidate presynaptic neurons per postsynaptic neuron.
            max_delay: Number of discrete delay bins modeled per candidate synapse.

        Returns:
            None. The constructor allocates the learnable weights, delay logits,
            and shared membrane parameters.
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.K = K
        self.max_delay = max_delay
        self.threshold_mode = str(threshold_mode).strip().lower()
        if self.threshold_mode not in {'adaptive', 'shared'}:
            raise ValueError(
                f"Unsupported threshold_mode={threshold_mode!r}; use 'adaptive' or 'shared'"
            )

        # ── Per-neuron learnable weights [n_neurons, K] ──
        # Initialized at zero; positive → exc, negative → inh, ~zero → no connection
        self.W = nn.Parameter(torch.zeros(n_neurons, K))

        # ── Per-neuron delay distributions [n_neurons, K, max_delay] ──
        # Softmax over discrete delays [0, 1, ..., max_delay-1] ms
        self.delay_logits = nn.Parameter(torch.zeros(n_neurons, K, max_delay))

        # ── Global membrane parameters (shared across all neurons) ──
        self.alpha_logit = nn.Parameter(torch.tensor(3.0))   # sigmoid→ ~0.95 → tau_m ~20ms
        if self.threshold_mode == 'adaptive':
            self.threshold_base = nn.Parameter(torch.ones(n_neurons))
            self.threshold_increment_raw = nn.Parameter(torch.full((n_neurons,), -2.0))
            self.threshold_decay_logit = nn.Parameter(torch.tensor(3.0))
            self.shared_threshold = None
        else:
            self.shared_threshold = nn.Parameter(torch.tensor(1.0))
            self.threshold_base = None
            self.threshold_increment_raw = None
            self.threshold_decay_logit = None
        self.beta = nn.Parameter(torch.tensor(5.0))
        self.reset_strength = nn.Parameter(torch.tensor(2.0))

    @property
    def alpha(self):
        """Return the membrane leak factor constrained to the open interval ``(0, 1)``.

        Args:
            None.

        Returns:
            The shared membrane leak factor derived from ``alpha_logit``.
        """
        return torch.sigmoid(self.alpha_logit)

    @property
    def threshold(self):
        """Return the shared threshold or mean adaptive baseline threshold."""
        if self.threshold_mode == 'adaptive':
            return self.threshold_base.mean()
        return self.shared_threshold

    @property
    def threshold_base_values(self):
        """Return per-neuron threshold baselines for logging and saving."""
        if self.threshold_mode == 'adaptive':
            return self.threshold_base
        return self.shared_threshold.expand(self.n_neurons)

    @property
    def threshold_increment(self):
        """Return the positive spike-triggered threshold increment per neuron."""
        if self.threshold_mode == 'adaptive':
            return F.softplus(self.threshold_increment_raw)
        return torch.zeros(
            self.n_neurons,
            device=self.shared_threshold.device,
            dtype=self.shared_threshold.dtype,
        )

    @property
    def threshold_decay(self):
        """Return the shared adaptive-threshold decay constrained to ``(0, 1)``."""
        if self.threshold_mode == 'adaptive':
            return torch.sigmoid(self.threshold_decay_logit)
        return self.shared_threshold.new_tensor(0.0)

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
        threshold_base = self.threshold_base_values[neuron_ids]
        threshold_increment = self.threshold_increment[neuron_ids]
        threshold_decay = self.threshold_decay
        reset = F.softplus(self.reset_strength)

        spike_probs_list = []
        voltages_list = []
        v = torch.zeros(B, device=device)
        threshold_adapt = torch.zeros(B, device=device)

        for chunk_start in range(0, T, tbptt_len):
            chunk_end = min(chunk_start + tbptt_len, T)
            I_chunk = I_syn[:, chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start

            # Detach voltage at chunk boundary (truncated BPTT)
            v = v.detach()
            threshold_adapt = threshold_adapt.detach()

            sp_chunk = torch.zeros(B, chunk_len, device=device)
            v_chunk = torch.zeros(B, chunk_len, device=device)

            for t in range(chunk_len):
                v = alpha * v + I_chunk[:, t]
                dynamic_threshold = threshold_base + threshold_adapt
                s = torch.sigmoid(beta * (v - dynamic_threshold))
                sp_chunk[:, t] = s
                v_chunk[:, t] = v
                threshold_adapt = threshold_decay * threshold_adapt + threshold_increment * s
                v = v - reset * s

            spike_probs_list.append(sp_chunk)
            voltages_list.append(v_chunk)

        spike_probs = torch.cat(spike_probs_list, dim=1)  # [B, T]
        voltages = torch.cat(voltages_list, dim=1)

        return spike_probs, voltages, w

    def get_connectivity_matrix(self, neighbor_indices):
        """
        Assemble full [n_neurons, n_neurons] connectivity matrix from learned weights.

        Args:
            neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.

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
        """Assemble a dense delay matrix from the learned delay distributions.

        Args:
            neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.

        Returns:
            A dense ``[n_neurons, n_neurons]`` matrix of expected learned delays.
        """
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
    """Compute the full-trace spike prediction and sparsity loss.

    Args:
        spike_probs: Predicted postsynaptic spike probabilities.
        post_spikes: Ground-truth postsynaptic spike train.
        weights: Learned candidate weights for the current batch.
        pos_weight: Positive-class weighting used in the spike BCE term.
        l1_lambda: Weight on L1 sparsity regularization.

    Returns:
        A tuple containing the total loss tensor, scalar spike BCE loss, and
        scalar L1 penalty.
    """
    weight_mask = torch.where(post_spikes == 1, pos_weight, 1.0)
    spike_loss = F.binary_cross_entropy(
        spike_probs.clamp(1e-7, 1 - 1e-7), post_spikes, weight=weight_mask
    )
    l1_loss = l1_lambda * weights.abs().mean()
    return spike_loss + l1_loss, spike_loss.item(), l1_loss.item()


def train_epoch(model, dataloader, optimizer, device, pos_weight, l1_lambda):
    """Run one training epoch on the full-trace per-neuron dataset.

    Args:
        model: ``PerNeuronLIF`` model being optimized.
        dataloader: DataLoader yielding full-trace neuron samples.
        optimizer: Optimizer used for parameter updates.
        device: Torch device where training runs.
        pos_weight: Positive-class weighting used in the spike BCE term.
        l1_lambda: Weight on L1 sparsity regularization.

    Returns:
        A tuple of mean total loss, mean spike loss, and mean L1 loss across the epoch.
    """
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
    return shared_compute_event_loss(
        spike_probs,
        post_spikes,
        weights,
        warmup,
        pos_weight=pos_weight,
        l1_lambda=l1_lambda,
    )


def train_epoch_events(model, dataloader, optimizer, device, pos_weight,
                       l1_lambda, warmup):
    return shared_train_epoch_events(
        model,
        dataloader,
        optimizer,
        device,
        pos_weight,
        l1_lambda,
        warmup,
    )


@torch.no_grad()
def evaluate_event_windows(model, dataloader, device, pos_weight,
                           l1_lambda, warmup):
    return shared_evaluate_event_windows(
        model,
        dataloader,
        device,
        pos_weight,
        l1_lambda,
        warmup,
    )


def estimate_surrogate_connectivity_score_sets(
        spike_matrix, neighbor_indices, n_neurons, K_actual, max_delay,
        threshold_mode, lr, batch_size, pos_weight, l1_lambda,
        pre_context=50, post_context=10, warmup=30,
        neg_ratio=1.0, neg_min_distance=100,
        boundaries=None, excluded_bins=None, val_fraction=0.2,
        device='cpu', n_surrogates=4, surrogate_epochs=2,
        surrogate_patience=1, surrogate_min_shift_fraction=0.10,
        surrogate_seed=1234):
    return shared_estimate_surrogate_connectivity_score_sets(
        PerNeuronLIF,
        spike_matrix,
        neighbor_indices,
        n_neurons,
        K_actual,
        max_delay,
        threshold_mode,
        lr,
        batch_size,
        pos_weight,
        l1_lambda,
        pre_context=pre_context,
        post_context=post_context,
        warmup=warmup,
        neg_ratio=neg_ratio,
        neg_min_distance=neg_min_distance,
        boundaries=boundaries,
        excluded_bins=excluded_bins,
        val_fraction=val_fraction,
        device=device,
        n_surrogates=n_surrogates,
        surrogate_epochs=surrogate_epochs,
        surrogate_patience=surrogate_patience,
        surrogate_min_shift_fraction=surrogate_min_shift_fraction,
        surrogate_seed=surrogate_seed,
    )


@torch.no_grad()
def evaluate_connectivity(model, neighbor_indices, true_binary, neuron_ids=None,
                          connectivity_threshold_mode='oracle_f1',
                          surrogate_score_sets=None,
                          surrogate_fdr=0.005):
    """Compatibility wrapper around the shared connectivity evaluator."""
    return shared_evaluate_connectivity(
        model,
        neighbor_indices,
        true_binary,
        neuron_ids=neuron_ids,
        connectivity_threshold_mode=connectivity_threshold_mode,
        surrogate_score_sets=surrogate_score_sets,
        surrogate_fdr=surrogate_fdr,
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(connectivity_results, scores, labels, conn_matrix,
                 train_losses, val_losses, conn_aucs, val_window_results,
                 neuron_positions, connections, neighbor_indices,
                 model, session_name, output_name, output_dir,
                 output_suffix='', threshold_label=None):
    """Render and save the standard spike-only learned-LIF summary figure.

    Args:
        connectivity_results: Aggregate connectivity metrics computed from learned weights.
        scores: Flattened connectivity scores used for thresholding and PR analysis.
        labels: Flattened ground-truth connectivity labels aligned with `scores`.
        conn_matrix: Full learned connectivity matrix.
        train_losses: Per-epoch training loss history.
        val_losses: Per-epoch validation loss history.
        conn_aucs: Per-epoch connectivity AUC history.
        val_window_results: Held-out event-window validation summary.
        neuron_positions: 2-D neuron coordinates from the simulation session.
        connections: Ground-truth connection table from the simulation session.
        neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
        model: Trained learned-LIF model.
        session_name: Human-readable session name used in figure titles.
        output_name: Stem used for the saved figure filename.
        output_dir: Output directory where the figure is written.
        output_suffix: Optional suffix appended to the saved figure name.
        threshold_label: Optional human-readable label for the threshold rule.

    Returns:
        The path to the saved summary figure.
    """

    n_neurons = len(neuron_positions)
    threshold_label = threshold_label or connectivity_results.get(
        'connectivity_threshold_mode', 'thresholded'
    )
    thresh = float(connectivity_results.get('threshold', 0.5))
    thresh_text = f'{thresh:.4f}' if np.isfinite(thresh) else 'inf'
    estimated_fdr = connectivity_results.get('estimated_fdr')
    expected_null_selected = connectivity_results.get('expected_null_selected')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Learned LIF Connectivity — {session_name} ({threshold_label})',
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
    if np.isfinite(thresh):
        ax.axvline(thresh, color='blue', linewidth=2,
                   label=f'{threshold_label}={thresh_text}')
    else:
        ax.text(0.02, 0.95, f'{threshold_label}=inf', transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
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
    ax.set_title(f'Predicted {threshold_label} (TP={tp}, FP={fp}, FN={fn})')
    ax.set_aspect('equal')

    # ---- Summary ----
    ax = axes[1, 2]
    ax.axis('off')
    alpha_val = torch.sigmoid(model.alpha_logit).item()
    tau_eff = -1.0 / np.log(alpha_val + 1e-10)
    summary_lines = [
        'LEARNED LIF CONNECTIVITY',
        '=' * 40,
        '',
        f'Network: {session_name}',
        f'Neurons: {n_neurons}',
        '',
        'Learned Membrane Parameters:',
        f'  threshold mode: {model.threshold_mode}',
        f'  alpha:     {alpha_val:.4f} (tau_m ~ {tau_eff:.1f} ms)',
        f'  threshold base: {model.threshold.item():.4f}',
        f'  threshold inc:  {model.threshold_increment.mean().item():.4f}',
        f'  threshold decay:{model.threshold_decay.item():.4f}',
        f'  beta:      {model.beta.item():.4f}',
        f'  reset:     {F.softplus(model.reset_strength).item():.4f}',
        '',
        'Held-out window validation:',
        f"  Loss:      {val_window_results.get('loss', 0):.4f}",
        f"  Spike:     {val_window_results.get('spike_loss', 0):.4f}",
        f"  L1:        {val_window_results.get('l1_loss', 0):.4f}",
        f"  Windows:   {val_window_results.get('n_windows', 0)}",
        '',
        'Connectivity (all fitted neurons):',
        f"  Rule:      {connectivity_results.get('connectivity_threshold_mode', threshold_label)}",
        f'  Threshold: {thresh_text}',
        f"  AUC:       {connectivity_results['auc']:.4f}",
        f"  AP:        {connectivity_results['ap']:.4f}",
        f"  F1:        {connectivity_results['f1']:.4f}",
        f"  Precision: {connectivity_results.get('precision', 0):.4f}",
        f"  Recall:    {connectivity_results.get('recall', 0):.4f}",
        f'  TP: {tp}  FP: {fp}  FN: {fn}',
    ]
    if estimated_fdr is not None:
        summary_lines.extend([
            f'  Est FDR:   {float(estimated_fdr):.4f}',
            f'  Null sel:  {float(expected_null_selected or 0.0):.2f}',
        ])
    summary_lines.extend([
        '',
        'Weight stats:',
        f"  Mean |w| (connected): {connectivity_results.get('mean_connected_weight', 0):.4f}",
        f"  Positives: {connectivity_results['n_positive']}/{connectivity_results['n_total']}",
    ])
    summary = '\n'.join(summary_lines)
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'learned_lif_{output_name}{output_suffix}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {path}")
    return path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_session(session_dir, recording_idx=0, dt=1.0):
    """Load one saved recording and its network metadata from a simulation session.

    Args:
        session_dir: Session directory containing saved recordings and network structure.
        recording_idx: Zero-based recording index to load from the session.
        dt: Bin width in milliseconds used to convert saved burst onset times into bins.

    Returns:
        A dictionary containing spike times, duration, burst-onset bins, connectivity,
        neuron positions, cluster assignments, and neuron count for the recording.
    """
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
    """Plot a single-recording raster and shade excluded windows.

    Args:
        spike_times: Per-neuron spike-time sequences in milliseconds.
        cluster_assignments: Optional cluster index for each neuron.
        duration_ms: Recording duration in milliseconds.
        excluded_windows: Optional exclusion windows in milliseconds.
        title: Figure title.
        output_path: Optional file path where the raster should be saved.

    Returns:
        A ``(fig, ax)`` tuple for the created raster plot.
    """
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


def run_pipeline(session_dir, K=100, recording_idx=0, n_epochs=100, lr=1e-3,
                 batch_size=64, patience=20, val_fraction=0.2, dt=1.0,
                 max_delay=5, l1_lambda=0.01, pos_weight=5.0,
                 subsample_T=None, device=None, output_tag=None,
                 pre_context=50, post_context=10, warmup=30,
                 neg_ratio=1.0, neg_min_distance=100, use_all_recordings=True,
                 event_anchor_mode='post', pre_event_min_lag=1,
                 pre_event_max_lag=None, pre_event_max_anchors=None,
                 candidate_mode='hybrid', candidate_spatial_frac=0.8,
                 candidate_min_lag=1, candidate_max_lag=None,
                 threshold_mode='adaptive',
                 connectivity_threshold_mode='oracle_f1',
                 surrogate_fdr=0.005,
                 n_threshold_surrogates=4,
                 surrogate_epochs=2,
                 surrogate_patience=1,
                 surrogate_min_shift_fraction=0.10,
                 surrogate_seed=1234,
                 exclude_detected_bursts=True,
                 burst_activity_bin_ms=100.0,
                 burst_smooth_bins=3,
                 burst_threshold_std=3.0,
                 burst_min_active_fraction=0.10,
                 burst_min_duration_ms=100.0,
                 burst_merge_gap_ms=150.0,
                 burst_pad_before_ms=100.0,
                 burst_pad_after_ms=250.0):
    """Train the spike-only learned-LIF connectivity model and export artifacts.

    Args:
        session_dir: Session directory containing saved simulation recordings.
        K: Number of candidate presynaptic neurons retained per postsynaptic neuron.
        recording_idx: Recording index used when fitting only a single recording.
        n_epochs: Maximum number of training epochs.
        lr: Optimizer learning rate.
        batch_size: Event-window batch size.
        patience: Early-stopping patience measured in epochs.
        val_fraction: Fraction of data held out for validation.
        dt: Spike bin size in milliseconds.
        max_delay: Maximum discrete synaptic delay in bins.
        l1_lambda: Weight on L1 sparsity regularization for learned weights.
        pos_weight: Positive-class weighting used in the spike BCE loss.
        subsample_T: Optional limit on the number of fitted time bins.
        device: Torch device string; defaults to CUDA when available.
        output_tag: Optional suffix appended to saved artifact names.
        pre_context: Number of pre-spike bins included in each event window.
        post_context: Number of post-spike bins included in each event window.
        warmup: Number of warmup bins excluded from the event loss.
        neg_ratio: Number of negative windows sampled per positive window.
        neg_min_distance: Minimum distance in bins between negative windows and real spikes.
        use_all_recordings: Whether to concatenate all session recordings before fitting.
        candidate_mode: Candidate proposal mode, typically spatial or hybrid.
        candidate_spatial_frac: Spatial fraction reserved in hybrid candidate mode.
        candidate_min_lag: Minimum causal lag in bins for temporal candidates.
        candidate_max_lag: Maximum causal lag in bins for temporal candidates.
        threshold_mode: Threshold parameterization, either ``adaptive`` or ``shared``.
        connectivity_threshold_mode: Edge-call thresholding rule, either
            ``oracle_f1`` or ``surrogate_fdr``.
        surrogate_fdr: Target false discovery rate used in surrogate threshold mode.
        n_threshold_surrogates: Number of circular-shift surrogate models used for null calibration.
        surrogate_epochs: Maximum epochs used to fit each surrogate null model.
        surrogate_patience: Early-stopping patience for each surrogate null model.
        surrogate_min_shift_fraction: Minimum per-recording circular shift applied to each neuron.
        surrogate_seed: Base random seed used for surrogate generation.
        exclude_detected_bursts: Whether to detect and exclude burst windows from fitting.
        burst_activity_bin_ms: Bin width used for burst detection.
        burst_smooth_bins: Smoothing width applied to population activity for burst detection.
        burst_threshold_std: Threshold factor applied to burst detection.
        burst_min_active_fraction: Minimum active-neuron fraction for a detected burst.
        burst_min_duration_ms: Minimum duration of a detected burst window.
        burst_merge_gap_ms: Maximum gap between burst segments before merging.
        burst_pad_before_ms: Padding added before each excluded burst window.
        burst_pad_after_ms: Padding added after each excluded burst window.

    Returns:
        A tuple containing the aggregate connectivity metrics dictionary and the
        learned full connectivity matrix.
    """
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
    print(f"Event anchor mode: {event_anchor_mode}")
    print(f"Threshold mode: {threshold_mode}")
    print(f"Connectivity thresholding: {connectivity_threshold_mode}")
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
    # Optionally detect burst-dominated periods and remove them from both scoring and training windows.
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

    # Union saved stimulation onsets with detected burst bins into one exclusion mask.
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
    # Build temporal candidates on the training recordings only so the validation slice stays untouched.
    candidate_train_boundaries, _ = split_recording_boundaries(boundaries, val_fraction)
    candidate_max_lag = max_delay if candidate_max_lag is None else candidate_max_lag
    pre_event_max_lag = max_delay if pre_event_max_lag is None else pre_event_max_lag
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
        event_anchor_mode=event_anchor_mode,
        pre_event_min_lag=pre_event_min_lag,
        pre_event_max_lag=pre_event_max_lag,
        pre_event_max_anchors=pre_event_max_anchors,
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

    # Window batches can mix neurons, but each sample still points back to its own learned weight row.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # Model
    model = PerNeuronLIF(
        n_neurons=n_neurons,
        K=K_actual,
        max_delay=max_delay,
        threshold_mode=threshold_mode,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    membrane_threshold_params = 2 * n_neurons + 4 if model.threshold_mode == 'adaptive' else 4
    print(f"  Parameters: {n_params:,} "
          f"(W: {n_neurons*K_actual:,}, delays: {n_neurons*K_actual*max_delay:,}, "
            f"membrane+threshold: {membrane_threshold_params:,})")

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
            thresh_inc = model.threshold_increment.mean().item()
            thresh_decay = model.threshold_decay.item()
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1:3d}: loss={loss:.4f} (spike={sl:.4f} l1={l1l:.4f}) "
                  f"val_loss={val_loss:.4f} conn_AUC={conn_auc:.4f} "
                                f"alpha={alpha:.3f} theta_mode={model.threshold_mode} theta0={thresh:.3f} eta={thresh_inc:.3f} rho={thresh_decay:.3f} "
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

    oracle_results, all_scores, all_labels, conn_matrix = evaluate_connectivity(
        model,
        neighbor_indices,
        true_binary,
        connectivity_threshold_mode='oracle_f1',
    )

    print(
        f"\n  Calibrating non-leaky connectivity threshold with "
        f"{n_threshold_surrogates} circular-shift surrogates "
        f"(target FDR={surrogate_fdr:.3f}) for surrogate-threshold output..."
    )
    surrogate_score_sets = estimate_surrogate_connectivity_score_sets(
        spike_matrix,
        neighbor_indices,
        n_neurons=n_neurons,
        K_actual=K_actual,
        max_delay=max_delay,
        threshold_mode=threshold_mode,
        lr=lr,
        batch_size=batch_size,
        pos_weight=pos_weight,
        l1_lambda=l1_lambda,
        pre_context=pre_context,
        post_context=post_context,
        warmup=warmup,
        neg_ratio=neg_ratio,
        neg_min_distance=neg_min_distance,
        boundaries=boundaries,
        excluded_bins=excluded_bins,
        val_fraction=val_fraction,
        device=device,
        n_surrogates=n_threshold_surrogates,
        surrogate_epochs=surrogate_epochs,
        surrogate_patience=surrogate_patience,
        surrogate_min_shift_fraction=surrogate_min_shift_fraction,
        surrogate_seed=surrogate_seed,
    )
    print(
        f"  Surrogate score sets: {surrogate_score_sets.shape[0]} models x "
        f"{surrogate_score_sets.shape[1]} edges"
    )
    surrogate_results, _, _, _ = evaluate_connectivity(
        model,
        neighbor_indices,
        true_binary,
        connectivity_threshold_mode='surrogate_fdr',
        surrogate_score_sets=surrogate_score_sets,
        surrogate_fdr=surrogate_fdr,
    )

    threshold_results = {
        'oracle_f1': oracle_results,
        'surrogate_fdr': surrogate_results,
    }
    all_results = threshold_results[connectivity_threshold_mode]

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
    print(f"  Threshold rule: {all_results.get('connectivity_threshold_mode', connectivity_threshold_mode)}")
    if all_results.get('estimated_fdr') is not None:
        print(
            f"  Estimated FDR: {all_results['estimated_fdr']:.4f} "
            f"(target {all_results.get('surrogate_fdr_target', surrogate_fdr):.4f})"
        )
    print(
        f"  Oracle figure threshold: {oracle_results.get('threshold', 0.5):.4f}"
    )
    surrogate_threshold = float(surrogate_results.get('threshold', np.inf))
    surrogate_threshold_text = (
        f'{surrogate_threshold:.4f}' if np.isfinite(surrogate_threshold) else 'inf'
    )
    print(
        f"  Surrogate figure threshold: {surrogate_threshold_text}"
    )

    alpha = torch.sigmoid(model.alpha_logit).item()
    thresh_inc = model.threshold_increment.mean().item()
    thresh_decay = model.threshold_decay.item()
    print(f"\n  Learned: alpha={alpha:.4f} (tau_m~{-1/np.log(alpha+1e-10):.1f}ms), threshold_mode={model.threshold_mode}, "
            f"threshold_base={model.threshold.item():.4f}, "
            f"threshold_inc={thresh_inc:.4f}, threshold_decay={thresh_decay:.4f}, "
            f"beta={model.beta.item():.4f}")

    # Visualize
    output_dir = os.path.join(PROJECT_ROOT, 'learned_lif_outputs')
    plot_suffixes = {
        'oracle_f1': '_oracle_f1',
        'surrogate_fdr': '_surrogate_fdr',
    }
    threshold_labels = {
        'oracle_f1': 'Oracle F1',
        'surrogate_fdr': f'Surrogate FDR {surrogate_fdr:.2f}',
    }
    primary_plot_path = plot_results(
        all_results, all_scores, all_labels, conn_matrix,
        train_losses, val_losses, conn_aucs, val_window_results,
        positions, connections, neighbor_indices,
        model, session_name, output_name, output_dir,
        output_suffix=plot_suffixes[connectivity_threshold_mode],
        threshold_label=threshold_labels[connectivity_threshold_mode],
    )
    secondary_mode = 'surrogate_fdr' if connectivity_threshold_mode == 'oracle_f1' else 'oracle_f1'
    secondary_results = threshold_results[secondary_mode]
    secondary_plot_path = plot_results(
        secondary_results, all_scores, all_labels, conn_matrix,
        train_losses, val_losses, conn_aucs, val_window_results,
        positions, connections, neighbor_indices,
        model, session_name, output_name, output_dir,
        output_suffix=plot_suffixes[secondary_mode],
        threshold_label=threshold_labels[secondary_mode],
    )
    oracle_plot_path = (
        primary_plot_path if connectivity_threshold_mode == 'oracle_f1' else secondary_plot_path
    )
    surrogate_plot_path = (
        primary_plot_path if connectivity_threshold_mode == 'surrogate_fdr' else secondary_plot_path
    )

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
        'threshold_mode': model.threshold_mode,
        'connectivity_threshold_mode': connectivity_threshold_mode,
        'event_window_config': {
            'event_anchor_mode': event_anchor_mode,
            'pre_event_min_lag': int(pre_event_min_lag),
            'pre_event_max_lag': int(pre_event_max_lag),
            'pre_event_max_anchors': (
                None if pre_event_max_anchors is None else int(pre_event_max_anchors)
            ),
        },
        'saved_burst_onset_bins': saved_burst_onset_bins,
        'detected_burst_windows': detected_burst_info['windows'],
        'excluded_bins': excluded_bins,
        'neighbor_indices': neighbor_indices,
        'connectivity_matrix': conn_matrix,
        'results_window_val': val_window_results,
        'results_val': all_results,
        'results_all': all_results,
        'results_oracle_f1': oracle_results,
        'results_surrogate_fdr': surrogate_results,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': conn_aucs,
        'connectivity_aucs': conn_aucs,
        'connectivity_figure_paths': {
            'primary': primary_plot_path,
            'oracle_f1': oracle_plot_path,
            'surrogate_fdr': surrogate_plot_path,
        },
        'connectivity_thresholding': {
            'mode': connectivity_threshold_mode,
            'surrogate_fdr': float(surrogate_fdr),
            'n_threshold_surrogates': int(n_threshold_surrogates),
            'surrogate_epochs': int(surrogate_epochs),
            'surrogate_patience': int(surrogate_patience),
            'surrogate_min_shift_fraction': float(surrogate_min_shift_fraction),
            'surrogate_seed': int(surrogate_seed),
            'estimated_fdr': all_results.get('estimated_fdr'),
            'expected_null_selected': all_results.get('expected_null_selected'),
            'selected_edges': all_results.get('selected_edges'),
            'oracle_threshold': oracle_results.get('threshold'),
            'surrogate_threshold': surrogate_results.get('threshold'),
        },
        'adaptive_threshold': {
            'threshold_mode': model.threshold_mode,
            'threshold_base_mean': float(model.threshold.item()),
            'threshold_base_std': float(model.threshold_base_values.std(unbiased=False).item()),
            'threshold_increment_mean': float(model.threshold_increment.mean().item()),
            'threshold_decay': float(model.threshold_decay.item()),
        },
    }, model_path)
    print(f"  Model + connectivity saved: {model_path}")

    # Save connectivity matrix as separate .npz for easy access
    conn_path = os.path.join(output_dir, f'connectivity_{output_name}.npz')
    np.savez_compressed(
        conn_path,
        connectivity_matrix=conn_matrix,
        threshold=all_results.get('threshold', 0.5),
        connectivity_threshold_mode=connectivity_threshold_mode,
        estimated_fdr=all_results.get('estimated_fdr'),
        neighbor_indices=neighbor_indices,
        neuron_positions=positions,
        cluster_assignments=cluster_assignments,
        saved_burst_onset_bins=saved_burst_onset_bins,
        detected_burst_windows=detected_burst_info['windows'],
        excluded_bins=excluded_bins,
    )
    print(f"  Connectivity matrix saved: {conn_path}")

    # Surface the prepared inputs so training-free baselines (e.g. the CCG
    # baseline) can be benchmarked on the exact same data the learned model used.
    # Additive only: existing keys are left untouched.
    all_results.setdefault('spike_matrix', spike_matrix)
    all_results.setdefault('neighbor_indices', neighbor_indices)
    all_results.setdefault('boundaries', boundaries)
    all_results.setdefault('excluded_bins', excluded_bins)
    all_results.setdefault('true_binary', true_binary)
    all_results.setdefault('neuron_ids', all_neuron_ids)

    return all_results, conn_matrix


# ============================================================================
# CLI
# ============================================================================

def select_session():
    """Interactively choose a saved session from the default output folder.

    Args:
        None.

    Returns:
        The filesystem path of the selected session directory.
    """
    data_dir = os.path.join(PROJECT_ROOT, "LIF data")
    sessions = sorted([s for s in glob.glob(os.path.join(data_dir, "*"))
                       if os.path.isdir(s)])
    if not sessions:
        print("No sessions in LIF data/"); sys.exit(1)
    print("\nSessions:")
    for i, s in enumerate(sessions):
        print(f"  [{i}] {os.path.basename(s)}")
    choice = input("Select (Enter=first): ").strip()
    return sessions[0] if choice == '' else sessions[int(choice)]


def build_parser():
    """Build the CLI parser for the spike-only learned-LIF pipeline.

    Args:
        None.

    Returns:
        An ``argparse.ArgumentParser`` configured for the spike-only inference CLI.
    """
    parser = argparse.ArgumentParser(description='Learned LIF Connectivity')
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional suffix for saved artifact names')
    parser.add_argument('--k', type=int, default=100)
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
    parser.add_argument('--threshold-mode', type=str, default='adaptive',
                        choices=['adaptive', 'shared'],
                        help='Use per-neuron adaptive thresholds or one shared threshold for all neurons')
    parser.add_argument('--connectivity-threshold-mode', type=str, default='oracle_f1',
                        choices=['oracle_f1', 'surrogate_fdr'],
                        help='Choose the binary edge cutoff from ground-truth F1 or surrogate null calibration')
    parser.add_argument('--surrogate-fdr', type=float, default=0.005,
                        help='Target false discovery rate for surrogate thresholding')
    parser.add_argument('--n-threshold-surrogates', type=int, default=4,
                        help='Number of surrogate null models fitted for surrogate thresholding')
    parser.add_argument('--surrogate-epochs', type=int, default=2,
                        help='Maximum epochs per surrogate null model')
    parser.add_argument('--surrogate-patience', type=int, default=1,
                        help='Early-stopping patience for surrogate null models')
    parser.add_argument('--surrogate-min-shift-frac', type=float, default=0.10,
                        help='Minimum circular shift size as a fraction of each recording segment')
    parser.add_argument('--surrogate-seed', type=int, default=1234,
                        help='Base random seed used for surrogate threshold calibration')
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
    parser.add_argument('--event-anchor-mode', type=str, default='post',
                        choices=['post', 'pre', 'both'],
                        help='Anchor event windows on post spikes, candidate pre spikes, or both')
    parser.add_argument('--pre-event-min-lag', type=int, default=1,
                        help='Minimum causal lag in bins for pre-centered positive windows')
    parser.add_argument('--pre-event-max-lag', type=int, default=None,
                        help='Maximum causal lag in bins for pre-centered positive windows (default: use --max-delay)')
    parser.add_argument('--pre-event-max-anchors', type=int, default=None,
                        help='Optional cap on unique pre-centered trigger bins sampled per postsynaptic neuron')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                        help='Validation fraction: held-out recordings when possible, otherwise held-out windows')
    burst_group = parser.add_mutually_exclusive_group()
    burst_group.add_argument('--exclude-detected-bursts', dest='exclude_detected_bursts', action='store_true',
                             help='Detect network burst windows and exclude them from candidate scoring and event windows (default)')
    burst_group.add_argument('--include-detected-bursts', dest='exclude_detected_bursts', action='store_false',
                             help='Keep detected network burst windows in candidate scoring and event windows')
    parser.set_defaults(exclude_detected_bursts=True)
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
    return parser


def main(argv=None):
    """Parse CLI arguments and launch the spike-only learned-LIF pipeline.

    Args:
        argv: Optional CLI argument list. When omitted, arguments are read from
            ``sys.argv``.

    Returns:
        None. The function resolves the session and runs the spike-only inference
        pipeline.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

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
        event_anchor_mode=args.event_anchor_mode,
        pre_event_min_lag=args.pre_event_min_lag,
        pre_event_max_lag=args.pre_event_max_lag,
        pre_event_max_anchors=args.pre_event_max_anchors,
        use_all_recordings=not args.single_recording,
        candidate_mode=args.candidate_mode,
        candidate_spatial_frac=args.candidate_spatial_frac,
        candidate_min_lag=args.candidate_min_lag,
        candidate_max_lag=args.candidate_max_lag,
        threshold_mode=args.threshold_mode,
        connectivity_threshold_mode=args.connectivity_threshold_mode,
        surrogate_fdr=args.surrogate_fdr,
        n_threshold_surrogates=args.n_threshold_surrogates,
        surrogate_epochs=args.surrogate_epochs,
        surrogate_patience=args.surrogate_patience,
        surrogate_min_shift_fraction=args.surrogate_min_shift_frac,
        surrogate_seed=args.surrogate_seed,
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


if __name__ == "__main__":
    main()
