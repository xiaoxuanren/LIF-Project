"""
Voltage-augmented learned LIF connectivity inference.

This pipeline extends the spike-only learned-LIF approach by supervising the
model with cleaned subthreshold voltage traces in addition to postsynaptic
spikes. Presynaptic inputs remain spike trains; the voltage target is the
postsynaptic membrane trace after masking spike neighborhoods and excluding the
synthetic +20 mV visualization peaks saved in the simulation output.

Usage:
    python voltage_augmented_learned_lif_connectivity.py --session "LIF data/20260425_110211"
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lif_simulation.voltage_storage import resolve_recording_voltage

from .shared_data import (
    build_segmentwise_circular_shift_surrogates,
    build_ground_truth,
    spike_times_to_binary,
)
from .learned_lif_connectivity import (
    combine_excluded_bins,
    compute_binary_classification_metrics,
    compute_neighbor_indices,
    detect_network_burst_windows,
    flatten_candidate_scores,
    find_event_windows,
    select_connectivity_threshold,
    split_event_windows,
    split_recording_boundaries,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spike_times_to_sample_bins(spike_times, sample_rate_ms, n_samples):
    """Map spike times into unique sample indices at the stored voltage resolution.

    Args:
        spike_times: Per-neuron spike times in milliseconds.
        sample_rate_ms: Voltage sampling interval in milliseconds.
        n_samples: Number of available voltage samples.

    Returns:
        A sorted array of unique sample indices aligned to the stored voltage trace.
    """
    spikes = np.asarray(spike_times, dtype=np.float64)
    if spikes.size == 0:
        return np.empty(0, dtype=np.int32)

    bins = np.floor(spikes / float(sample_rate_ms) + 1e-9).astype(np.int32)
    bins = bins[(bins >= 0) & (bins < n_samples)]
    if bins.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(bins)


def ms_to_bins(time_ms, sample_rate_ms):
    """Convert a duration in milliseconds into at least one sample bin.

    Args:
        time_ms: Duration in milliseconds.
        sample_rate_ms: Sample spacing in milliseconds.

    Returns:
        The nearest positive integer number of sample bins.
    """
    return max(1, int(round(float(time_ms) / float(sample_rate_ms))))


def preprocess_voltage_recording(voltage_traces, spike_times, sample_rate_ms,
                                 mask_pre_ms=1.0, mask_post_ms=5.0,
                                 peak_threshold_mv=15.0):
    """Mask spike neighborhoods and normalize voltage traces for subthreshold supervision.

    Args:
        voltage_traces: Raw saved voltage array with shape [n_neurons, n_samples].
        spike_times: Per-neuron spike-time arrays used to mask spike neighborhoods.
        sample_rate_ms: Voltage sampling interval in milliseconds.
        mask_pre_ms: Time masked before each spike.
        mask_post_ms: Time masked after each spike.
        peak_threshold_mv: Optional upper voltage threshold used to drop synthetic spike peaks.

    Returns:
        A dictionary containing normalized voltages, a valid-sample mask, per-neuron
        valid fractions, and the normalization baseline and scale used per neuron.
    """
    n_neurons, n_samples = voltage_traces.shape
    mask_pre_bins = int(round(float(mask_pre_ms) / float(sample_rate_ms)))
    mask_post_bins = int(round(float(mask_post_ms) / float(sample_rate_ms)))

    # Start from every sample valid, then carve out spike neighborhoods and visualization peaks.
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

    normalized = np.zeros_like(voltage_traces, dtype=np.float32)
    valid_fraction = np.zeros(n_neurons, dtype=np.float32)
    baseline_medians = np.zeros(n_neurons, dtype=np.float32)
    baseline_scales = np.ones(n_neurons, dtype=np.float32)

    # Normalize each neuron from its surviving subthreshold samples so voltage loss compares shapes, not offsets.
    for neuron_id in range(n_neurons):
        row = voltage_traces[neuron_id].astype(np.float32, copy=False)
        valid_values = row[valid_mask[neuron_id]]

        if valid_values.size == 0:
            baseline = float(np.median(row))
            scale = float(np.std(row))
        else:
            baseline = float(np.median(valid_values))
            scale = float(np.std(valid_values))

        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        normalized_row = (row - baseline) / scale
        normalized_row[~valid_mask[neuron_id]] = 0.0
        normalized[neuron_id] = normalized_row

        valid_fraction[neuron_id] = float(valid_mask[neuron_id].mean())
        baseline_medians[neuron_id] = baseline
        baseline_scales[neuron_id] = scale

    return {
        'normalized_voltage': normalized,
        'valid_mask': valid_mask.astype(np.float32),
        'valid_fraction': valid_fraction,
        'baseline_medians': baseline_medians,
        'baseline_scales': baseline_scales,
    }


def resolve_voltage_trace_array(data, recording_path=None):
    """Return the preferred saved voltage array and the key that supplied it.

    Args:
        data: Loaded recording bundle returned by ``np.load``.
        recording_path: Optional filesystem path to the recording file.

    Returns:
        A tuple ``(voltage_array, source_key)`` selecting the preferred saved
        voltage trace field.
    """
    voltage_bundle = resolve_recording_voltage(data, recording_path=recording_path, load_into_memory=True)
    if voltage_bundle is None:
        raise KeyError('No saved voltage traces found in recording file')
    return np.asarray(voltage_bundle['traces'], dtype=np.float32), voltage_bundle['source_key']


def load_all_recordings_with_voltage(session_dir, dt=1.0,
                                    mask_pre_ms=1.0, mask_post_ms=5.0,
                                    peak_threshold_mv=15.0):
    """Load, clean, and concatenate all voltage-enabled recordings from one session.

    Args:
        session_dir: Session directory containing saved recordings and a network file.
        dt: Spike bin width in milliseconds.
        mask_pre_ms: Time masked before each spike when cleaning voltage traces.
        mask_post_ms: Time masked after each spike when cleaning voltage traces.
        peak_threshold_mv: Optional voltage threshold used to drop visualization peaks.

    Returns:
        A dictionary containing concatenated spike and voltage matrices, validity masks,
        recording boundaries, network metadata, total duration, and per-recording summaries.
    """
    rec_files = sorted(glob.glob(os.path.join(session_dir, 'recording[0-9][0-9][0-9].npz')))
    if not rec_files:
        raise FileNotFoundError(f'No recordings in {session_dir}')

    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f'No network file in {session_dir}')

    net_data = np.load(net_files[0], allow_pickle=True)

    spike_matrices = []
    voltage_matrices = []
    voltage_masks = []
    boundaries = [0]
    total_duration = 0.0
    recording_summaries = []
    burst_onset_bins = []

    # Clean each recording independently before concatenation so masks and normalization stay recording-local.
    for rec_file in rec_files:
        data = np.load(rec_file, allow_pickle=True)
        duration = float(data['duration'])
        rec_start_bin = boundaries[-1]
        spike_matrix = spike_times_to_binary(data['spike_times'], duration, dt)

        sample_rate_ms = float(data['voltage_sample_rate'])
        if not np.isclose(sample_rate_ms, dt):
            raise ValueError(
                f'Voltage sample rate {sample_rate_ms} ms does not match dt={dt} ms in {rec_file}'
            )

        voltage_traces, voltage_source_key = resolve_voltage_trace_array(data, recording_path=rec_file)
        n_common = min(spike_matrix.shape[1], voltage_traces.shape[1])
        if n_common <= 0:
            raise ValueError(f'No overlapping samples in {rec_file}')

        spike_matrix = spike_matrix[:, :n_common]
        voltage_traces = voltage_traces[:, :n_common]

        if 'burst_onset_times' in data.files:
            rec_burst_onsets = np.asarray(data['burst_onset_times'], dtype=float)
            rec_burst_onsets = rec_burst_onsets[
                (rec_burst_onsets >= 0.0) & (rec_burst_onsets < duration)
            ]
            if rec_burst_onsets.size > 0:
                rec_burst_bins = (rec_burst_onsets / dt).astype(np.int32)
                rec_burst_bins = rec_burst_bins[rec_burst_bins < n_common]
                burst_onset_bins.extend((rec_burst_bins + rec_start_bin).tolist())

        processed = preprocess_voltage_recording(
            voltage_traces,
            data['spike_times'],
            sample_rate_ms,
            mask_pre_ms=mask_pre_ms,
            mask_post_ms=mask_post_ms,
            peak_threshold_mv=peak_threshold_mv,
        )

        spike_matrices.append(spike_matrix)
        voltage_matrices.append(processed['normalized_voltage'])
        voltage_masks.append(processed['valid_mask'])

        total_duration += n_common * dt
        boundaries.append(boundaries[-1] + n_common)
        recording_summaries.append({
            'path': rec_file,
            'voltage_source_key': voltage_source_key,
            'sample_rate_ms': sample_rate_ms,
            'duration_ms': float(n_common * dt),
            'mean_valid_fraction': float(processed['valid_fraction'].mean()),
            'min_valid_fraction': float(processed['valid_fraction'].min()),
            'max_valid_fraction': float(processed['valid_fraction'].max()),
        })

    return {
        'spike_matrix': np.concatenate(spike_matrices, axis=1),
        'voltage_matrix': np.concatenate(voltage_matrices, axis=1),
        'voltage_mask': np.concatenate(voltage_masks, axis=1),
        'total_duration': total_duration,
        'boundaries': boundaries,
        'burst_onset_bins': np.unique(np.asarray(burst_onset_bins, dtype=np.int32)),
        'n_recordings': len(rec_files),
        'connections': net_data['connections'],
        'neuron_positions': net_data['neuron_positions'],
        'n_neurons': len(net_data['neuron_positions']),
        'recording_summaries': recording_summaries,
    }


class VoltageEventWindowDataset(Dataset):
    """Event-window dataset that pairs spike inputs with masked voltage targets.

    Args:
        spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
        voltage_matrix: Cleaned and normalized voltage matrix.
        voltage_mask: Binary mask marking valid supervised voltage samples.
        neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
        neuron_ids: Optional subset of postsynaptic neurons to expose.
        pre_context: Number of causal bins kept before each event.
        post_context: Number of bins kept after each event.
        warmup: Number of leading bins excluded from the loss.
        neg_ratio: Target number of negative windows per positive window.
        neg_min_distance: Minimum distance from any postsynaptic spike for negatives.
        boundaries: Optional concatenated recording boundaries.
        excluded_bins: Optional sorted bins excluded from all windows.
        rng_seed: Base random seed used when sampling windows.
        windows: Optional precomputed windows to load instead of regenerating.

    Returns:
        An initialized ``VoltageEventWindowDataset`` instance.
    """

    def __init__(self, spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
                 neuron_ids=None, pre_context=50, post_context=10, warmup=30,
                 neg_ratio=1.0, neg_min_distance=100, boundaries=None,
                 excluded_bins=None,
                 rng_seed=42, windows=None):
        """Build or load event windows for voltage-augmented training and validation.

        Args:
            spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
            voltage_matrix: Cleaned and normalized voltage matrix.
            voltage_mask: Binary mask marking valid supervised voltage samples.
            neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
            neuron_ids: Optional subset of postsynaptic neurons to expose.
            pre_context: Number of causal bins kept before each event.
            post_context: Number of bins kept after each event.
            warmup: Number of leading bins excluded from the loss.
            neg_ratio: Target number of negative windows per positive window.
            neg_min_distance: Minimum distance from any postsynaptic spike for negatives.
            boundaries: Optional concatenated recording boundaries.
            excluded_bins: Optional sorted bins excluded from all windows.
            rng_seed: Base random seed used when sampling windows.
            windows: Optional precomputed windows to load instead of regenerating.

        Returns:
            None. The constructor populates the window list and summary counts.
        """
        self.spike_matrix = spike_matrix
        self.voltage_matrix = voltage_matrix
        self.voltage_mask = voltage_mask
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

            self.windows = []
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
        """Return the number of stored event windows.

        Args:
            None.

        Returns:
            The number of stored event windows.
        """
        return len(self.windows)

    def __getitem__(self, idx):
        """Return one window of spike inputs, voltage targets, masks, and metadata.

        Args:
            idx: Dataset index selecting one event window.

        Returns:
            A tuple containing candidate presynaptic spikes, postsynaptic spikes,
            target voltage, target-voltage mask, postsynaptic neuron id, and a
            positive/negative label.
        """
        post_id, start, end, is_pos = self.windows[idx]
        pre_ids = self.neighbor_indices[post_id]

        pre_spikes = self.spike_matrix[pre_ids, start:end].astype(np.float32)
        post_spikes = self.spike_matrix[post_id, start:end].astype(np.float32)
        post_voltage = self.voltage_matrix[post_id, start:end].astype(np.float32)
        post_voltage_mask = self.voltage_mask[post_id, start:end].astype(np.float32)

        return (
            torch.from_numpy(pre_spikes),
            torch.from_numpy(post_spikes),
            torch.from_numpy(post_voltage),
            torch.from_numpy(post_voltage_mask),
            post_id,
            is_pos,
        )


def build_train_val_voltage_datasets(spike_matrix, voltage_matrix, voltage_mask,
                                     neighbor_indices, neuron_ids,
                                     pre_context=50, post_context=10, warmup=30,
                                     neg_ratio=1.0, neg_min_distance=100,
                                     boundaries=None, excluded_bins=None,
                                     val_fraction=0.2,
                                     rng_seed=42):
    """Construct train and validation event datasets for voltage-augmented training.

    Args:
        spike_matrix: Binary spike matrix with shape ``[n_neurons, T]``.
        voltage_matrix: Cleaned and normalized voltage matrix.
        voltage_mask: Binary mask marking valid supervised voltage samples.
        neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
        neuron_ids: Postsynaptic neurons to include in the datasets.
        pre_context: Number of causal bins kept before each event.
        post_context: Number of bins kept after each event.
        warmup: Number of leading bins excluded from the loss.
        neg_ratio: Target number of negative windows per positive window.
        neg_min_distance: Minimum distance from any postsynaptic spike for negatives.
        boundaries: Optional concatenated recording boundaries.
        excluded_bins: Optional sorted bins excluded from all windows.
        val_fraction: Fraction reserved for validation.
        rng_seed: Base random seed used for window sampling and splitting.

    Returns:
        A tuple ``(train_ds, val_ds, strategy)`` containing the two datasets and a
        human-readable description of the split strategy that was used.
    """
    train_boundaries, val_boundaries = split_recording_boundaries(boundaries, val_fraction)

    if val_boundaries is not None:
        train_ds = VoltageEventWindowDataset(
            spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
            neuron_ids=neuron_ids,
            pre_context=pre_context, post_context=post_context, warmup=warmup,
            neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
            boundaries=train_boundaries, excluded_bins=excluded_bins, rng_seed=rng_seed,
        )
        val_ds = VoltageEventWindowDataset(
            spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
            neuron_ids=neuron_ids,
            pre_context=pre_context, post_context=post_context, warmup=warmup,
            neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
            boundaries=val_boundaries, excluded_bins=excluded_bins, rng_seed=rng_seed + 1000,
        )
        if len(train_ds) > 0 and len(val_ds) > 0:
            strategy = (
                f'held-out recordings ({len(train_boundaries) - 1} train, '
                f'{len(val_boundaries) - 1} val)'
            )
            if excluded_bins is not None and len(excluded_bins) > 0:
                strategy += f', excluding {len(excluded_bins)} bins'
            return train_ds, val_ds, strategy

    full_ds = VoltageEventWindowDataset(
        spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
        neuron_ids=neuron_ids,
        pre_context=pre_context, post_context=post_context, warmup=warmup,
        neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
        boundaries=boundaries, excluded_bins=excluded_bins, rng_seed=rng_seed,
    )
    train_windows, val_windows = split_event_windows(
        full_ds.windows, val_fraction=val_fraction, rng_seed=rng_seed + 2000,
    )
    train_ds = VoltageEventWindowDataset(
        spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
        warmup=warmup, windows=train_windows,
    )
    val_ds = VoltageEventWindowDataset(
        spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
        warmup=warmup, windows=val_windows,
    )
    return train_ds, val_ds, 'held-out event windows'


class VoltageAugmentedPerNeuronLIF(nn.Module):
    """Differentiable per-neuron LIF model supervised by spikes and voltage traces.

    Args:
        n_neurons: Number of postsynaptic neurons modeled in parallel.
        K: Number of candidate presynaptic neurons per postsynaptic neuron.
        max_delay: Number of discrete delay bins modeled per candidate synapse.

    Returns:
        An initialized ``VoltageAugmentedPerNeuronLIF`` module with learnable
        connectivity, delays, bias, and adaptive threshold dynamics.
    """

    def __init__(self, n_neurons, K, max_delay=5, threshold_mode='adaptive'):
        """Initialize the voltage-augmented learned-LIF model parameters.

        Args:
            n_neurons: Number of postsynaptic neurons modeled in parallel.
            K: Number of candidate presynaptic neurons per postsynaptic neuron.
            max_delay: Number of discrete delay bins modeled per candidate synapse.

        Returns:
            None. The constructor allocates the learnable weights, delays, bias,
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

        self.W = nn.Parameter(torch.zeros(n_neurons, K))
        self.delay_logits = nn.Parameter(torch.zeros(n_neurons, K, max_delay))
        self.bias = nn.Parameter(torch.zeros(n_neurons))

        self.alpha_logit = nn.Parameter(torch.tensor(3.0))
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
        """Simulate spike probabilities and membrane voltages for a batch of event windows.

        Args:
            pre_spikes: Candidate presynaptic spike windows with shape [B, K, T].
            post_spikes: Postsynaptic spike windows with shape [B, T]; unused but kept
                for API compatibility with the spike-only model.
            neuron_ids: Batch of postsynaptic neuron indices used to gather parameters.
            tbptt_len: Chunk length used for truncated backpropagation through time.

        Returns:
            A tuple of predicted spike probabilities, predicted voltages, and the
            learned candidate weights for the batch's postsynaptic neurons.
        """
        del post_spikes
        B, K, T = pre_spikes.shape
        device = pre_spikes.device

        w = self.W[neuron_ids]
        delay_logits = self.delay_logits[neuron_ids]
        delay_weights = F.softmax(delay_logits, dim=-1)
        bias = self.bias[neuron_ids]

        delayed_inputs = torch.zeros(B, K, T, device=device)
        for d in range(self.max_delay):
            if d == 0:
                shifted = pre_spikes
            else:
                shifted = F.pad(pre_spikes[:, :, :-d], (d, 0))
            delayed_inputs += shifted * delay_weights[:, :, d].unsqueeze(-1)

        I_syn = (w.unsqueeze(-1) * delayed_inputs).sum(dim=1) + bias.unsqueeze(-1)

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

        spike_probs = torch.cat(spike_probs_list, dim=1)
        voltages = torch.cat(voltages_list, dim=1)
        return spike_probs, voltages, w

    def get_connectivity_matrix(self, neighbor_indices):
        """Expand candidate weights into a full postsynaptic-by-presynaptic matrix.

        Args:
            neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.

        Returns:
            A dense ``[n_neurons, n_neurons]`` connectivity matrix assembled from
            the learned candidate weights.
        """
        W_np = self.W.detach().cpu().numpy()
        n = self.n_neurons
        conn_matrix = np.zeros((n, n), dtype=np.float32)

        for j in range(n):
            pre_ids = neighbor_indices[j]
            conn_matrix[j, pre_ids] = W_np[j, :len(pre_ids)]

        return conn_matrix

    def get_learned_delays(self, neighbor_indices):
        """Expand expected learned delays into a full neuron-by-neuron matrix.

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


def compute_voltage_augmented_event_loss(spike_probs, predicted_voltage,
                                         post_spikes, target_voltage,
                                         target_voltage_mask, weights, warmup,
                                         pos_weight=5.0, l1_lambda=0.01,
                                         voltage_lambda=1.0):
    """Combine spike BCE, masked voltage loss, and L1 sparsity into one objective.

    Args:
        spike_probs: Predicted postsynaptic spike probabilities for each event window.
        predicted_voltage: Predicted membrane voltages for each event window.
        post_spikes: Ground-truth postsynaptic spike windows.
        target_voltage: Masked and normalized target voltage windows.
        target_voltage_mask: Binary mask indicating which target-voltage samples are valid.
        weights: Learned candidate weights for the batch.
        warmup: Number of leading bins excluded from the loss.
        pos_weight: Positive-class weighting used in the spike BCE term.
        l1_lambda: Weight on L1 sparsity regularization.
        voltage_lambda: Weight on the masked voltage supervision term.

    Returns:
        The total loss tensor, scalar spike loss, scalar voltage loss, scalar L1 loss,
        and the number of valid voltage points contributing to the batch loss.
    """
    sp = spike_probs[:, warmup:]
    ps = post_spikes[:, warmup:]
    vp = predicted_voltage[:, warmup:]
    vt = target_voltage[:, warmup:]
    vm = target_voltage_mask[:, warmup:] > 0.5

    weight_mask = torch.where(ps == 1, pos_weight, 1.0)
    spike_loss = F.binary_cross_entropy(
        sp.clamp(1e-7, 1 - 1e-7), ps, weight=weight_mask,
    )

    # Some windows lose every voltage target after masking; keep the loss well-defined in that case.
    if torch.any(vm):
        voltage_loss = F.smooth_l1_loss(vp[vm], vt[vm])
        n_voltage_points = int(vm.sum().item())
    else:
        voltage_loss = vp.sum() * 0.0
        n_voltage_points = 0

    l1_loss = l1_lambda * weights.abs().mean()
    total = spike_loss + voltage_lambda * voltage_loss + l1_loss
    return total, spike_loss.item(), voltage_loss.item(), l1_loss.item(), n_voltage_points


def train_epoch_events(model, dataloader, optimizer, device, warmup,
                       pos_weight, l1_lambda, voltage_lambda):
    """Run one event-window training epoch for the voltage-augmented model.

    Args:
        model: Voltage-augmented learned-LIF model being optimized.
        dataloader: Event-window dataloader yielding training batches.
        optimizer: Torch optimizer used to update model parameters.
        device: Torch device where computation runs.
        warmup: Number of leading bins excluded from the event loss.
        pos_weight: Positive-class weighting used in the spike BCE term.
        l1_lambda: Weight on L1 sparsity regularization.
        voltage_lambda: Weight on the masked voltage supervision term.

    Returns:
        A dictionary containing averaged total, spike, voltage, and L1 losses plus
        batch and valid-voltage-point counts.
    """
    model.train()
    total_loss = 0.0
    total_spike = 0.0
    total_voltage = 0.0
    total_l1 = 0.0
    total_voltage_points = 0
    n = 0

    for pre_sp, post_sp, post_v, post_vm, neuron_ids, is_pos in dataloader:
        del is_pos
        pre_sp = pre_sp.to(device)
        post_sp = post_sp.to(device)
        post_v = post_v.to(device)
        post_vm = post_vm.to(device)
        neuron_ids = neuron_ids.to(device)

        optimizer.zero_grad()
        window_len = pre_sp.shape[2]
        spike_probs, voltages, weights = model(
            pre_sp, post_sp, neuron_ids, tbptt_len=window_len,
        )
        loss, sl, vl, l1l, n_voltage_points = compute_voltage_augmented_event_loss(
            spike_probs, voltages, post_sp, post_v, post_vm, weights,
            warmup, pos_weight=pos_weight, l1_lambda=l1_lambda,
            voltage_lambda=voltage_lambda,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_spike += sl
        total_voltage += vl
        total_l1 += l1l
        total_voltage_points += n_voltage_points
        n += 1

    return {
        'loss': total_loss / max(n, 1),
        'spike_loss': total_spike / max(n, 1),
        'voltage_loss': total_voltage / max(n, 1),
        'l1_loss': total_l1 / max(n, 1),
        'n_batches': n,
        'n_voltage_points': total_voltage_points,
    }


@torch.no_grad()
def evaluate_event_windows(model, dataloader, device, warmup,
                           pos_weight, l1_lambda, voltage_lambda):
    """Evaluate held-out event windows under the combined spike and voltage objective.

    Args:
        model: Voltage-augmented learned-LIF model being evaluated.
        dataloader: Event-window dataloader yielding validation batches.
        device: Torch device where computation runs.
        warmup: Number of leading bins excluded from the event loss.
        pos_weight: Positive-class weighting used in the spike BCE term.
        l1_lambda: Weight on L1 sparsity regularization.
        voltage_lambda: Weight on the masked voltage supervision term.

    Returns:
        A dictionary containing averaged held-out loss terms plus batch, window,
        and valid-voltage-point counts.
    """
    model.eval()
    total_loss = 0.0
    total_spike = 0.0
    total_voltage = 0.0
    total_l1 = 0.0
    total_voltage_points = 0
    n = 0

    for pre_sp, post_sp, post_v, post_vm, neuron_ids, is_pos in dataloader:
        del is_pos
        pre_sp = pre_sp.to(device)
        post_sp = post_sp.to(device)
        post_v = post_v.to(device)
        post_vm = post_vm.to(device)
        neuron_ids = neuron_ids.to(device)

        window_len = pre_sp.shape[2]
        spike_probs, voltages, weights = model(
            pre_sp, post_sp, neuron_ids, tbptt_len=window_len,
        )
        loss, sl, vl, l1l, n_voltage_points = compute_voltage_augmented_event_loss(
            spike_probs, voltages, post_sp, post_v, post_vm, weights,
            warmup, pos_weight=pos_weight, l1_lambda=l1_lambda,
            voltage_lambda=voltage_lambda,
        )

        total_loss += loss.item()
        total_spike += sl
        total_voltage += vl
        total_l1 += l1l
        total_voltage_points += n_voltage_points
        n += 1

    return {
        'loss': total_loss / max(n, 1),
        'spike_loss': total_spike / max(n, 1),
        'voltage_loss': total_voltage / max(n, 1),
        'l1_loss': total_l1 / max(n, 1),
        'n_batches': n,
        'n_windows': len(dataloader.dataset),
        'n_voltage_points': total_voltage_points,
    }


def estimate_surrogate_connectivity_score_sets(
        spike_matrix, voltage_matrix, voltage_mask,
        neighbor_indices, n_neurons, K_actual, max_delay,
        threshold_mode, lr, batch_size, warmup,
        pos_weight, l1_lambda, voltage_lambda,
        pre_context=50, post_context=10,
        neg_ratio=1.0, neg_min_distance=100,
        boundaries=None, excluded_bins=None, val_fraction=0.2,
        device='cpu', n_surrogates=4, surrogate_epochs=2,
        surrogate_patience=1, surrogate_min_shift_fraction=0.10,
        surrogate_seed=1234):
    """Fit lightweight null models on circular-shift spike+voltage surrogates."""
    rng = np.random.default_rng(surrogate_seed)
    all_neuron_ids = np.arange(n_neurons)
    score_sets = []

    for surrogate_idx in range(int(n_surrogates)):
        surrogate_spike_matrix, surrogate_voltage_matrix, surrogate_voltage_mask = (
            build_segmentwise_circular_shift_surrogates(
                [spike_matrix, voltage_matrix, voltage_mask],
                boundaries=boundaries,
                rng=rng,
                min_shift_fraction=surrogate_min_shift_fraction,
            )
        )
        dataset_seed = surrogate_seed + 1000 * (surrogate_idx + 1)
        train_ds, val_ds, _ = build_train_val_voltage_datasets(
            surrogate_spike_matrix,
            surrogate_voltage_matrix,
            surrogate_voltage_mask,
            neighbor_indices,
            all_neuron_ids,
            pre_context=pre_context,
            post_context=post_context,
            warmup=warmup,
            neg_ratio=neg_ratio,
            neg_min_distance=neg_min_distance,
            boundaries=boundaries,
            excluded_bins=excluded_bins,
            val_fraction=val_fraction,
            rng_seed=dataset_seed,
        )
        if len(train_ds) == 0:
            raise RuntimeError('Surrogate threshold calibration produced zero training windows')

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        torch.manual_seed(surrogate_seed + surrogate_idx)
        surrogate_model = VoltageAugmentedPerNeuronLIF(
            n_neurons=n_neurons,
            K=K_actual,
            max_delay=max_delay,
            threshold_mode=threshold_mode,
        ).to(device)
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=lr, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0
        max_epochs = max(int(surrogate_epochs), 1)
        max_patience = max(int(surrogate_patience), 1)

        for _ in range(max_epochs):
            train_epoch_events(
                surrogate_model, train_loader, optimizer, device, warmup,
                pos_weight, l1_lambda, voltage_lambda,
            )
            val_stats = evaluate_event_windows(
                surrogate_model, val_loader, device, warmup,
                pos_weight, l1_lambda, voltage_lambda,
            )

            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                best_state = {k: v.detach().clone() for k, v in surrogate_model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= max_patience:
                break

        if best_state is not None:
            surrogate_model.load_state_dict(best_state)

        surrogate_conn_matrix = surrogate_model.get_connectivity_matrix(neighbor_indices)
        score_sets.append(
            flatten_candidate_scores(
                surrogate_conn_matrix,
                neighbor_indices,
                neuron_ids=all_neuron_ids,
                absolute=True,
            )
        )

    return np.stack(score_sets, axis=0)


@torch.no_grad()
def evaluate_connectivity(model, neighbor_indices, true_binary, true_weights,
                          neuron_ids=None,
                          connectivity_threshold_mode='oracle_f1',
                          surrogate_score_sets=None,
                          surrogate_fdr=0.005):
    """Score learned weights against ground-truth connectivity and sign information.

    Args:
        model: Trained voltage-augmented learned-LIF model.
        neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
        true_binary: Ground-truth binary connectivity matrix.
        true_weights: Ground-truth signed connection-weight matrix.
        neuron_ids: Optional subset of postsynaptic neurons to evaluate.
        connectivity_threshold_mode: Binary thresholding rule for edge calls.
        surrogate_score_sets: Optional surrogate null-score matrix used when
            ``connectivity_threshold_mode='surrogate_fdr'``.
        surrogate_fdr: Target false discovery rate used in surrogate mode.

    Returns:
        A tuple containing the aggregate connectivity metrics dictionary, flattened
        absolute scores, flattened labels, flattened signed scores, flattened true
        weights, and the full learned connectivity matrix.
    """
    model.eval()
    conn_matrix = model.get_connectivity_matrix(neighbor_indices)
    n_neurons = model.n_neurons

    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)

    all_signed_scores = []
    all_abs_scores = []
    all_labels = []
    all_true_weights = []

    for j in neuron_ids:
        pre_ids = neighbor_indices[j]
        signed_scores = conn_matrix[j, pre_ids]
        all_signed_scores.append(signed_scores)
        all_abs_scores.append(np.abs(signed_scores))
        all_labels.append(true_binary[j, pre_ids].astype(np.float32))
        all_true_weights.append(true_weights[j, pre_ids].astype(np.float32))

    signed_scores = np.concatenate(all_signed_scores)
    abs_scores = np.concatenate(all_abs_scores)
    labels = np.concatenate(all_labels)
    flat_true_weights = np.concatenate(all_true_weights)

    results = {}
    if len(np.unique(labels)) > 1:
        results['auc'] = float(roc_auc_score(labels, abs_scores))
        results['ap'] = float(average_precision_score(labels, abs_scores))

        threshold_info = select_connectivity_threshold(
            labels,
            abs_scores,
            mode=connectivity_threshold_mode,
            surrogate_score_sets=surrogate_score_sets,
            surrogate_fdr=surrogate_fdr,
            default_threshold=0.5,
        )
        predicted = np.zeros_like(labels, dtype=np.int32)
        if np.isfinite(threshold_info['threshold']):
            predicted = (abs_scores >= threshold_info['threshold']).astype(np.int32)

        results.update(threshold_info)
        results.update(compute_binary_classification_metrics(labels, predicted))
    else:
        predicted = np.zeros_like(labels, dtype=int)
        results.update({
            'auc': 0.0,
            'ap': 0.0,
            'f1': 0.0,
            'threshold': 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': int(np.sum(labels == 0)),
            'connectivity_threshold_mode': str(connectivity_threshold_mode).strip().lower(),
            'estimated_fdr': None,
            'expected_null_selected': None,
            'selected_edges': 0,
        })

    connected_mask = labels == 1
    if np.sum(connected_mask) > 0:
        results['sign_accuracy'] = float(np.mean(
            np.sign(signed_scores[connected_mask]) == np.sign(flat_true_weights[connected_mask])
        ))
        results['mean_connected_weight'] = float(np.mean(np.abs(signed_scores[connected_mask])))
    else:
        results['sign_accuracy'] = 0.0
        results['mean_connected_weight'] = 0.0

    if np.sum(connected_mask) > 1:
        results['weight_corr'] = float(np.corrcoef(
            signed_scores[connected_mask], flat_true_weights[connected_mask]
        )[0, 1])
    else:
        results['weight_corr'] = 0.0

    predicted_tp = (predicted == 1) & connected_mask
    if np.sum(predicted_tp) > 0:
        results['predicted_tp_sign_accuracy'] = float(np.mean(
            np.sign(signed_scores[predicted_tp]) == np.sign(flat_true_weights[predicted_tp])
        ))
    else:
        results['predicted_tp_sign_accuracy'] = 0.0

    results['n_positive'] = int(labels.sum())
    results['n_total'] = int(len(labels))
    return results, abs_scores, labels, signed_scores, flat_true_weights, conn_matrix


def plot_score_separation_histogram(abs_scores, labels, output_path,
                                    threshold=None, threshold_label=None,
                                    title='Voltage-Augmented Score Separation',
                                    bins=80, log_floor=1e-10,
                                    surrogate_scores=None,
                                    surrogate_label='Surrogate null',
                                    surrogate_threshold=None,
                                    surrogate_threshold_label=None):
    """Render a log10 score histogram split by ground-truth connectivity labels.

    Args:
        abs_scores: Flattened nonnegative learned connectivity magnitudes.
        labels: Flattened binary ground-truth labels aligned with ``abs_scores``.
        output_path: Path where the histogram PNG should be written.
        threshold: Optional absolute-score cutoff to annotate on the histogram.
        threshold_label: Optional label describing the annotated cutoff source.
        title: Figure title.
        bins: Number of histogram bins.
        log_floor: Minimum score used before taking ``log10``.
        surrogate_scores: Optional null-score samples to overlay as an outline histogram.
        surrogate_label: Legend label used for the null overlay.
        surrogate_threshold: Optional surrogate-derived absolute-score cutoff to annotate.
        surrogate_threshold_label: Optional label describing the surrogate cutoff source.

    Returns:
        The saved PNG path.
    """
    abs_scores = np.asarray(abs_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if abs_scores.shape != labels.shape:
        raise ValueError('abs_scores and labels must have identical flattened shapes')

    neg_scores = np.log10(np.maximum(abs_scores[labels == 0], float(log_floor)))
    pos_scores = np.log10(np.maximum(abs_scores[labels == 1], float(log_floor)))
    surrogate_scores = None if surrogate_scores is None else np.asarray(
        surrogate_scores, dtype=np.float64
    ).reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    if neg_scores.size > 0:
        ax.hist(
            neg_scores,
            bins=bins,
            alpha=0.5,
            density=True,
            color='darkorange',
            label=f'Non-edges (n={neg_scores.size})',
        )
    if pos_scores.size > 0:
        ax.hist(
            pos_scores,
            bins=bins,
            alpha=0.5,
            density=True,
            color='seagreen',
            label=f'Edges (n={pos_scores.size})',
        )
    if surrogate_scores is not None and surrogate_scores.size > 0:
        ax.hist(
            np.log10(np.maximum(surrogate_scores, float(log_floor))),
            bins=bins,
            density=True,
            histtype='step',
            linewidth=1.8,
            color='slategray',
            label=f'{surrogate_label} (n={surrogate_scores.size})',
        )
    if threshold is not None and np.isfinite(float(threshold)):
        cutoff = float(threshold)
        cutoff_label = str(threshold_label).strip() if threshold_label is not None else ''
        prefix = f'{cutoff_label} cutoff' if cutoff_label else 'Cutoff'
        ax.axvline(
            np.log10(max(cutoff, float(log_floor))),
            color='navy',
            linewidth=2,
            linestyle='--',
            label=f'{prefix}={cutoff:.4g}',
        )
    if surrogate_threshold is not None and np.isfinite(float(surrogate_threshold)):
        cutoff = float(surrogate_threshold)
        cutoff_label = (
            str(surrogate_threshold_label).strip()
            if surrogate_threshold_label is not None else 'Surrogate'
        )
        prefix = f'{cutoff_label} cutoff' if cutoff_label else 'Surrogate cutoff'
        ax.axvline(
            np.log10(max(cutoff, float(log_floor))),
            color='firebrick',
            linewidth=2,
            linestyle=':',
            label=f'{prefix}={cutoff:.4g}',
        )
    ax.set_xlabel('log10 |W|')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_results(connectivity_results, abs_scores, labels, signed_scores,
                 conn_matrix, train_history, val_history, conn_aucs,
                 val_window_results, neuron_positions, connections,
                 neighbor_indices, model, session_name, output_name,
                 output_dir):
    """Render and save the standard voltage-augmented learned-LIF summary figure.

    Args:
        connectivity_results: Aggregate connectivity metrics computed from learned weights.
        abs_scores: Flattened absolute connectivity scores used for thresholding.
        labels: Flattened ground-truth connectivity labels aligned with `abs_scores`.
        signed_scores: Flattened signed learned weights aligned with `labels`.
        conn_matrix: Full learned connectivity matrix.
        train_history: Per-epoch training-stat dictionaries.
        val_history: Per-epoch validation-stat dictionaries.
        conn_aucs: Per-epoch connectivity AUC history.
        val_window_results: Held-out event-window validation summary.
        neuron_positions: 2-D neuron coordinates from the simulation session.
        connections: Ground-truth connection table from the simulation session.
        neighbor_indices: Candidate presynaptic indices for each postsynaptic neuron.
        model: Trained voltage-augmented learned-LIF model.
        session_name: Human-readable session name used in figure titles.
        output_name: Stem used for the saved figure filename.
        output_dir: Output directory where the figure is written.

    Returns:
        The path to the saved summary figure.
    """
    n_neurons = len(neuron_positions)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Voltage-Augmented Learned LIF - {session_name}',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot([item['loss'] for item in train_history], 'b-', alpha=0.75, label='Train total')
    ax.plot([item['loss'] for item in val_history], 'g-', alpha=0.75, label='Val total')
    ax.plot([item['voltage_loss'] for item in val_history], 'm--', alpha=0.75, label='Val voltage')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training / Validation Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(conn_aucs, 'r-', alpha=0.7, label='Conn AUC')
    ax2.set_ylabel('Connectivity AUC', color='red')
    ax2.legend(loc='center right', fontsize=8)

    ax = axes[0, 1]
    pos_scores = abs_scores[labels == 1]
    neg_scores = abs_scores[labels == 0]
    ax.hist(neg_scores, bins=50, alpha=0.6, color='darkorange',
            label=f'No conn (n={len(neg_scores)})', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, color='seagreen',
            label=f'Connected (n={len(pos_scores)})', density=True)
    thresh = connectivity_results.get('threshold', 0.5)
    ax.axvline(thresh, color='blue', linewidth=2, label=f'Thresh={thresh:.4f}')
    ax.set_xlabel('|Learned Weight|')
    ax.set_ylabel('Density')
    ax.set_title('Weight Score Distribution')
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    if connectivity_results['auc'] > 0:
        prec, rec, _ = precision_recall_curve(labels, abs_scores)
        ax.plot(rec, prec, 'b-', linewidth=2)
        ax.fill_between(rec, prec, alpha=0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(
        f'PR Curve (AUC={connectivity_results["auc"]:.3f}, '
        f'AP={connectivity_results["ap"]:.3f})'
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=16, edgecolors='navy', linewidths=0.45, zorder=3)
    true_abs_weights = np.abs(connections[:, 2].astype(float)) if len(connections) > 0 else np.array([1.0])
    max_true_abs = float(np.max(true_abs_weights)) if len(true_abs_weights) > 0 else 1.0
    max_true_abs = max(max_true_abs, 1e-6)
    draw_order = np.argsort(true_abs_weights) if len(connections) > 0 else []
    for conn_idx in draw_order:
        c = connections[conn_idx]
        i, j = int(c[0]), int(c[1])
        strength = abs(float(c[2])) / max_true_abs
        color = 'crimson' if float(c[2]) > 0 else 'royalblue'
        ax.plot([neuron_positions[i, 0], neuron_positions[j, 0]],
                [neuron_positions[i, 1], neuron_positions[j, 1]],
                color=color,
                alpha=0.22 + 0.45 * strength,
                linewidth=0.45 + 1.15 * strength,
                solid_capstyle='round',
                zorder=2)
    ax.set_title(f'True Connections (n={len(connections)})')
    ax.set_aspect('equal')

    ax = axes[1, 1]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=16, edgecolors='navy', linewidths=0.45, zorder=3)
    predicted_edges = []
    for j in range(n_neurons):
        pre_ids = neighbor_indices[j]
        for pre in pre_ids:
            weight = float(conn_matrix[j, pre])
            if abs(weight) >= thresh:
                predicted_edges.append((pre, j, weight))
    if predicted_edges:
        pred_abs_weights = np.array([abs(weight) for _, _, weight in predicted_edges], dtype=np.float32)
        max_pred_abs = max(float(np.max(pred_abs_weights)), 1e-6)
        for edge_idx in np.argsort(pred_abs_weights):
            pre, post, weight = predicted_edges[edge_idx]
            strength = abs(weight) / max_pred_abs
            color = 'crimson' if weight >= 0 else 'royalblue'
            ax.plot([neuron_positions[pre, 0], neuron_positions[post, 0]],
                    [neuron_positions[pre, 1], neuron_positions[post, 1]],
                    color=color,
                    alpha=0.32 + 0.55 * strength,
                    linewidth=0.7 + 1.5 * strength,
                    solid_capstyle='round',
                    zorder=2)
    tp = connectivity_results.get('tp', 0)
    fp = connectivity_results.get('fp', 0)
    fn = connectivity_results.get('fn', 0)
    ax.set_title(f'Predicted (TP={tp}, FP={fp}, FN={fn})')
    ax.set_aspect('equal')

    ax = axes[1, 2]
    ax.axis('off')
    alpha_val = torch.sigmoid(model.alpha_logit).item()
    tau_eff = -1.0 / np.log(alpha_val + 1e-10)
    bias_abs = float(model.bias.abs().mean().item())
    summary = f"""
VOLTAGE-AUGMENTED LEARNED LIF
========================================

Network: {session_name}
Neurons: {n_neurons}

Learned Membrane Parameters:
    threshold mode: {model.threshold_mode}
  alpha:     {alpha_val:.4f} (tau_m ~ {tau_eff:.1f} ms)
    threshold base: {model.threshold.item():.4f}
    threshold inc:  {model.threshold_increment.mean().item():.4f}
    threshold decay:{model.threshold_decay.item():.4f}
  beta:      {model.beta.item():.4f}
  reset:     {F.softplus(model.reset_strength).item():.4f}
  mean |bias|: {bias_abs:.4f}

Held-out window validation:
  Loss:        {val_window_results.get('loss', 0):.4f}
  Spike loss:  {val_window_results.get('spike_loss', 0):.4f}
  Voltage loss:{val_window_results.get('voltage_loss', 0):.4f}
  L1 loss:     {val_window_results.get('l1_loss', 0):.4f}
  Windows:     {val_window_results.get('n_windows', 0)}

Connectivity:
  AUC:         {connectivity_results['auc']:.4f}
  AP:          {connectivity_results['ap']:.4f}
  F1:          {connectivity_results['f1']:.4f}
  Precision:   {connectivity_results.get('precision', 0):.4f}
  Recall:      {connectivity_results.get('recall', 0):.4f}
  Sign acc:    {connectivity_results.get('sign_accuracy', 0):.4f}
  Weight corr: {connectivity_results.get('weight_corr', 0):.4f}
  TP/FP/FN:    {tp}/{fp}/{fn}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'voltage_augmented_learned_lif_{output_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Visualization saved: {path}')
    return path


def load_single_recording_with_voltage(session_dir, recording_idx=0, dt=1.0,
                                       mask_pre_ms=1.0, mask_post_ms=5.0,
                                       peak_threshold_mv=15.0):
    """Load and clean one voltage-enabled recording for single-recording training.

    Args:
        session_dir: Session directory containing saved recordings and a network file.
        recording_idx: Zero-based recording index to load.
        dt: Spike bin width in milliseconds.
        mask_pre_ms: Time masked before each spike when cleaning voltage traces.
        mask_post_ms: Time masked after each spike when cleaning voltage traces.
        peak_threshold_mv: Optional voltage threshold used to drop visualization peaks.

    Returns:
        A dictionary containing one recording's spike matrix, cleaned voltage matrix,
        validity mask, boundaries, network metadata, duration, and recording summary.
    """
    rec_path = os.path.join(session_dir, f'recording{recording_idx:03d}.npz')
    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f'No network file in {session_dir}')

    rec_data = np.load(rec_path, allow_pickle=True)
    net_data = np.load(net_files[0], allow_pickle=True)

    duration = float(rec_data['duration'])
    spike_matrix = spike_times_to_binary(rec_data['spike_times'], duration, dt)
    sample_rate_ms = float(rec_data['voltage_sample_rate'])
    if not np.isclose(sample_rate_ms, dt):
        raise ValueError(
            f'Voltage sample rate {sample_rate_ms} ms does not match dt={dt} ms in {rec_path}'
        )

    voltage_traces, voltage_source_key = resolve_voltage_trace_array(rec_data, recording_path=rec_path)
    n_common = min(spike_matrix.shape[1], voltage_traces.shape[1])
    spike_matrix = spike_matrix[:, :n_common]
    voltage_traces = voltage_traces[:, :n_common]

    burst_onset_bins = np.array([], dtype=np.int32)
    if 'burst_onset_times' in rec_data.files:
        rec_burst_onsets = np.asarray(rec_data['burst_onset_times'], dtype=float)
        rec_burst_onsets = rec_burst_onsets[
            (rec_burst_onsets >= 0.0) & (rec_burst_onsets < duration)
        ]
        if rec_burst_onsets.size > 0:
            burst_onset_bins = np.unique((rec_burst_onsets / dt).astype(np.int32))
            burst_onset_bins = burst_onset_bins[burst_onset_bins < n_common]

    processed = preprocess_voltage_recording(
        voltage_traces,
        rec_data['spike_times'],
        sample_rate_ms,
        mask_pre_ms=mask_pre_ms,
        mask_post_ms=mask_post_ms,
        peak_threshold_mv=peak_threshold_mv,
    )

    return {
        'spike_matrix': spike_matrix,
        'voltage_matrix': processed['normalized_voltage'],
        'voltage_mask': processed['valid_mask'],
        'boundaries': [0, n_common],
        'burst_onset_bins': burst_onset_bins,
        'connections': net_data['connections'],
        'neuron_positions': net_data['neuron_positions'],
        'n_neurons': len(net_data['neuron_positions']),
        'n_recordings': 1,
        'total_duration': float(n_common * dt),
        'recording_summaries': [{
            'path': rec_path,
            'voltage_source_key': voltage_source_key,
            'sample_rate_ms': sample_rate_ms,
            'duration_ms': float(n_common * dt),
            'mean_valid_fraction': float(processed['valid_fraction'].mean()),
            'min_valid_fraction': float(processed['valid_fraction'].min()),
            'max_valid_fraction': float(processed['valid_fraction'].max()),
        }],
    }


def run_pipeline(session_dir, K=50, recording_idx=0, n_epochs=40, lr=1e-3,
                 batch_size=128, patience=20, val_fraction=0.2, dt=1.0,
                 max_delay=8, l1_lambda=0.01, pos_weight=5.0,
                 voltage_lambda=1.0, subsample_T=None, device=None,
                 output_tag=None, pre_context=50, post_context=10,
                 warmup=30, neg_ratio=1.0, neg_min_distance=100,
                 use_all_recordings=True, candidate_mode='hybrid',
                 candidate_spatial_frac=0.8, candidate_min_lag=1,
                 candidate_max_lag=None, mask_pre_ms=1.0,
                 mask_post_ms=5.0, peak_threshold_mv=15.0,
                 threshold_mode='adaptive',
                 connectivity_threshold_mode='oracle_f1',
                 surrogate_fdr=0.005,
                 n_threshold_surrogates=4,
                 surrogate_epochs=2,
                 surrogate_patience=1,
                 surrogate_min_shift_fraction=0.10,
                 surrogate_seed=1234,
                 exclude_detected_bursts=False,
                 burst_activity_bin_ms=100.0,
                 burst_smooth_bins=3,
                 burst_threshold_std=3.0,
                 burst_min_active_fraction=0.10,
                 burst_min_duration_ms=100.0,
                 burst_merge_gap_ms=150.0,
                 burst_pad_before_ms=100.0,
                 burst_pad_after_ms=250.0):
    """Train the voltage-augmented learned-LIF model and export saved artifacts.

    Args:
        session_dir: Session directory containing saved simulation recordings.
        K: Number of candidate presynaptic neurons retained per postsynaptic neuron.
        recording_idx: Recording index used when fitting only a single recording.
        n_epochs: Maximum number of training epochs.
        lr: Optimizer learning rate.
        batch_size: Event-window batch size.
        patience: Early-stopping patience measured in epochs.
        val_fraction: Fraction of data held out for validation.
        dt: Spike and voltage bin size in milliseconds.
        max_delay: Maximum discrete synaptic delay in bins.
        l1_lambda: Weight on L1 sparsity regularization for learned weights.
        pos_weight: Positive-class weighting used in the spike BCE loss.
        voltage_lambda: Weight on the masked subthreshold voltage loss.
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
        mask_pre_ms: Time masked before each spike in the voltage target.
        mask_post_ms: Time masked after each spike in the voltage target.
        peak_threshold_mv: Optional voltage threshold used to drop visualization peaks.
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
    print('VOLTAGE-AUGMENTED LEARNED LIF CONNECTIVITY')
    print(f'Session: {session_name}')
    if output_tag:
        print(f'Output tag: {output_tag}')
    print(f'K={K}, epochs={n_epochs}, lr={lr}, max_delay={max_delay}, l1={l1_lambda}, voltage_lambda={voltage_lambda}')
    print(f'Window: warmup={warmup}, pre={pre_context}, post={post_context} ({warmup + pre_context + post_context} bins)')
    print(f'Voltage cleaning: mask_pre={mask_pre_ms}ms, mask_post={mask_post_ms}ms, peak<{peak_threshold_mv}mV')
    print(f'Threshold mode: {threshold_mode}')
    print(f'Connectivity thresholding: {connectivity_threshold_mode}')
    print(f'Device: {device}')
    print(f"{'='*70}")

    print('\n  Loading data...')
    if use_all_recordings:
        data = load_all_recordings_with_voltage(
            session_dir, dt=dt,
            mask_pre_ms=mask_pre_ms,
            mask_post_ms=mask_post_ms,
            peak_threshold_mv=peak_threshold_mv,
        )
        print(f'  Loaded {data["n_recordings"]} recordings, total duration: {data["total_duration"] / 1000:.0f}s')
    else:
        data = load_single_recording_with_voltage(
            session_dir, recording_idx=recording_idx, dt=dt,
            mask_pre_ms=mask_pre_ms,
            mask_post_ms=mask_post_ms,
            peak_threshold_mv=peak_threshold_mv,
        )
        print(f'  Loaded recording {recording_idx}, duration: {data["total_duration"] / 1000:.1f}s')

    spike_matrix = data['spike_matrix']
    voltage_matrix = data['voltage_matrix']
    voltage_mask = data['voltage_mask']
    boundaries = data['boundaries']
    n_neurons = data['n_neurons']
    connections = data['connections']
    positions = data['neuron_positions']
    saved_burst_onset_bins = np.asarray(
        data.get('burst_onset_bins', np.array([], dtype=np.int32)),
        dtype=np.int32,
    )

    if subsample_T is not None and subsample_T < spike_matrix.shape[1]:
        print(f'  Using first {subsample_T} ms')
        spike_matrix = spike_matrix[:, :subsample_T]
        voltage_matrix = voltage_matrix[:, :subsample_T]
        voltage_mask = voltage_mask[:, :subsample_T]
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
    valid_voltage_fraction = float(voltage_mask.mean())
    print(f'  Neurons: {n_neurons}, Connections: {len(connections)}')
    print(f'  Spike matrix: [{n_neurons}, {T}] ({total_spikes} total spikes)')
    print(f'  Voltage matrix: [{n_neurons}, {T}] (valid fraction={valid_voltage_fraction:.3f})')
    print(f'  Recording boundaries: {boundaries}')
    if len(saved_burst_onset_bins) > 0:
        print(f'  Saved stimulation onsets: {len(saved_burst_onset_bins)}')
    if exclude_detected_bursts:
        print(f'  Detected burst windows: {len(detected_burst_info["windows"])}')
        print(f'  Detected excluded bins: {len(detected_burst_info["excluded_bins"])}')
        if detected_burst_info['thresholds'].size > 0:
            print(
                f'  Mean burst threshold: {np.mean(detected_burst_info["thresholds"]):.3f} '
                f'active fraction per {burst_activity_bin_ms:.0f} ms bin'
            )
    if len(excluded_bins) > 0:
        print(f'  Total excluded bins: {len(excluded_bins)}')

    # Match the spike-only path by building temporal candidates from the training recordings only.
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
    print(f'  Candidate mode: {candidate_info["mode"]}')
    if candidate_info['mode'] == 'hybrid':
        print(f'  Candidate mix: {candidate_info["n_spatial"]} spatial + {candidate_info["n_temporal"]} temporal (lag {candidate_info["temporal_min_lag"]}-{candidate_info["temporal_max_lag"]} bins)')
        if len(excluded_bins) > 0:
            print('  Temporal candidate scoring excludes configured excluded bins')
        print(f'  Mean temporal-only candidates per neuron: {candidate_info["mean_temporal_only"]:.1f}')
    print(f'  K={K_actual}, coverage: {total_in_K}/{total_true} ({total_in_K / max(total_true, 1):.1%})')

    print('\n  Extracting event windows...')
    all_neuron_ids = np.arange(n_neurons)
    train_ds, val_ds, validation_strategy = build_train_val_voltage_datasets(
        spike_matrix, voltage_matrix, voltage_mask, neighbor_indices,
        all_neuron_ids,
        pre_context=pre_context,
        post_context=post_context,
        warmup=warmup,
        neg_ratio=neg_ratio,
        neg_min_distance=neg_min_distance,
        boundaries=boundaries,
        excluded_bins=excluded_bins,
        val_fraction=val_fraction,
        rng_seed=42,
    )
    print(f'  Validation strategy: {validation_strategy}')
    print(f'  Train windows: {len(train_ds)} ({train_ds.n_pos} pos, {train_ds.n_neg} neg)')
    print(f'  Val windows:   {len(val_ds)} ({val_ds.n_pos} pos, {val_ds.n_neg} neg)')

    if len(train_ds) == 0:
        raise RuntimeError('No training windows extracted. Check spike activity and window settings.')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VoltageAugmentedPerNeuronLIF(
        n_neurons=n_neurons,
        K=K_actual,
        max_delay=max_delay,
        threshold_mode=threshold_mode,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    membrane_threshold_params = 2 * n_neurons + 4 if model.threshold_mode == 'adaptive' else 4
    print(f'  Parameters: {n_params:,} (W: {n_neurons * K_actual:,}, delays: {n_neurons * K_actual * max_delay:,}, bias: {n_neurons:,}, membrane+threshold: {membrane_threshold_params:,})')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7,
    )

    # Early stopping tracks the combined spike+voltage objective rather than connectivity AUC alone.
    print('\n  Training with event windows...')
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    train_history = []
    val_history = []
    conn_aucs = []
    t0 = time.time()

    for epoch in range(n_epochs):
        train_stats = train_epoch_events(
            model, train_loader, optimizer, device, warmup,
            pos_weight, l1_lambda, voltage_lambda,
        )
        train_history.append(train_stats)

        val_stats = evaluate_event_windows(
            model, val_loader, device, warmup,
            pos_weight, l1_lambda, voltage_lambda,
        )
        val_history.append(val_stats)

        conn_results, _, _, _, _, _ = evaluate_connectivity(
            model, neighbor_indices, true_binary, true_weights,
        )
        conn_aucs.append(conn_results['auc'])
        scheduler.step(val_stats['loss'])

        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            alpha = torch.sigmoid(model.alpha_logit).item()
            thresh_inc = model.threshold_increment.mean().item()
            thresh_decay = model.threshold_decay.item()
            elapsed = time.time() - t0
            print(
                f'    Epoch {epoch + 1:3d}: '
                f'train={train_stats["loss"]:.4f} '
                f'(spike={train_stats["spike_loss"]:.4f} voltage={train_stats["voltage_loss"]:.4f} l1={train_stats["l1_loss"]:.4f}) '
                f'val={val_stats["loss"]:.4f} conn_AUC={conn_results["auc"]:.4f} '
                f'alpha={alpha:.3f} theta_mode={model.threshold_mode} theta0={model.threshold.item():.3f} eta={thresh_inc:.3f} rho={thresh_decay:.3f} ({elapsed:.0f}s)'
            )

        if epochs_no_improve >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'  Done in {time.time() - t0:.0f}s, best val loss={best_val_loss:.4f}')

    val_window_results = evaluate_event_windows(
        model, val_loader, device, warmup,
        pos_weight, l1_lambda, voltage_lambda,
    )

    surrogate_score_sets = None
    if connectivity_threshold_mode == 'surrogate_fdr':
        print(
            f'\n  Calibrating non-leaky connectivity threshold with '
            f'{n_threshold_surrogates} circular-shift surrogates '
            f'(target FDR={surrogate_fdr:.3f})...'
        )
        surrogate_score_sets = estimate_surrogate_connectivity_score_sets(
            spike_matrix,
            voltage_matrix,
            voltage_mask,
            neighbor_indices,
            n_neurons=n_neurons,
            K_actual=K_actual,
            max_delay=max_delay,
            threshold_mode=threshold_mode,
            lr=lr,
            batch_size=batch_size,
            warmup=warmup,
            pos_weight=pos_weight,
            l1_lambda=l1_lambda,
            voltage_lambda=voltage_lambda,
            pre_context=pre_context,
            post_context=post_context,
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
            f'  Surrogate score sets: {surrogate_score_sets.shape[0]} models x '
            f'{surrogate_score_sets.shape[1]} edges'
        )
    all_results, all_abs_scores, all_labels, all_signed_scores, all_true_weights, conn_matrix = evaluate_connectivity(
        model, neighbor_indices, true_binary, true_weights,
        connectivity_threshold_mode=connectivity_threshold_mode,
        surrogate_score_sets=surrogate_score_sets,
        surrogate_fdr=surrogate_fdr,
    )

    print(f"\n  {'='*50}")
    print('  HELD-OUT WINDOW VALIDATION')
    print(f"  {'='*50}")
    print(f'  Strategy:      {validation_strategy}')
    print(f'  Loss:          {val_window_results["loss"]:.4f}')
    print(f'  Spike loss:    {val_window_results["spike_loss"]:.4f}')
    print(f'  Voltage loss:  {val_window_results["voltage_loss"]:.4f}')
    print(f'  L1 loss:       {val_window_results["l1_loss"]:.4f}')
    print(f'  Windows:       {val_window_results["n_windows"]}')

    print('\n  CONNECTIVITY RESULTS (all fitted neurons)')
    print(f'  AUC:           {all_results["auc"]:.4f}')
    print(f'  AP:            {all_results["ap"]:.4f}')
    print(f'  F1:            {all_results["f1"]:.4f}')
    print(f'  Threshold rule: {all_results.get("connectivity_threshold_mode", connectivity_threshold_mode)}')
    if all_results.get('estimated_fdr') is not None:
        print(
            f'  Estimated FDR: {all_results["estimated_fdr"]:.4f} '
            f'(target {all_results.get("surrogate_fdr_target", surrogate_fdr):.4f})'
        )
    print(f'  Sign accuracy: {all_results["sign_accuracy"]:.4f}')
    print(f'  Weight corr:   {all_results["weight_corr"]:.4f}')

    output_dir = os.path.join(PROJECT_ROOT, 'voltage_augmented_learned_lif_outputs')
    plot_results(
        all_results, all_abs_scores, all_labels, all_signed_scores,
        conn_matrix, train_history, val_history, conn_aucs,
        val_window_results, positions, connections, neighbor_indices,
        model, session_name, output_name, output_dir,
    )
    plot_score_separation_histogram(
        all_abs_scores,
        all_labels,
        os.path.join(
            output_dir,
            f'voltage_augmented_learned_lif_{output_name}_score_separation.png',
        ),
        threshold=all_results.get('threshold'),
        threshold_label=all_results.get('connectivity_threshold_mode'),
        title=f'Voltage-Augmented Score Separation - {session_name}',
    )

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'voltage_augmented_learned_lif_{output_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'K': K_actual,
        'T': T,
        'dt': dt,
        'max_delay': max_delay,
        'n_neurons': n_neurons,
        'session_name': session_name,
        'output_name': output_name,
        'validation_strategy': validation_strategy,
        'candidate_info': candidate_info,
        'threshold_mode': model.threshold_mode,
        'connectivity_threshold_mode': connectivity_threshold_mode,
        'neighbor_indices': neighbor_indices,
        'connectivity_matrix': conn_matrix,
        'results_window_val': val_window_results,
        'results_all': all_results,
        'train_history': train_history,
        'val_history': val_history,
        'connectivity_aucs': conn_aucs,
        'recording_summaries': data['recording_summaries'],
        'saved_burst_onset_bins': saved_burst_onset_bins,
        'detected_burst_windows': detected_burst_info['windows'],
        'excluded_bins': excluded_bins,
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
        },
        'adaptive_threshold': {
            'threshold_mode': model.threshold_mode,
            'threshold_base_mean': float(model.threshold.item()),
            'threshold_base_std': float(model.threshold_base_values.std(unbiased=False).item()),
            'threshold_increment_mean': float(model.threshold_increment.mean().item()),
            'threshold_decay': float(model.threshold_decay.item()),
        },
        'voltage_cleaning': {
            'mask_pre_ms': mask_pre_ms,
            'mask_post_ms': mask_post_ms,
            'peak_threshold_mv': peak_threshold_mv,
            'voltage_lambda': voltage_lambda,
        },
    }, model_path)
    print(f'  Model + connectivity saved: {model_path}')

    conn_path = os.path.join(output_dir, f'connectivity_{output_name}.npz')
    np.savez_compressed(
        conn_path,
        connectivity_matrix=conn_matrix,
        threshold=all_results.get('threshold', 0.5),
        connectivity_threshold_mode=connectivity_threshold_mode,
        estimated_fdr=all_results.get('estimated_fdr'),
        neighbor_indices=neighbor_indices,
        neuron_positions=positions,
        true_weights=true_weights,
        saved_burst_onset_bins=saved_burst_onset_bins,
        detected_burst_windows=detected_burst_info['windows'],
        excluded_bins=excluded_bins,
    )
    print(f'  Connectivity matrix saved: {conn_path}')

    return all_results, conn_matrix


def select_session():
    """Interactively choose a saved session from the default output folder.

    Args:
        None.

    Returns:
        The filesystem path of the selected session directory.
    """
    data_dir = os.path.join(PROJECT_ROOT, 'LIF data')
    sessions = sorted([s for s in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(s)])
    if not sessions:
        print('No sessions in LIF data/')
        sys.exit(1)

    print('\nSessions:')
    for i, s in enumerate(sessions):
        print(f'  [{i}] {os.path.basename(s)}')
    choice = input('Select (Enter=first): ').strip()
    return sessions[0] if choice == '' else sessions[int(choice)]


def build_parser():
    """Build the CLI parser for the voltage-augmented learned-LIF pipeline.

    Args:
        None.

    Returns:
        An ``argparse.ArgumentParser`` configured for the voltage-augmented
        inference CLI.
    """
    parser = argparse.ArgumentParser(description='Voltage-augmented learned LIF connectivity')
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional suffix for saved artifact names')
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--max-delay', type=int, default=8)
    parser.add_argument('--l1', type=float, default=0.01)
    parser.add_argument('--pos-weight', type=float, default=5.0)
    parser.add_argument('--voltage-lambda', type=float, default=1.0,
                        help='Weight on masked subthreshold voltage loss')
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--recording', type=int, default=0)
    parser.add_argument('--single-recording', action='store_true',
                        help='Use only one recording instead of all')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Use only first N ms for faster testing')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--candidate-mode', type=str, default='hybrid',
                        choices=['spatial', 'hybrid'])
    parser.add_argument('--candidate-spatial-frac', type=float, default=0.8)
    parser.add_argument('--candidate-min-lag', type=int, default=1)
    parser.add_argument('--candidate-max-lag', type=int, default=None)
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
    parser.add_argument('--pre-context', type=int, default=50)
    parser.add_argument('--post-context', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--neg-ratio', type=float, default=1.0)
    parser.add_argument('--neg-min-dist', type=int, default=100)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--mask-pre-ms', type=float, default=1.0)
    parser.add_argument('--mask-post-ms', type=float, default=5.0)
    parser.add_argument('--peak-threshold-mv', type=float, default=15.0)
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
    return parser


def main(argv=None):
    """Parse CLI arguments and launch the voltage-augmented learned-LIF pipeline.

    Args:
        argv: Optional CLI argument list. When omitted, arguments are read from
            ``sys.argv``.

    Returns:
        None. The function resolves the session and runs the voltage-augmented
        inference pipeline.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    session_dir = args.session if args.session else select_session()

    run_pipeline(
        session_dir, K=args.k, recording_idx=args.recording,
        n_epochs=args.epochs, lr=args.lr, batch_size=args.batch,
        patience=args.patience, dt=args.dt, max_delay=args.max_delay,
        l1_lambda=args.l1, pos_weight=args.pos_weight,
        voltage_lambda=args.voltage_lambda,
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
        threshold_mode=args.threshold_mode,
        connectivity_threshold_mode=args.connectivity_threshold_mode,
        surrogate_fdr=args.surrogate_fdr,
        n_threshold_surrogates=args.n_threshold_surrogates,
        surrogate_epochs=args.surrogate_epochs,
        surrogate_patience=args.surrogate_patience,
        surrogate_min_shift_fraction=args.surrogate_min_shift_frac,
        surrogate_seed=args.surrogate_seed,
        mask_pre_ms=args.mask_pre_ms,
        mask_post_ms=args.mask_post_ms,
        peak_threshold_mv=args.peak_threshold_mv,
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


if __name__ == '__main__':
    main()