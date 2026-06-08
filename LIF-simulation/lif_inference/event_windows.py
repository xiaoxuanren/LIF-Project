"""Event-window sampling and dataset helpers for learned-LIF inference."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .burst_exclusion import window_overlaps_excluded_bins


def find_event_windows(spike_matrix, neuron_id, pre_context=50, post_context=10,
                       warmup=30, neg_ratio=1.0, neg_min_distance=100,
                       boundaries=None, excluded_bins=None, rng=None):
    """Extract positive (spike-centered) and negative (no-spike) event windows."""
    if rng is None:
        rng = np.random.RandomState(42 + int(neuron_id))

    total_bins = spike_matrix.shape[1]

    if boundaries is None:
        boundaries = [0, total_bins]

    post_spikes = np.where(spike_matrix[neuron_id] == 1)[0]

    valid_ranges = []
    for idx in range(len(boundaries) - 1):
        rec_start = boundaries[idx]
        rec_end = boundaries[idx + 1]
        min_t = rec_start + warmup + pre_context
        max_t = rec_end - post_context
        if max_t > min_t:
            valid_ranges.append((min_t, max_t))

    pos_windows = []
    for spike_bin in post_spikes:
        for range_start, range_end in valid_ranges:
            if range_start <= spike_bin < range_end:
                start = spike_bin - pre_context - warmup
                end = spike_bin + post_context
                if not window_overlaps_excluded_bins(start, end, excluded_bins):
                    pos_windows.append((int(start), int(end)))
                break

    n_pos = len(pos_windows)
    n_neg = int(n_pos * neg_ratio)
    neg_windows = []

    if n_neg > 0 and valid_ranges:
        attempts = 0
        max_attempts = n_neg * 200

        while len(neg_windows) < n_neg and attempts < max_attempts:
            attempts += 1
            range_idx = rng.randint(len(valid_ranges))
            range_start, range_end = valid_ranges[range_idx]
            if range_end <= range_start:
                continue
            sample_t = rng.randint(range_start, range_end)

            nearby_min = max(0, sample_t - neg_min_distance)
            nearby_max = min(total_bins, sample_t + neg_min_distance)
            if spike_matrix[neuron_id, nearby_min:nearby_max].sum() == 0:
                start = sample_t - pre_context - warmup
                end = sample_t + post_context
                if not window_overlaps_excluded_bins(start, end, excluded_bins):
                    neg_windows.append((int(start), int(end)))

    return pos_windows, neg_windows


def find_pre_centered_event_windows(spike_matrix, neighbor_indices, neuron_id,
                                    pre_context=50, post_context=10,
                                    warmup=30, neg_ratio=1.0,
                                    pre_min_lag=1, pre_max_lag=8,
                                    boundaries=None, excluded_bins=None,
                                    rng=None, max_anchors=None):
    """Extract windows anchored on candidate presynaptic spike bins."""
    if rng is None:
        rng = np.random.RandomState(4242 + int(neuron_id))

    total_bins = spike_matrix.shape[1]
    if boundaries is None:
        boundaries = [0, total_bins]

    candidate_pre_ids = np.asarray(neighbor_indices[neuron_id], dtype=np.int32)
    if candidate_pre_ids.size == 0:
        return [], []

    pre_max_lag = max(int(pre_min_lag), int(pre_max_lag))
    post_spikes = np.flatnonzero(spike_matrix[neuron_id])
    if post_spikes.size == 0:
        return [], []

    valid_ranges = []
    for rec_idx in range(len(boundaries) - 1):
        rec_start = int(boundaries[rec_idx])
        rec_end = int(boundaries[rec_idx + 1])
        min_t = rec_start + warmup + pre_context
        max_t = rec_end - post_context
        if max_t > min_t:
            valid_ranges.append((min_t, max_t))

    if not valid_ranges:
        return [], []

    trigger_bins = np.flatnonzero(np.any(spike_matrix[candidate_pre_ids] > 0, axis=0))
    pos_anchors = []
    neg_anchors = []

    for trigger_bin in trigger_bins:
        trigger_bin = int(trigger_bin)
        in_valid_range = False
        for range_start, range_end in valid_ranges:
            if range_start <= trigger_bin < range_end:
                in_valid_range = True
                break
        if not in_valid_range:
            continue

        start = trigger_bin - pre_context - warmup
        end = trigger_bin + post_context
        if window_overlaps_excluded_bins(start, end, excluded_bins):
            continue

        left = np.searchsorted(post_spikes, trigger_bin + int(pre_min_lag), side='left')
        has_post_response = (
            left < len(post_spikes) and
            int(post_spikes[left]) <= trigger_bin + pre_max_lag
        )
        if has_post_response:
            pos_anchors.append(trigger_bin)
        else:
            neg_anchors.append(trigger_bin)

    if not pos_anchors:
        return [], []

    if max_anchors is not None and len(pos_anchors) > int(max_anchors):
        order = rng.permutation(len(pos_anchors))[:int(max_anchors)]
        pos_anchors = [pos_anchors[int(idx)] for idx in np.sort(order)]

    n_neg = min(len(neg_anchors), int(round(len(pos_anchors) * float(neg_ratio))))
    if n_neg > 0:
        neg_order = rng.permutation(len(neg_anchors))[:n_neg]
        neg_anchors = [neg_anchors[int(idx)] for idx in np.sort(neg_order)]
    else:
        neg_anchors = []

    pos_windows = [
        (int(trigger_bin - pre_context - warmup), int(trigger_bin + post_context))
        for trigger_bin in pos_anchors
    ]
    neg_windows = [
        (int(trigger_bin - pre_context - warmup), int(trigger_bin + post_context))
        for trigger_bin in neg_anchors
    ]
    return pos_windows, neg_windows


class EventWindowDataset(Dataset):
    """Event-window dataset for focused training."""

    def __init__(self, spike_matrix, neighbor_indices, neuron_ids=None,
                 pre_context=50, post_context=10, warmup=30,
                 neg_ratio=1.0, neg_min_distance=100, boundaries=None,
                 excluded_bins=None, event_anchor_mode='post',
                 pre_event_min_lag=1, pre_event_max_lag=8,
                 pre_event_max_anchors=None, rng_seed=42, windows=None):
        self.spike_matrix = spike_matrix
        self.neighbor_indices = neighbor_indices
        self.pre_context = pre_context
        self.post_context = post_context
        self.warmup = warmup
        self.window_len = warmup + pre_context + post_context
        self.event_anchor_mode = str(event_anchor_mode).strip().lower()
        if self.event_anchor_mode not in {'post', 'pre', 'both'}:
            raise ValueError(
                f"Unsupported event_anchor_mode={event_anchor_mode!r}; use 'post', 'pre', or 'both'"
            )
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
                if self.event_anchor_mode in {'post', 'both'}:
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

                if self.event_anchor_mode in {'pre', 'both'}:
                    pos_w, neg_w = find_pre_centered_event_windows(
                        spike_matrix, neighbor_indices, post_id,
                        pre_context=pre_context,
                        post_context=post_context,
                        warmup=warmup,
                        neg_ratio=neg_ratio,
                        pre_min_lag=pre_event_min_lag,
                        pre_max_lag=pre_event_max_lag,
                        boundaries=boundaries,
                        excluded_bins=self.excluded_bins,
                        rng=rng,
                        max_anchors=pre_event_max_anchors,
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
    """Split event windows into train and validation subsets."""
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
    """Split concatenated recording boundaries into train and validation groups."""
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
                                   event_anchor_mode='post',
                                   pre_event_min_lag=1,
                                   pre_event_max_lag=8,
                                   pre_event_max_anchors=None,
                                   rng_seed=42):
    """Build train and validation event datasets for the per-neuron model."""
    train_boundaries, val_boundaries = split_recording_boundaries(boundaries, val_fraction)

    if val_boundaries is not None:
        train_ds = EventWindowDataset(
            spike_matrix, neighbor_indices, neuron_ids=neuron_ids,
            pre_context=pre_context, post_context=post_context, warmup=warmup,
            neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
            boundaries=train_boundaries, excluded_bins=excluded_bins,
            event_anchor_mode=event_anchor_mode,
            pre_event_min_lag=pre_event_min_lag,
            pre_event_max_lag=pre_event_max_lag,
            pre_event_max_anchors=pre_event_max_anchors,
            rng_seed=rng_seed,
        )
        val_ds = EventWindowDataset(
            spike_matrix, neighbor_indices, neuron_ids=neuron_ids,
            pre_context=pre_context, post_context=post_context, warmup=warmup,
            neg_ratio=neg_ratio, neg_min_distance=neg_min_distance,
            boundaries=val_boundaries, excluded_bins=excluded_bins,
            event_anchor_mode=event_anchor_mode,
            pre_event_min_lag=pre_event_min_lag,
            pre_event_max_lag=pre_event_max_lag,
            pre_event_max_anchors=pre_event_max_anchors,
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
        boundaries=boundaries, excluded_bins=excluded_bins,
        event_anchor_mode=event_anchor_mode,
        pre_event_min_lag=pre_event_min_lag,
        pre_event_max_lag=pre_event_max_lag,
        pre_event_max_anchors=pre_event_max_anchors,
        rng_seed=rng_seed,
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
    strategy = 'held-out event windows'
    return train_ds, val_ds, strategy