import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lif_simulation.voltage_storage import resolve_recording_voltage
from lif_inference.burst_exclusion import merge_excluded_windows


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunningTraceStats:
    def __init__(self, length):
        self.sum_values = np.zeros(int(length), dtype=np.float64)
        self.sum_squares = np.zeros(int(length), dtype=np.float64)
        self.counts = np.zeros(int(length), dtype=np.int64)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        self.sum_values[valid] += values[valid]
        self.sum_squares[valid] += values[valid] ** 2
        self.counts[valid] += 1

    def mean(self):
        result = np.full_like(self.sum_values, np.nan, dtype=np.float64)
        valid = self.counts > 0
        result[valid] = self.sum_values[valid] / self.counts[valid]
        return result

    def sem(self):
        result = np.full_like(self.sum_values, np.nan, dtype=np.float64)
        valid = self.counts > 1
        means = self.mean()
        variances = np.zeros_like(self.sum_values, dtype=np.float64)
        variances[valid] = (
            self.sum_squares[valid] / self.counts[valid] - means[valid] ** 2
        )
        variances = np.maximum(variances, 0.0)
        result[valid] = np.sqrt(variances[valid] / self.counts[valid])
        return result

    def pooled_variance(self, mask):
        mask = np.asarray(mask, dtype=bool) & (self.counts > 1)
        if not np.any(mask):
            return None
        means = self.mean()
        variances = self.sum_squares[mask] / self.counts[mask] - means[mask] ** 2
        variances = np.maximum(variances, 0.0)
        weights = self.counts[mask].astype(np.float64)
        return float(np.average(variances, weights=weights))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze postsynaptic voltage changes around presynaptic spikes for true connected pairs.'
    )
    parser.add_argument('--session', type=str, default='LIF data/20260531_041642')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-pairs-per-type', type=int, default=250)
    parser.add_argument('--max-events-per-pair', type=int, default=80)
    parser.add_argument('--pre-ms', type=float, default=10.0)
    parser.add_argument('--post-ms', type=float, default=35.0)
    parser.add_argument('--response-ms', type=float, default=25.0)
    parser.add_argument('--baseline-start-ms', type=float, default=-5.0)
    parser.add_argument('--baseline-end-ms', type=float, default=-1.0)
    parser.add_argument('--mask-post-spike-pre-ms', type=float, default=0.0)
    parser.add_argument('--mask-post-spike-post-ms', type=float, default=2.0)
    parser.add_argument('--peak-threshold-mv', type=float, default=15.0)
    parser.add_argument('--analysis-dt-ms', type=float, default=1.0)
    parser.add_argument('--reference-max-delay-ms', type=float, default=8.0)
    parser.add_argument('--include-saved-stimulation', action='store_true')
    parser.add_argument('--include-detected-bursts', action='store_true')
    parser.add_argument('--burst-activity-bin-ms', type=float, default=100.0)
    parser.add_argument('--burst-smooth-bins', type=int, default=3)
    parser.add_argument('--burst-threshold-std', type=float, default=3.0)
    parser.add_argument('--burst-min-active-frac', type=float, default=0.10)
    parser.add_argument('--burst-min-duration-ms', type=float, default=100.0)
    parser.add_argument('--burst-merge-gap-ms', type=float, default=150.0)
    parser.add_argument('--burst-pad-before-ms', type=float, default=100.0)
    parser.add_argument('--burst-pad-after-ms', type=float, default=250.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def resolve_session_dir(raw_session):
    session_path = Path(raw_session)
    if not session_path.is_absolute():
        session_path = REPO_ROOT / session_path
    if not session_path.exists():
        raise FileNotFoundError(f'Session directory not found: {session_path}')
    return session_path


def load_network(session_dir):
    network_files = sorted(session_dir.glob('network_*.npz'))
    if not network_files:
        raise FileNotFoundError(f'No network_*.npz file found in {session_dir}')
    with np.load(network_files[0], allow_pickle=True) as network_data:
        return np.asarray(network_data['connections'], dtype=object)


def scan_recording_spike_counts(recording_files):
    total_counts = None
    recording_summaries = []
    for recording_file in recording_files:
        with np.load(recording_file, allow_pickle=True) as recording_data:
            spike_times = recording_data['spike_times']
            counts = np.asarray([len(spikes) for spikes in spike_times], dtype=np.int64)
            if total_counts is None:
                total_counts = counts.copy()
            else:
                total_counts += counts
            recording_summaries.append({
                'path': str(recording_file),
                'duration_ms': float(recording_data['duration']),
                'spike_count': int(counts.sum()),
                'voltage_sample_rate_ms': float(recording_data['voltage_sample_rate']),
                'voltage_n_samples': int(recording_data['voltage_n_samples']),
            })
    return total_counts, recording_summaries


def connection_kind(connection_row):
    if len(connection_row) >= 4:
        label = str(connection_row[3]).lower()
        if label.startswith('exc'):
            return 'exc'
        if label.startswith('inh'):
            return 'inh'
    return 'exc' if float(connection_row[2]) > 0.0 else 'inh'


def select_connected_pairs(connections, spike_counts, max_pairs_per_type):
    selected = []
    for target_kind in ('exc', 'inh'):
        kind_rows = []
        for connection_row in connections:
            if connection_kind(connection_row) != target_kind:
                continue
            presynaptic_id = int(connection_row[0])
            postsynaptic_id = int(connection_row[1])
            weight = float(connection_row[2])
            activity_score = float(spike_counts[presynaptic_id]) * max(abs(weight), 1e-9)
            kind_rows.append((activity_score, presynaptic_id, postsynaptic_id, weight, target_kind))
        kind_rows.sort(reverse=True)
        for _, presynaptic_id, postsynaptic_id, weight, kind in kind_rows[:max_pairs_per_type]:
            selected.append({
                'pre': presynaptic_id,
                'post': postsynaptic_id,
                'weight': weight,
                'kind': kind,
                'source': 'connected',
            })
    return selected


def select_null_pairs(connected_pairs, connections, n_neurons, rng):
    connected_lookup = {(int(row[0]), int(row[1])) for row in connections}
    null_pairs = []
    for connected_pair in connected_pairs:
        presynaptic_id = int(connected_pair['pre'])
        candidate_posts = [
            postsynaptic_id for postsynaptic_id in range(n_neurons)
            if postsynaptic_id != presynaptic_id
            and (presynaptic_id, postsynaptic_id) not in connected_lookup
        ]
        if not candidate_posts:
            continue
        postsynaptic_id = int(rng.choice(candidate_posts))
        null_pairs.append({
            'pre': presynaptic_id,
            'post': postsynaptic_id,
            'weight': 0.0,
            'kind': 'null',
            'source': f'null_matched_to_{connected_pair["kind"]}',
        })
    return null_pairs


def find_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def detect_recording_burst_windows_ms(spike_times, duration_ms, args):
    n_neurons = len(spike_times)
    activity_bin_ms = float(args.burst_activity_bin_ms)
    n_bins = int(np.ceil(float(duration_ms) / activity_bin_ms))
    active = np.zeros((n_neurons, n_bins), dtype=bool)
    for neuron_id, neuron_spikes in enumerate(spike_times):
        neuron_spikes = np.asarray(neuron_spikes, dtype=np.float64)
        if neuron_spikes.size == 0:
            continue
        bins = np.floor(neuron_spikes / activity_bin_ms + 1e-9).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < n_bins)]
        active[neuron_id, np.unique(bins)] = True
    active_fraction = active.mean(axis=0).astype(np.float32)
    smooth_bins = max(int(args.burst_smooth_bins), 1)
    kernel = np.ones(smooth_bins, dtype=np.float32) / float(smooth_bins)
    smoothed = np.convolve(active_fraction, kernel, mode='same')
    threshold = max(
        float(np.mean(smoothed) + float(args.burst_threshold_std) * np.std(smoothed)),
        float(args.burst_min_active_frac),
    )
    min_duration_ms = float(args.burst_min_duration_ms)
    raw_windows = []
    for start_bin, end_bin in find_true_segments(smoothed >= threshold):
        start_ms = start_bin * activity_bin_ms
        end_ms = min(float(duration_ms), end_bin * activity_bin_ms)
        if end_ms - start_ms >= min_duration_ms:
            raw_windows.append((start_ms, end_ms))
    if not raw_windows:
        return np.zeros((0, 2), dtype=np.float64), threshold
    raw_bins = np.asarray(
        [[int(round(start)), int(round(end))] for start, end in raw_windows],
        dtype=np.int32,
    )
    merged = merge_excluded_windows(raw_bins, max_gap_bins=int(round(float(args.burst_merge_gap_ms))))
    padded = [
        (
            max(0.0, float(start) - float(args.burst_pad_before_ms)),
            min(float(duration_ms), float(end) + float(args.burst_pad_after_ms)),
        )
        for start, end in merged
    ]
    return np.asarray(padded, dtype=np.float64), threshold


def build_exclusion_windows(recording_files, args):
    windows_by_recording = []
    summaries = []
    for recording_file in recording_files:
        with np.load(recording_file, allow_pickle=True) as recording_data:
            duration_ms = float(recording_data['duration'])
            windows = []
            n_saved = 0
            if not args.include_saved_stimulation and 'burst_onset_times' in recording_data.files:
                burst_onsets = np.asarray(recording_data['burst_onset_times'], dtype=np.float64)
                burst_onsets = burst_onsets[(burst_onsets >= 0.0) & (burst_onsets < duration_ms)]
                n_saved = int(burst_onsets.size)
                for onset_ms in burst_onsets:
                    windows.append((
                        max(0.0, float(onset_ms) - float(args.burst_pad_before_ms)),
                        min(duration_ms, float(onset_ms) + float(args.burst_pad_after_ms)),
                    ))
            n_detected = 0
            burst_threshold = None
            if not args.include_detected_bursts:
                detected_windows, burst_threshold = detect_recording_burst_windows_ms(
                    recording_data['spike_times'], duration_ms, args,
                )
                n_detected = int(len(detected_windows))
                windows.extend((float(start), float(end)) for start, end in detected_windows)
            if windows:
                windows_array = np.asarray(
                    [[int(round(start)), int(round(end))] for start, end in windows],
                    dtype=np.int32,
                )
                windows_array = merge_excluded_windows(windows_array, max_gap_bins=0).astype(np.float64)
            else:
                windows_array = np.zeros((0, 2), dtype=np.float64)
            windows_by_recording.append(windows_array)
            summaries.append({
                'path': str(recording_file),
                'saved_stimulation_onsets': n_saved,
                'detected_burst_windows': n_detected,
                'combined_exclusion_windows': int(len(windows_array)),
                'detected_burst_threshold': None if burst_threshold is None else float(burst_threshold),
            })
    return windows_by_recording, summaries


def overlaps_excluded_window(start_ms, end_ms, excluded_windows):
    if excluded_windows is None or len(excluded_windows) == 0:
        return False
    starts = excluded_windows[:, 0]
    ends = excluded_windows[:, 1]
    return bool(np.any((starts < float(end_ms)) & (ends > float(start_ms))))


def sample_bins(spike_times, sample_rate_ms):
    spike_times = np.asarray(spike_times, dtype=np.float64)
    return np.floor(spike_times / float(sample_rate_ms) + 1e-9).astype(np.int64)


def plan_events(recording_files, pairs, pre_ms, post_ms, max_events_per_pair, rng,
                exclusion_windows_by_recording=None):
    event_plans = {pair_index: [] for pair_index in range(len(pairs))}
    for recording_index, recording_file in enumerate(recording_files):
        with np.load(recording_file, allow_pickle=True) as recording_data:
            spike_times = recording_data['spike_times']
            sample_rate_ms = float(recording_data['voltage_sample_rate'])
            n_samples = int(recording_data['voltage_n_samples'])
            pre_bins = int(round(float(pre_ms) / sample_rate_ms))
            post_bins = int(round(float(post_ms) / sample_rate_ms))
            excluded_windows = None
            if exclusion_windows_by_recording is not None:
                excluded_windows = exclusion_windows_by_recording[recording_index]
            for pair_index, pair in enumerate(pairs):
                presynaptic_spikes = np.asarray(spike_times[int(pair['pre'])], dtype=np.float64)
                if presynaptic_spikes.size == 0:
                    continue
                presynaptic_bins = sample_bins(presynaptic_spikes, sample_rate_ms)
                valid = (presynaptic_bins >= pre_bins) & (presynaptic_bins + post_bins < n_samples)
                for event_time_ms, event_bin in zip(presynaptic_spikes[valid], presynaptic_bins[valid]):
                    if overlaps_excluded_window(
                            float(event_time_ms) - float(pre_ms),
                            float(event_time_ms) + float(post_ms),
                            excluded_windows):
                        continue
                    event_plans[pair_index].append((recording_index, float(event_time_ms), int(event_bin)))

    sampled_plans = {}
    for pair_index, events in event_plans.items():
        if len(events) > max_events_per_pair:
            selected_indices = rng.choice(len(events), size=max_events_per_pair, replace=False)
            sampled_plans[pair_index] = [events[int(index)] for index in np.sort(selected_indices)]
        else:
            sampled_plans[pair_index] = events
    return sampled_plans


def mask_postsynaptic_spikes(valid_mask, event_time_ms, post_spike_times,
                             sample_rate_ms, pre_bins, mask_pre_ms, mask_post_ms):
    if post_spike_times.size == 0:
        return
    window_start_ms = event_time_ms - pre_bins * sample_rate_ms
    window_end_ms = window_start_ms + (len(valid_mask) - 1) * sample_rate_ms
    local_spikes = post_spike_times[
        (post_spike_times >= window_start_ms - mask_pre_ms)
        & (post_spike_times <= window_end_ms + mask_post_ms)
    ]
    mask_pre_bins = int(round(float(mask_pre_ms) / sample_rate_ms))
    mask_post_bins = int(round(float(mask_post_ms) / sample_rate_ms))
    for spike_time_ms in local_spikes:
        spike_bin = int(np.floor((float(spike_time_ms) - window_start_ms) / sample_rate_ms + 1e-9))
        start = max(0, spike_bin - mask_pre_bins)
        end = min(len(valid_mask), spike_bin + mask_post_bins + 1)
        valid_mask[start:end] = False


def downsample_trace(values, times_ms, factor, method):
    usable = (len(values) // factor) * factor
    if usable <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    trimmed_values = np.asarray(values[:usable], dtype=np.float64)
    trimmed_times = np.asarray(times_ms[:usable], dtype=np.float64)
    value_blocks = trimmed_values.reshape(-1, factor)
    time_blocks = trimmed_times.reshape(-1, factor)
    if method == 'mean':
        valid = np.isfinite(value_blocks)
        valid_counts = valid.sum(axis=1)
        coarse = np.full(value_blocks.shape[0], np.nan, dtype=np.float64)
        enough = valid_counts / float(factor) > 0.5
        if np.any(enough):
            coarse[enough] = np.nansum(value_blocks[enough], axis=1) / valid_counts[enough]
    elif method == 'stride':
        coarse = value_blocks[:, 0].copy()
    else:
        raise ValueError(f'Unknown downsample method: {method}')
    coarse_times = time_blocks.mean(axis=1)
    return coarse, coarse_times


def add_event_window(trace_stats_by_kind, downsample_stats_by_kind, pair_stats,
                     pair_index, pair, delta_trace, time_axis_ms,
                     analysis_factor):
    pair_stats[pair_index].add(delta_trace)
    trace_stats_by_kind[pair['kind']].add(delta_trace)
    for method in ('mean', 'stride'):
        coarse_values, coarse_times = downsample_trace(delta_trace, time_axis_ms, analysis_factor, method)
        if coarse_values.size > 0:
            downsample_stats_by_kind[pair['kind']][method].add(coarse_values)
            downsample_stats_by_kind[pair['kind']]['times_ms'] = coarse_times


def accumulate_windows(recording_files, pairs, event_plans, args, rng):
    first_recording = np.load(recording_files[0], allow_pickle=True)
    sample_rate_ms = float(first_recording['voltage_sample_rate'])
    first_recording.close()
    pre_bins = int(round(float(args.pre_ms) / sample_rate_ms))
    post_bins = int(round(float(args.post_ms) / sample_rate_ms))
    window_len = pre_bins + post_bins + 1
    time_axis_ms = (np.arange(window_len, dtype=np.float64) - pre_bins) * sample_rate_ms
    analysis_factor = int(round(float(args.analysis_dt_ms) / sample_rate_ms))
    if analysis_factor < 1 or not np.isclose(args.analysis_dt_ms / sample_rate_ms, analysis_factor):
        raise ValueError('analysis-dt-ms must be an integer multiple of the native voltage sample rate')

    coarse_len = window_len // analysis_factor
    trace_stats_by_kind = {kind: RunningTraceStats(window_len) for kind in ('exc', 'inh', 'null')}
    downsample_stats_by_kind = {
        kind: {
            'mean': RunningTraceStats(coarse_len),
            'stride': RunningTraceStats(coarse_len),
            'times_ms': None,
        }
        for kind in ('exc', 'inh', 'null')
    }
    pair_stats = {pair_index: RunningTraceStats(window_len) for pair_index in range(len(pairs))}
    event_counts = np.zeros(len(pairs), dtype=np.int64)
    skipped_events = 0

    events_by_recording = {}
    for pair_index, events in event_plans.items():
        for recording_index, event_time_ms, event_bin in events:
            events_by_recording.setdefault(recording_index, []).append((pair_index, event_time_ms, event_bin))

    for recording_index, recording_file in enumerate(recording_files):
        recording_events = events_by_recording.get(recording_index, [])
        if not recording_events:
            continue
        with np.load(recording_file, allow_pickle=True) as recording_data:
            spike_times = recording_data['spike_times']
            voltage_bundle = resolve_recording_voltage(
                recording_data,
                recording_path=str(recording_file),
                load_into_memory=False,
            )
            voltage_traces = voltage_bundle['traces']
            for pair_index, event_time_ms, event_bin in recording_events:
                pair = pairs[pair_index]
                postsynaptic_id = int(pair['post'])
                start = int(event_bin) - pre_bins
                end = int(event_bin) + post_bins + 1
                raw_trace = np.asarray(voltage_traces[postsynaptic_id, start:end], dtype=np.float32)
                if raw_trace.size != window_len:
                    skipped_events += 1
                    continue
                valid_mask = np.isfinite(raw_trace)
                if args.peak_threshold_mv is not None:
                    valid_mask &= raw_trace < float(args.peak_threshold_mv)
                post_spike_times = np.asarray(spike_times[postsynaptic_id], dtype=np.float64)
                mask_postsynaptic_spikes(
                    valid_mask,
                    event_time_ms,
                    post_spike_times,
                    sample_rate_ms,
                    pre_bins,
                    args.mask_post_spike_pre_ms,
                    args.mask_post_spike_post_ms,
                )
                baseline_mask = (
                    valid_mask
                    & (time_axis_ms >= float(args.baseline_start_ms))
                    & (time_axis_ms < float(args.baseline_end_ms))
                )
                if int(np.sum(baseline_mask)) < max(3, analysis_factor):
                    skipped_events += 1
                    continue
                baseline = float(np.mean(raw_trace[baseline_mask]))
                delta_trace = raw_trace.astype(np.float64) - baseline
                delta_trace[~valid_mask] = np.nan
                add_event_window(
                    trace_stats_by_kind,
                    downsample_stats_by_kind,
                    pair_stats,
                    pair_index,
                    pair,
                    delta_trace,
                    time_axis_ms,
                    analysis_factor,
                )
                event_counts[pair_index] += 1
            close_method = getattr(voltage_traces, 'close', None)
            if close_method is not None:
                close_method()

    return {
        'sample_rate_ms': sample_rate_ms,
        'time_axis_ms': time_axis_ms,
        'trace_stats_by_kind': trace_stats_by_kind,
        'downsample_stats_by_kind': downsample_stats_by_kind,
        'pair_stats': pair_stats,
        'event_counts': event_counts,
        'skipped_events': int(skipped_events),
    }


def response_stat_from_trace(time_axis_ms, mean_trace, kind, response_ms):
    response_mask = (time_axis_ms >= 0.0) & (time_axis_ms <= float(response_ms)) & np.isfinite(mean_trace)
    if not np.any(response_mask):
        return None
    response_times = time_axis_ms[response_mask]
    response_values = mean_trace[response_mask]
    if kind == 'inh':
        local_index = int(np.nanargmin(response_values))
    else:
        local_index = int(np.nanargmax(response_values))
    return {
        'delay_ms': float(response_times[local_index]),
        'amplitude_mv': float(response_values[local_index]),
    }


def summarize_pairs(pairs, pair_stats, event_counts, time_axis_ms, response_ms, reference_max_delay_ms):
    rows = []
    for pair_index, pair in enumerate(pairs):
        event_count = int(event_counts[pair_index])
        if event_count <= 0:
            continue
        mean_trace = pair_stats[pair_index].mean()
        stat = response_stat_from_trace(time_axis_ms, mean_trace, pair['kind'], response_ms)
        if stat is None:
            continue
        rows.append({
            'pair_index': pair_index,
            'pre': int(pair['pre']),
            'post': int(pair['post']),
            'kind': pair['kind'],
            'source': pair['source'],
            'weight': float(pair['weight']),
            'n_events': event_count,
            'peak_or_trough_delay_ms': stat['delay_ms'],
            'peak_or_trough_amplitude_mv': stat['amplitude_mv'],
            'after_reference_delay': bool(stat['delay_ms'] > float(reference_max_delay_ms)),
        })
    return rows


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        'n': int(values.size),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)),
    }


def build_summary(args, session_dir, recording_summaries, exclusion_summaries,
                  pairs, pair_rows, accumulation):
    time_axis_ms = accumulation['time_axis_ms']
    summary = {
        'session': session_dir.name,
        'session_dir': str(session_dir),
        'n_recordings': len(recording_summaries),
        'total_duration_s': float(sum(item['duration_ms'] for item in recording_summaries) / 1000.0),
        'native_voltage_sample_rate_ms': float(accumulation['sample_rate_ms']),
        'analysis_dt_ms': float(args.analysis_dt_ms),
        'reference_max_delay_ms': float(args.reference_max_delay_ms),
        'pre_ms': float(args.pre_ms),
        'post_ms': float(args.post_ms),
        'response_ms': float(args.response_ms),
        'selected_pairs': {
            kind: int(sum(1 for pair in pairs if pair['kind'] == kind))
            for kind in ('exc', 'inh', 'null')
        },
        'exclusion_policy': {
            'saved_stimulation_windows_excluded': not bool(args.include_saved_stimulation),
            'detected_burst_windows_excluded': not bool(args.include_detected_bursts),
            'pad_before_ms': float(args.burst_pad_before_ms),
            'pad_after_ms': float(args.burst_pad_after_ms),
            'saved_stimulation_onsets': int(sum(item['saved_stimulation_onsets'] for item in exclusion_summaries)),
            'detected_burst_windows': int(sum(item['detected_burst_windows'] for item in exclusion_summaries)),
            'combined_exclusion_windows': int(sum(item['combined_exclusion_windows'] for item in exclusion_summaries)),
        },
        'usable_events': {
            kind: int(sum(row['n_events'] for row in pair_rows if row['kind'] == kind))
            for kind in ('exc', 'inh', 'null')
        },
        'skipped_events': int(accumulation['skipped_events']),
        'pair_level': {},
        'population_trace': {},
        'downsample_comparison': {},
    }
    for kind in ('exc', 'inh', 'null'):
        kind_rows = [row for row in pair_rows if row['kind'] == kind]
        delays = [row['peak_or_trough_delay_ms'] for row in kind_rows]
        amplitudes = [row['peak_or_trough_amplitude_mv'] for row in kind_rows]
        events = [row['n_events'] for row in kind_rows]
        after_reference = [row for row in kind_rows if row['after_reference_delay']]
        summary['pair_level'][kind] = {
            'n_pairs': len(kind_rows),
            'events_per_pair': percentile_summary(events),
            'delay_ms': percentile_summary(delays),
            'amplitude_mv': percentile_summary(amplitudes),
            'fraction_delays_after_reference': (
                float(len(after_reference) / len(kind_rows)) if kind_rows else None
            ),
        }
        mean_trace = accumulation['trace_stats_by_kind'][kind].mean()
        stat = response_stat_from_trace(time_axis_ms, mean_trace, kind, args.response_ms)
        summary['population_trace'][kind] = stat
        downsample_entry = {}
        for method in ('mean', 'stride'):
            downsample_stats = accumulation['downsample_stats_by_kind'][kind][method]
            coarse_times = accumulation['downsample_stats_by_kind'][kind]['times_ms']
            if coarse_times is None:
                continue
            coarse_mean = downsample_stats.mean()
            downsample_entry[method] = {
                'response_stat': response_stat_from_trace(coarse_times, coarse_mean, kind, args.response_ms),
                'response_variance': downsample_stats.pooled_variance(
                    (coarse_times >= 0.0) & (coarse_times <= float(args.response_ms))
                ),
            }
        if 'mean' in downsample_entry and 'stride' in downsample_entry:
            mean_variance = downsample_entry['mean']['response_variance']
            stride_variance = downsample_entry['stride']['response_variance']
            if mean_variance is not None and mean_variance > 0 and stride_variance is not None:
                downsample_entry['stride_to_mean_variance_ratio'] = float(stride_variance / mean_variance)
        summary['downsample_comparison'][kind] = downsample_entry
    return summary


def write_pair_csv(path, rows):
    fieldnames = [
        'pair_index', 'pre', 'post', 'kind', 'source', 'weight', 'n_events',
        'peak_or_trough_delay_ms', 'peak_or_trough_amplitude_mv', 'after_reference_delay',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_population_traces(path, accumulation, args):
    time_axis_ms = accumulation['time_axis_ms']
    colors = {'exc': 'tab:red', 'inh': 'tab:blue', 'null': '0.35'}
    labels = {'exc': 'True excitatory edges', 'inh': 'True inhibitory edges', 'null': 'Matched unconnected pairs'}
    plt.figure(figsize=(10, 5.5))
    for kind in ('exc', 'inh', 'null'):
        stats = accumulation['trace_stats_by_kind'][kind]
        mean_trace = stats.mean()
        sem_trace = stats.sem()
        plt.plot(time_axis_ms, mean_trace, color=colors[kind], linewidth=1.8, label=labels[kind])
        valid = np.isfinite(sem_trace)
        plt.fill_between(
            time_axis_ms[valid],
            mean_trace[valid] - sem_trace[valid],
            mean_trace[valid] + sem_trace[valid],
            color=colors[kind],
            alpha=0.18,
            linewidth=0,
        )
    plt.axvline(0.0, color='black', linewidth=1.0, alpha=0.7, label='Presynaptic spike')
    plt.axvline(args.reference_max_delay_ms, color='darkorange', linestyle='--', linewidth=1.2, label='8 ms reference')
    plt.axvline(20.0, color='purple', linestyle=':', linewidth=1.2, label='20 ms')
    plt.axhline(0.0, color='black', linewidth=0.8, alpha=0.4)
    plt.xlabel('Time from presynaptic spike (ms)')
    plt.ylabel('Baseline-subtracted postsynaptic voltage (mV)')
    plt.title('Connected-Pair Postsynaptic Voltage Triggered By Presynaptic Spikes')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_delay_histogram(path, pair_rows, reference_max_delay_ms):
    plt.figure(figsize=(10, 5.5))
    for kind, color, label in (
            ('exc', 'tab:red', 'Excitatory peak delay'),
            ('inh', 'tab:blue', 'Inhibitory trough delay')):
        delays = [row['peak_or_trough_delay_ms'] for row in pair_rows if row['kind'] == kind]
        if delays:
            plt.hist(delays, bins=np.arange(0, 26, 1), alpha=0.55, color=color, label=label)
    plt.axvline(reference_max_delay_ms, color='darkorange', linestyle='--', linewidth=1.4, label='8 ms reference')
    plt.axvline(20.0, color='purple', linestyle=':', linewidth=1.4, label='20 ms')
    plt.xlabel('Pair-level peak/trough delay (ms)')
    plt.ylabel('Connected-pair count')
    plt.title('Connected-Pair PSP Delay Distribution')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_example_pairs(path, pairs, pair_rows, accumulation):
    time_axis_ms = accumulation['time_axis_ms']
    selected_rows = []
    for kind in ('exc', 'inh'):
        kind_rows = [row for row in pair_rows if row['kind'] == kind]
        if kind == 'exc':
            kind_rows.sort(key=lambda row: row['peak_or_trough_amplitude_mv'], reverse=True)
        else:
            kind_rows.sort(key=lambda row: row['peak_or_trough_amplitude_mv'])
        selected_rows.extend(kind_rows[:4])
    if not selected_rows:
        return
    n_rows = 2
    n_cols = 4
    plt.figure(figsize=(14, 6.5))
    for plot_index, row in enumerate(selected_rows[:n_rows * n_cols], start=1):
        pair = pairs[int(row['pair_index'])]
        mean_trace = accumulation['pair_stats'][int(row['pair_index'])].mean()
        color = 'tab:red' if pair['kind'] == 'exc' else 'tab:blue'
        plt.subplot(n_rows, n_cols, plot_index)
        plt.plot(time_axis_ms, mean_trace, color=color, linewidth=1.4)
        plt.axvline(0.0, color='black', linewidth=0.8, alpha=0.6)
        plt.axvline(8.0, color='darkorange', linestyle='--', linewidth=0.8)
        plt.axhline(0.0, color='black', linewidth=0.6, alpha=0.35)
        plt.title(
            f"{pair['kind']} {pair['pre']}->{pair['post']}\n"
            f"n={row['n_events']}, delay={row['peak_or_trough_delay_ms']:.1f} ms"
        )
        plt.xlabel('ms')
        plt.ylabel('Delta V (mV)')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_downsample_comparison(path, accumulation, args):
    plt.figure(figsize=(11, 6.5))
    for subplot_index, kind in enumerate(('exc', 'inh'), start=1):
        plt.subplot(1, 2, subplot_index)
        native_times = accumulation['time_axis_ms']
        native_mean = accumulation['trace_stats_by_kind'][kind].mean()
        plt.plot(native_times, native_mean, color='black', linewidth=1.5, alpha=0.65, label='Native 0.1 ms')
        for method, color, marker in (('mean', 'tab:green', 'o'), ('stride', 'tab:orange', 's')):
            downsample_stats = accumulation['downsample_stats_by_kind'][kind][method]
            coarse_times = accumulation['downsample_stats_by_kind'][kind]['times_ms']
            if coarse_times is None:
                continue
            plt.plot(
                coarse_times,
                downsample_stats.mean(),
                color=color,
                marker=marker,
                markersize=3,
                linewidth=1.2,
                label=f'{args.analysis_dt_ms:g} ms {method}',
            )
        plt.axvline(0.0, color='black', linewidth=0.8, alpha=0.6)
        plt.axvline(args.reference_max_delay_ms, color='darkorange', linestyle='--', linewidth=0.9)
        plt.axhline(0.0, color='black', linewidth=0.6, alpha=0.35)
        plt.title('EPSP downsample check' if kind == 'exc' else 'IPSP downsample check')
        plt.xlabel('Time from presynaptic spike (ms)')
        plt.ylabel('Delta V (mV)')
        plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def print_console_summary(summary, output_dir):
    print(f"Session: {summary['session']} ({summary['n_recordings']} recordings, {summary['total_duration_s']:.0f} s)")
    print(f"Native voltage dt: {summary['native_voltage_sample_rate_ms']:.3f} ms; analysis dt: {summary['analysis_dt_ms']:.3f} ms")
    print(f"Selected pairs: {summary['selected_pairs']}")
    print(f"Usable events: {summary['usable_events']}; skipped events: {summary['skipped_events']}")
    for kind, label in (('exc', 'EPSP'), ('inh', 'IPSP'), ('null', 'Null')):
        pair_summary = summary['pair_level'][kind]
        pop_summary = summary['population_trace'][kind]
        if pair_summary['n_pairs'] == 0 or pop_summary is None:
            print(f'{label}: no usable pairs')
            continue
        delay = pair_summary['delay_ms']
        amplitude = pair_summary['amplitude_mv']
        print(
            f"{label}: pairs={pair_summary['n_pairs']}, "
            f"pair delay median={delay['median']:.2f} ms (IQR {delay['p25']:.2f}-{delay['p75']:.2f}), "
            f"amplitude median={amplitude['median']:.4f} mV, "
            f"population extremum={pop_summary['amplitude_mv']:.4f} mV at {pop_summary['delay_ms']:.2f} ms, "
            f"fraction after {summary['reference_max_delay_ms']:.1f} ms={pair_summary['fraction_delays_after_reference']:.3f}"
        )
    print(f'Outputs written to: {output_dir}')


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    session_dir = resolve_session_dir(args.session)
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / 'voltage_augmented_learned_lif_outputs' / f'psp_trace_analysis_{session_dir.name}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_files = sorted(session_dir.glob('recording[0-9][0-9][0-9].npz'))
    if not recording_files:
        raise FileNotFoundError(f'No recordingNNN.npz files found in {session_dir}')
    connections = load_network(session_dir)
    spike_counts, recording_summaries = scan_recording_spike_counts(recording_files)
    exclusion_windows_by_recording, exclusion_summaries = build_exclusion_windows(recording_files, args)
    connected_pairs = select_connected_pairs(connections, spike_counts, args.max_pairs_per_type)
    null_pairs = select_null_pairs(connected_pairs, connections, len(spike_counts), rng)
    pairs = connected_pairs + null_pairs
    event_plans = plan_events(
        recording_files,
        pairs,
        args.pre_ms,
        args.post_ms,
        args.max_events_per_pair,
        rng,
        exclusion_windows_by_recording=exclusion_windows_by_recording,
    )
    accumulation = accumulate_windows(recording_files, pairs, event_plans, args, rng)
    pair_rows = summarize_pairs(
        pairs,
        accumulation['pair_stats'],
        accumulation['event_counts'],
        accumulation['time_axis_ms'],
        args.response_ms,
        args.reference_max_delay_ms,
    )
    summary = build_summary(
        args,
        session_dir,
        recording_summaries,
        exclusion_summaries,
        pairs,
        pair_rows,
        accumulation,
    )

    write_pair_csv(output_dir / 'pair_psp_stats.csv', pair_rows)
    with open(output_dir / 'psp_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    plot_population_traces(output_dir / 'connected_pair_psp_average.png', accumulation, args)
    plot_delay_histogram(output_dir / 'connected_pair_psp_delay_hist.png', pair_rows, args.reference_max_delay_ms)
    plot_example_pairs(output_dir / 'connected_pair_psp_examples.png', pairs, pair_rows, accumulation)
    plot_downsample_comparison(output_dir / 'downsample_mean_vs_stride.png', accumulation, args)
    print_console_summary(summary, output_dir)


if __name__ == '__main__':
    main()