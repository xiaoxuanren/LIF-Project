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


REPO_ROOT = Path(__file__).resolve().parents[1]


class PhaseAccumulator:
    def __init__(self):
        self.sample_sum = 0.0
        self.sample_sumsq = 0.0
        self.sample_count = 0
        self.threshold_counts = {-55.0: 0, -52.0: 0, -50.0: 0}
        self.event_means = []
        self.event_threshold_fractions = {threshold: [] for threshold in self.threshold_counts}

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        valid_values = values[np.isfinite(values)]
        if valid_values.size == 0:
            return
        self.sample_sum += float(np.sum(valid_values))
        self.sample_sumsq += float(np.sum(valid_values ** 2))
        self.sample_count += int(valid_values.size)
        self.event_means.append(float(np.mean(valid_values)))
        for threshold in self.threshold_counts:
            count = int(np.sum(valid_values >= threshold))
            self.threshold_counts[threshold] += count
            self.event_threshold_fractions[threshold].append(float(count / valid_values.size))

    def summary(self):
        if self.sample_count == 0:
            return {
                'n_samples': 0,
                'sample_mean_mv': None,
                'sample_std_mv': None,
                'event_mean_mv_mean': None,
                'event_mean_mv_median': None,
                'event_mean_mv_p25': None,
                'event_mean_mv_p75': None,
                'fraction_ge_minus55': None,
                'fraction_ge_minus52': None,
                'fraction_ge_minus50': None,
            }
        sample_mean = self.sample_sum / self.sample_count
        variance = max(self.sample_sumsq / self.sample_count - sample_mean ** 2, 0.0)
        event_means = np.asarray(self.event_means, dtype=np.float64)
        result = {
            'n_samples': int(self.sample_count),
            'sample_mean_mv': float(sample_mean),
            'sample_std_mv': float(np.sqrt(variance)),
            'event_mean_mv_mean': float(np.mean(event_means)),
            'event_mean_mv_median': float(np.median(event_means)),
            'event_mean_mv_p25': float(np.percentile(event_means, 25)),
            'event_mean_mv_p75': float(np.percentile(event_means, 75)),
        }
        for threshold in self.threshold_counts:
            suffix = str(int(abs(threshold)))
            result[f'fraction_ge_minus{suffix}'] = float(
                self.threshold_counts[threshold] / self.sample_count
            )
            fractions = np.asarray(self.event_threshold_fractions[threshold], dtype=np.float64)
            result[f'event_fraction_ge_minus{suffix}_median'] = float(np.median(fractions))
        return result


class TraceAccumulator:
    def __init__(self, length):
        self.sum_values = np.zeros(int(length), dtype=np.float64)
        self.sum_squares = np.zeros(int(length), dtype=np.float64)
        self.counts = np.zeros(int(length), dtype=np.int64)

    def add_matrix(self, values):
        values = np.asarray(values, dtype=np.float64)
        valid = np.isfinite(values)
        clean_values = np.where(valid, values, 0.0)
        self.sum_values += clean_values.sum(axis=0)
        self.sum_squares += (clean_values ** 2).sum(axis=0)
        self.counts += valid.sum(axis=0)

    def mean(self):
        result = np.full_like(self.sum_values, np.nan, dtype=np.float64)
        valid = self.counts > 0
        result[valid] = self.sum_values[valid] / self.counts[valid]
        return result

    def sem(self):
        result = np.full_like(self.sum_values, np.nan, dtype=np.float64)
        valid = self.counts > 1
        mean_values = self.mean()
        variance = np.zeros_like(self.sum_values, dtype=np.float64)
        variance[valid] = self.sum_squares[valid] / self.counts[valid] - mean_values[valid] ** 2
        variance = np.maximum(variance, 0.0)
        result[valid] = np.sqrt(variance[valid] / self.counts[valid])
        return result


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze raw and spike-masked voltage during burst windows.')
    parser.add_argument('--session', type=str, default='LIF data/20260531_041642')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--pre-ms', type=float, default=200.0)
    parser.add_argument('--post-ms', type=float, default=900.0)
    parser.add_argument('--baseline-ms', type=float, default=100.0)
    parser.add_argument('--saved-response-ms', type=float, default=650.0)
    parser.add_argument('--mask-spike-pre-ms', type=float, default=0.0)
    parser.add_argument('--mask-spike-post-ms', type=float, default=2.0)
    parser.add_argument('--peak-threshold-mv', type=float, default=15.0)
    parser.add_argument('--burst-activity-bin-ms', type=float, default=100.0)
    parser.add_argument('--burst-smooth-bins', type=int, default=3)
    parser.add_argument('--burst-threshold-std', type=float, default=3.0)
    parser.add_argument('--burst-min-active-frac', type=float, default=0.10)
    parser.add_argument('--burst-min-duration-ms', type=float, default=100.0)
    parser.add_argument('--burst-merge-gap-ms', type=float, default=150.0)
    parser.add_argument('--burst-pad-before-ms', type=float, default=100.0)
    parser.add_argument('--burst-pad-after-ms', type=float, default=250.0)
    parser.add_argument('--max-example-neurons', type=int, default=8)
    return parser.parse_args()


def resolve_session_dir(raw_session):
    session_dir = Path(raw_session)
    if not session_dir.is_absolute():
        session_dir = REPO_ROOT / session_dir
    if not session_dir.exists():
        raise FileNotFoundError(f'Session directory not found: {session_dir}')
    return session_dir


def find_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []
    padded = np.pad(mask.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def merge_windows_ms(windows, max_gap_ms=0.0):
    ordered = sorted((float(start), float(end)) for start, end in windows if end > start)
    if not ordered:
        return []
    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start <= current[1] + float(max_gap_ms):
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def detect_burst_windows_from_spike_times(spike_times, duration_ms, args):
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
        if bins.size > 0:
            active[neuron_id, np.unique(bins)] = True
    active_fraction = active.mean(axis=0).astype(np.float32)
    smooth_bins = max(int(args.burst_smooth_bins), 1)
    kernel = np.ones(smooth_bins, dtype=np.float32) / float(smooth_bins)
    smoothed = np.convolve(active_fraction, kernel, mode='same')
    threshold = max(
        float(np.mean(smoothed) + float(args.burst_threshold_std) * np.std(smoothed)),
        float(args.burst_min_active_frac),
    )
    raw_windows = []
    for start_bin, end_bin in find_true_segments(smoothed >= threshold):
        start_ms = start_bin * activity_bin_ms
        end_ms = min(float(duration_ms), end_bin * activity_bin_ms)
        if end_ms - start_ms >= float(args.burst_min_duration_ms):
            raw_windows.append((start_ms, end_ms))
    merged = merge_windows_ms(raw_windows, max_gap_ms=float(args.burst_merge_gap_ms))
    padded = [
        (
            max(0.0, start - float(args.burst_pad_before_ms)),
            min(float(duration_ms), end + float(args.burst_pad_after_ms)),
        )
        for start, end in merged
    ]
    return merge_windows_ms(padded, max_gap_ms=0.0), threshold


def load_burst_events(recording_files, args):
    events = []
    recording_summaries = []
    for recording_index, recording_file in enumerate(recording_files):
        with np.load(recording_file, allow_pickle=True) as recording_data:
            duration_ms = float(recording_data['duration'])
            spike_times = recording_data['spike_times']
            total_spikes = int(sum(len(neuron_spikes) for neuron_spikes in spike_times))
            saved_onsets = np.asarray(
                recording_data['burst_onset_times'] if 'burst_onset_times' in recording_data.files else [],
                dtype=np.float64,
            )
            saved_onsets = saved_onsets[(saved_onsets >= 0.0) & (saved_onsets < duration_ms)]
            for saved_index, onset_ms in enumerate(saved_onsets):
                events.append({
                    'source': 'saved_stimulation',
                    'recording_index': recording_index,
                    'recording_file': str(recording_file),
                    'event_index': saved_index,
                    'anchor_ms': float(onset_ms),
                    'start_ms': float(onset_ms),
                    'end_ms': float(min(duration_ms, onset_ms + float(args.saved_response_ms))),
                })
            detected_windows, threshold = detect_burst_windows_from_spike_times(spike_times, duration_ms, args)
            for detected_index, (start_ms, end_ms) in enumerate(detected_windows):
                events.append({
                    'source': 'detected_network_burst',
                    'recording_index': recording_index,
                    'recording_file': str(recording_file),
                    'event_index': detected_index,
                    'anchor_ms': float(start_ms),
                    'start_ms': float(start_ms),
                    'end_ms': float(end_ms),
                })
            recording_summaries.append({
                'path': str(recording_file),
                'duration_ms': duration_ms,
                'spike_count': total_spikes,
                'saved_stimulation_onsets': int(saved_onsets.size),
                'detected_network_burst_windows': int(len(detected_windows)),
                'detected_burst_threshold': float(threshold),
                'voltage_sample_rate_ms': float(recording_data['voltage_sample_rate']),
                'voltage_n_samples': int(recording_data['voltage_n_samples']),
            })
    return events, recording_summaries


def mask_spike_neighborhoods(valid_mask, spike_times, start_bin, sample_rate_ms, args):
    pre_bins = int(round(float(args.mask_spike_pre_ms) / float(sample_rate_ms)))
    post_bins = int(round(float(args.mask_spike_post_ms) / float(sample_rate_ms)))
    n_neurons, n_samples = valid_mask.shape
    for neuron_id in range(n_neurons):
        neuron_spikes = np.asarray(spike_times[neuron_id], dtype=np.float64)
        if neuron_spikes.size == 0:
            continue
        spike_bins = np.floor(neuron_spikes / float(sample_rate_ms) + 1e-9).astype(np.int64)
        in_window = (spike_bins >= start_bin - pre_bins) & (spike_bins < start_bin + n_samples + post_bins)
        for spike_bin in spike_bins[in_window]:
            local_bin = int(spike_bin) - int(start_bin)
            start = max(0, local_bin - pre_bins)
            end = min(n_samples, local_bin + post_bins + 1)
            valid_mask[neuron_id, start:end] = False


def phase_intervals(event, args):
    start_ms = float(event['start_ms'])
    end_ms = float(event['end_ms'])
    return {
        'baseline': (max(0.0, start_ms - float(args.baseline_ms)), start_ms),
        'during': (start_ms, end_ms),
        'after': (end_ms, end_ms + float(args.baseline_ms)),
    }


def count_spikes_in_interval(spike_times, start_ms, end_ms):
    total = 0
    active = 0
    for neuron_spikes in spike_times:
        neuron_spikes = np.asarray(neuron_spikes, dtype=np.float64)
        count = int(np.sum((neuron_spikes >= start_ms) & (neuron_spikes < end_ms)))
        total += count
        active += int(count > 0)
    return total, active / max(len(spike_times), 1)


def add_phase_voltage(phase_accumulators, source, phase, values, valid_mask):
    raw_values = np.asarray(values, dtype=np.float64)
    masked_values = np.where(valid_mask, raw_values, np.nan)
    phase_accumulators[source]['raw'][phase].add(raw_values)
    phase_accumulators[source]['spike_masked'][phase].add(masked_values)


def read_interval_voltage(voltage_traces, start_ms, end_ms, sample_rate_ms, n_samples):
    start_bin = max(0, int(np.floor(float(start_ms) / float(sample_rate_ms) + 1e-9)))
    end_bin = min(n_samples, int(np.ceil(float(end_ms) / float(sample_rate_ms) - 1e-9)))
    if end_bin <= start_bin:
        return None, start_bin, end_bin
    return np.asarray(voltage_traces[:, start_bin:end_bin], dtype=np.float32), start_bin, end_bin


def initialize_accumulators(events, window_len):
    sources = sorted({event['source'] for event in events})
    phase_accumulators = {
        source: {
            mode: {phase: PhaseAccumulator() for phase in ('baseline', 'during', 'after')}
            for mode in ('raw', 'spike_masked')
        }
        for source in sources
    }
    trace_accumulators = {
        source: {
            'raw': TraceAccumulator(window_len),
            'spike_masked': TraceAccumulator(window_len),
        }
        for source in sources
    }
    return phase_accumulators, trace_accumulators


def analyze_burst_voltage(recording_files, events, args):
    first_recording = np.load(recording_files[0], allow_pickle=True)
    sample_rate_ms = float(first_recording['voltage_sample_rate'])
    first_recording.close()
    pre_bins = int(round(float(args.pre_ms) / sample_rate_ms))
    post_bins = int(round(float(args.post_ms) / sample_rate_ms))
    window_len = pre_bins + post_bins + 1
    time_axis_ms = (np.arange(window_len, dtype=np.float64) - pre_bins) * sample_rate_ms
    phase_accumulators, trace_accumulators = initialize_accumulators(events, window_len)
    event_rows = []
    events_by_recording = {}
    for event in events:
        events_by_recording.setdefault(int(event['recording_index']), []).append(event)

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
            n_samples = int(recording_data['voltage_n_samples'])
            for event in recording_events:
                source = event['source']
                spike_count, active_fraction = count_spikes_in_interval(
                    spike_times, float(event['start_ms']), float(event['end_ms'])
                )
                event_rows.append({
                    'source': source,
                    'recording_index': int(event['recording_index']),
                    'event_index': int(event['event_index']),
                    'anchor_ms': float(event['anchor_ms']),
                    'start_ms': float(event['start_ms']),
                    'end_ms': float(event['end_ms']),
                    'duration_ms': float(event['end_ms']) - float(event['start_ms']),
                    'spike_count': int(spike_count),
                    'active_fraction': float(active_fraction),
                })
                for phase, (phase_start_ms, phase_end_ms) in phase_intervals(event, args).items():
                    values, start_bin, _ = read_interval_voltage(
                        voltage_traces, phase_start_ms, phase_end_ms, sample_rate_ms, n_samples
                    )
                    if values is None:
                        continue
                    valid_mask = np.isfinite(values)
                    if args.peak_threshold_mv is not None:
                        valid_mask &= values < float(args.peak_threshold_mv)
                    mask_spike_neighborhoods(valid_mask, spike_times, start_bin, sample_rate_ms, args)
                    add_phase_voltage(phase_accumulators, source, phase, values, valid_mask)

                anchor_bin = int(round(float(event['anchor_ms']) / sample_rate_ms))
                aligned_start = anchor_bin - pre_bins
                aligned_end = anchor_bin + post_bins + 1
                if aligned_start < 0 or aligned_end > n_samples:
                    continue
                aligned_values = np.asarray(voltage_traces[:, aligned_start:aligned_end], dtype=np.float32)
                baseline_mask = (time_axis_ms >= -float(args.baseline_ms)) & (time_axis_ms < 0.0)
                if not np.any(baseline_mask):
                    continue
                raw_baseline = np.nanmean(aligned_values[:, baseline_mask], axis=1)
                raw_delta = aligned_values.astype(np.float64) - raw_baseline[:, None]
                trace_accumulators[source]['raw'].add_matrix(raw_delta)
                valid_mask = np.isfinite(aligned_values)
                if args.peak_threshold_mv is not None:
                    valid_mask &= aligned_values < float(args.peak_threshold_mv)
                mask_spike_neighborhoods(valid_mask, spike_times, aligned_start, sample_rate_ms, args)
                masked_values = np.where(valid_mask, aligned_values, np.nan)
                masked_baseline = np.nanmean(masked_values[:, baseline_mask], axis=1)
                masked_delta = masked_values.astype(np.float64) - masked_baseline[:, None]
                trace_accumulators[source]['spike_masked'].add_matrix(masked_delta)
            close_method = getattr(voltage_traces, 'close', None)
            if close_method is not None:
                close_method()
    return {
        'sample_rate_ms': sample_rate_ms,
        'time_axis_ms': time_axis_ms,
        'phase_accumulators': phase_accumulators,
        'trace_accumulators': trace_accumulators,
        'event_rows': event_rows,
    }


def summarize_events(event_rows):
    summary = {}
    for source in sorted({row['source'] for row in event_rows}):
        rows = [row for row in event_rows if row['source'] == source]
        durations = np.asarray([row['duration_ms'] for row in rows], dtype=np.float64)
        spike_counts = np.asarray([row['spike_count'] for row in rows], dtype=np.float64)
        active_fractions = np.asarray([row['active_fraction'] for row in rows], dtype=np.float64)
        summary[source] = {
            'n_events': int(len(rows)),
            'duration_ms_median': float(np.median(durations)),
            'duration_ms_p10': float(np.percentile(durations, 10)),
            'duration_ms_p90': float(np.percentile(durations, 90)),
            'spike_count_median': float(np.median(spike_counts)),
            'spike_count_p10': float(np.percentile(spike_counts, 10)),
            'spike_count_p90': float(np.percentile(spike_counts, 90)),
            'active_fraction_median': float(np.median(active_fractions)),
            'active_fraction_p90': float(np.percentile(active_fractions, 90)),
        }
    return summary


def summarize_phase_accumulators(phase_accumulators):
    summary = {}
    for source, source_data in phase_accumulators.items():
        summary[source] = {}
        for mode, mode_data in source_data.items():
            summary[source][mode] = {phase: accumulator.summary() for phase, accumulator in mode_data.items()}
            baseline_mean = summary[source][mode]['baseline']['sample_mean_mv']
            if baseline_mean is None:
                continue
            for phase in ('during', 'after'):
                phase_mean = summary[source][mode][phase]['sample_mean_mv']
                summary[source][mode][phase]['delta_from_baseline_mv'] = (
                    None if phase_mean is None else float(phase_mean - baseline_mean)
                )
    return summary


def write_phase_csv(path, phase_summary):
    fieldnames = [
        'source', 'mode', 'phase', 'n_samples', 'sample_mean_mv', 'sample_std_mv',
        'event_mean_mv_mean', 'event_mean_mv_median', 'event_mean_mv_p25', 'event_mean_mv_p75',
        'delta_from_baseline_mv', 'fraction_ge_minus55', 'fraction_ge_minus52', 'fraction_ge_minus50',
        'event_fraction_ge_minus55_median', 'event_fraction_ge_minus52_median',
        'event_fraction_ge_minus50_median',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, source_data in phase_summary.items():
            for mode, mode_data in source_data.items():
                for phase, values in mode_data.items():
                    row = {'source': source, 'mode': mode, 'phase': phase}
                    row.update(values)
                    writer.writerow(row)


def write_event_csv(path, event_rows):
    fieldnames = [
        'source', 'recording_index', 'event_index', 'anchor_ms', 'start_ms', 'end_ms',
        'duration_ms', 'spike_count', 'active_fraction',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in event_rows:
            writer.writerow(row)


def plot_aligned_voltage(path, analysis, mode):
    time_axis_ms = analysis['time_axis_ms']
    trace_accumulators = analysis['trace_accumulators']
    colors = {
        'saved_stimulation': 'tab:orange',
        'detected_network_burst': 'tab:red',
    }
    labels = {
        'saved_stimulation': 'Saved stimulation onset',
        'detected_network_burst': 'Detected network-burst onset',
    }
    plt.figure(figsize=(10, 5.5))
    for source in sorted(trace_accumulators):
        accumulator = trace_accumulators[source][mode]
        mean_values = accumulator.mean()
        sem_values = accumulator.sem()
        color = colors.get(source, 'black')
        plt.plot(time_axis_ms, mean_values, color=color, linewidth=1.8, label=labels.get(source, source))
        valid = np.isfinite(sem_values)
        plt.fill_between(
            time_axis_ms[valid],
            mean_values[valid] - sem_values[valid],
            mean_values[valid] + sem_values[valid],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    plt.axvline(0.0, color='black', linewidth=1.0, alpha=0.7)
    plt.axhline(0.0, color='black', linewidth=0.8, alpha=0.4)
    plt.xlabel('Time from burst/stimulation onset (ms)')
    plt.ylabel('Delta voltage from pre-burst baseline (mV)')
    title_mode = 'Spike-masked subthreshold' if mode == 'spike_masked' else 'Raw all-neuron'
    plt.title(f'{title_mode} Voltage Aligned To Burst Onset')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_phase_bars(path, phase_summary, mode):
    sources = sorted(phase_summary)
    phases = ['baseline', 'during', 'after']
    x = np.arange(len(phases), dtype=np.float64)
    width = 0.35 if len(sources) <= 2 else 0.25
    colors = {'saved_stimulation': 'tab:orange', 'detected_network_burst': 'tab:red'}
    labels = {'saved_stimulation': 'Saved stimulation', 'detected_network_burst': 'Detected burst'}
    plt.figure(figsize=(9, 5.5))
    for source_index, source in enumerate(sources):
        offset = (source_index - (len(sources) - 1) / 2.0) * width
        centers = [phase_summary[source][mode][phase]['event_mean_mv_median'] for phase in phases]
        lower = [
            max(centers[idx] - phase_summary[source][mode][phase]['event_mean_mv_p25'], 0.0)
            if centers[idx] is not None else 0.0
            for idx, phase in enumerate(phases)
        ]
        upper = [
            max(phase_summary[source][mode][phase]['event_mean_mv_p75'] - centers[idx], 0.0)
            if centers[idx] is not None else 0.0
            for idx, phase in enumerate(phases)
        ]
        plt.bar(
            x + offset,
            centers,
            width=width,
            yerr=[lower, upper],
            color=colors.get(source, '0.4'),
            alpha=0.75,
            capsize=3,
            label=labels.get(source, source),
        )
    plt.xticks(x, phases)
    plt.ylabel('Voltage (mV)')
    title_mode = 'spike-masked subthreshold' if mode == 'spike_masked' else 'raw all-neuron'
    plt.title(f'Voltage Phase Summary ({title_mode})')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_example_burst(path, recording_files, event_rows, args):
    detected_rows = [row for row in event_rows if row['source'] == 'detected_network_burst']
    if detected_rows:
        event = max(detected_rows, key=lambda row: row['active_fraction'])
    elif event_rows:
        event = max(event_rows, key=lambda row: row['active_fraction'])
    else:
        return
    recording_file = recording_files[int(event['recording_index'])]
    with np.load(recording_file, allow_pickle=True) as recording_data:
        voltage_bundle = resolve_recording_voltage(
            recording_data,
            recording_path=str(recording_file),
            load_into_memory=False,
        )
        voltage_traces = voltage_bundle['traces']
        sample_rate_ms = float(recording_data['voltage_sample_rate'])
        n_samples = int(recording_data['voltage_n_samples'])
        start_ms = max(0.0, float(event['anchor_ms']) - float(args.pre_ms))
        end_ms = min(float(recording_data['duration']), float(event['anchor_ms']) + float(args.post_ms))
        values, start_bin, end_bin = read_interval_voltage(voltage_traces, start_ms, end_ms, sample_rate_ms, n_samples)
        if values is None:
            return
        local_times = np.arange(start_bin, end_bin, dtype=np.float64) * sample_rate_ms - float(event['anchor_ms'])
        spike_counts = np.asarray([len(spikes) for spikes in recording_data['spike_times']], dtype=np.int64)
        selected_neurons = np.argsort(spike_counts)[-max(int(args.max_example_neurons), 1):][::-1]
        population_mean = values.mean(axis=0)
        plt.figure(figsize=(11, 7))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(local_times, population_mean, color='black', linewidth=1.2)
        ax1.axvline(0.0, color='tab:red', linewidth=1.0)
        ax1.axhline(-50.0, color='firebrick', linestyle='--', linewidth=0.8, alpha=0.6)
        ax1.set_ylabel('Population mean V (mV)')
        ax1.set_title(
            f"Example {event['source']} window, rec {int(event['recording_index'])}, "
            f"active fraction {event['active_fraction']:.3f}"
        )
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        offsets = np.arange(len(selected_neurons), dtype=np.float64) * 12.0
        for offset, neuron_id in zip(offsets, selected_neurons):
            trace = values[int(neuron_id)].astype(np.float64)
            ax2.plot(local_times, trace + offset, linewidth=0.8, label=f'N{int(neuron_id)}')
        ax2.axvline(0.0, color='tab:red', linewidth=1.0)
        ax2.set_xlabel('Time from burst onset (ms)')
        ax2.set_ylabel('Example neuron traces (offset mV)')
        ax2.legend(loc='upper right', ncol=2, fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        close_method = getattr(voltage_traces, 'close', None)
        if close_method is not None:
            close_method()


def print_summary(session_dir, recording_summaries, event_summary, phase_summary, output_dir):
    total_duration_s = sum(item['duration_ms'] for item in recording_summaries) / 1000.0
    total_spikes = sum(item['spike_count'] for item in recording_summaries)
    print(f'Session: {session_dir.name} ({len(recording_summaries)} recordings, {total_duration_s:.0f} s, {total_spikes} spikes)')
    for source, values in event_summary.items():
        print(
            f"{source}: n={values['n_events']}, duration median={values['duration_ms_median']:.1f} ms, "
            f"spikes median={values['spike_count_median']:.1f}, active fraction median={values['active_fraction_median']:.3f}"
        )
        for mode in ('raw', 'spike_masked'):
            baseline = phase_summary[source][mode]['baseline']['sample_mean_mv']
            during = phase_summary[source][mode]['during']['sample_mean_mv']
            delta = phase_summary[source][mode]['during'].get('delta_from_baseline_mv')
            frac_50 = phase_summary[source][mode]['during']['fraction_ge_minus50']
            print(
                f"  {mode}: baseline={baseline:.3f} mV, during={during:.3f} mV, "
                f"delta={delta:.3f} mV, during fraction >= -50 mV={frac_50:.4f}"
            )
    print(f'Outputs written to: {output_dir}')


def main():
    args = parse_args()
    session_dir = resolve_session_dir(args.session)
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / 'voltage_augmented_learned_lif_outputs' / f'burst_voltage_analysis_{session_dir.name}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_files = sorted(session_dir.glob('recording[0-9][0-9][0-9].npz'))
    if not recording_files:
        raise FileNotFoundError(f'No recordingNNN.npz files found in {session_dir}')
    events, recording_summaries = load_burst_events(recording_files, args)
    if not events:
        raise RuntimeError('No saved stimulation onsets or detected network-burst windows found')
    analysis = analyze_burst_voltage(recording_files, events, args)
    event_summary = summarize_events(analysis['event_rows'])
    phase_summary = summarize_phase_accumulators(analysis['phase_accumulators'])
    summary = {
        'session': session_dir.name,
        'session_dir': str(session_dir),
        'recording_summary': {
            'n_recordings': int(len(recording_summaries)),
            'total_duration_s': float(sum(item['duration_ms'] for item in recording_summaries) / 1000.0),
            'total_spikes': int(sum(item['spike_count'] for item in recording_summaries)),
            'native_voltage_sample_rate_ms': float(analysis['sample_rate_ms']),
        },
        'burst_detection': {
            'activity_bin_ms': float(args.burst_activity_bin_ms),
            'smooth_bins': int(args.burst_smooth_bins),
            'threshold_std': float(args.burst_threshold_std),
            'min_active_fraction': float(args.burst_min_active_frac),
            'min_burst_duration_ms': float(args.burst_min_duration_ms),
            'merge_gap_ms': float(args.burst_merge_gap_ms),
            'pad_before_ms': float(args.burst_pad_before_ms),
            'pad_after_ms': float(args.burst_pad_after_ms),
        },
        'voltage_cleaning': {
            'mask_spike_pre_ms': float(args.mask_spike_pre_ms),
            'mask_spike_post_ms': float(args.mask_spike_post_ms),
            'peak_threshold_mv': None if args.peak_threshold_mv is None else float(args.peak_threshold_mv),
        },
        'event_summary': event_summary,
        'phase_summary': phase_summary,
    }
    with open(output_dir / 'burst_voltage_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    write_event_csv(output_dir / 'burst_voltage_events.csv', analysis['event_rows'])
    write_phase_csv(output_dir / 'burst_voltage_phase_stats.csv', phase_summary)
    plot_aligned_voltage(output_dir / 'burst_aligned_voltage_raw.png', analysis, mode='raw')
    plot_aligned_voltage(output_dir / 'burst_aligned_voltage_spike_masked.png', analysis, mode='spike_masked')
    plot_phase_bars(output_dir / 'burst_voltage_phase_raw.png', phase_summary, mode='raw')
    plot_phase_bars(output_dir / 'burst_voltage_phase_spike_masked.png', phase_summary, mode='spike_masked')
    plot_example_burst(output_dir / 'example_burst_voltage_traces.png', recording_files, analysis['event_rows'], args)
    print_summary(session_dir, recording_summaries, event_summary, phase_summary, output_dir)


if __name__ == '__main__':
    main()