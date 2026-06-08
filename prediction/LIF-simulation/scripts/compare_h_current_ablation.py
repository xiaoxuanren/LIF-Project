"""
Compare saved conductance-LIF sessions with h-current enabled vs disabled.

Reads one or more session folders or session_metadata.json files and reports:
    - per-session spike-rate and activity statistics
    - detected network-burst windows from population synchrony
    - optional raster plots with detected bursts shaded
    - whether h-current was enabled for each session
    - pairwise deltas for a matched on/off comparison
    - optional topology-match check for true matched pairs

Usage:
        python compare_h_current_ablation.py LIF data/20260518_123456 LIF data/20260518_124321
        python compare_h_current_ablation.py LIF data/20260518_123456/session_metadata.json --json-out report.json
        python compare_h_current_ablation.py LIF data/20260518_123456 --raster-out-dir ablation_rasters
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "LIF data"
BIN_WIDTH_MS = 100.0


def resolve_metadata_path(path_arg):
    """Resolve a session directory or metadata path into ``session_metadata.json``.

    Args:
        path_arg: Session directory path or direct metadata-file path.

    Returns:
        A ``Path`` pointing to the session metadata JSON file.
    """
    path = Path(path_arg)
    if path.is_dir():
        metadata_path = path / "session_metadata.json"
        if metadata_path.exists():
            return metadata_path
    elif path.is_file():
        return path
    raise FileNotFoundError(f"Could not resolve session metadata from: {path_arg}")


def resolve_saved_path(session_dir, stored_path):
    """Resolve a saved artifact path against session-local and repo-level roots.

    Args:
        session_dir: Directory containing the saved session outputs.
        stored_path: Stored path from metadata, which may be absolute or relative.

    Returns:
        A resolved ``Path`` when the artifact can be located, ``None`` when the
        metadata entry is empty, or the most likely session-local path when the
        file is not yet confirmed to exist.
    """
    if not stored_path:
        return None

    candidate = Path(stored_path)
    if candidate.is_file():
        return candidate

    basename = Path(stored_path).name
    session_candidate = session_dir / basename
    if session_candidate.exists():
        return session_candidate

    repo_candidate = REPO_ROOT / stored_path
    if repo_candidate.exists():
        return repo_candidate

    return session_candidate


def find_true_segments(mask):
    """Return index segments where a boolean mask remains true.

    Args:
        mask: One-dimensional boolean-like array.

    Returns:
        A list of ``(start_index, end_index)`` pairs using half-open indexing.
    """
    if len(mask) == 0:
        return []
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def merge_windows(windows, max_gap):
    """Merge overlapping windows and windows separated by at most ``max_gap``.

    Args:
        windows: Iterable of ``(start, end)`` windows.
        max_gap: Maximum gap allowed between adjacent windows before merging.

    Returns:
        A sorted list of merged windows.
    """
    if not windows:
        return []

    merged = []
    for start, end in sorted((float(start), float(end)) for start, end in windows):
        if not merged:
            merged.append([start, end])
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + float(max_gap):
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def compute_population_activity_trace(spike_times, duration_ms, bin_width_ms=BIN_WIDTH_MS):
    """Compute the fraction of neurons active in each coarse time bin.

    Args:
        spike_times: Per-neuron spike-time sequences in milliseconds.
        duration_ms: Recording duration in milliseconds.
        bin_width_ms: Width of each activity bin in milliseconds.

    Returns:
        A one-dimensional array whose entries give the active-neuron fraction in
        each bin.
    """
    n_neurons = len(spike_times)
    n_bins = int(np.ceil(duration_ms / bin_width_ms))
    active_counts = np.zeros(n_bins, dtype=np.int32)

    for neuron_spikes in spike_times:
        spikes = np.asarray(neuron_spikes, dtype=np.float64)
        if spikes.size == 0:
            continue
        bin_indices = np.floor(spikes / bin_width_ms).astype(np.int64)
        bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < n_bins)]
        if bin_indices.size > 0:
            active_counts[np.unique(bin_indices)] += 1

    return active_counts / max(n_neurons, 1)


def compute_population_activity_stats(spike_times, duration_ms, bin_width_ms=BIN_WIDTH_MS):
    """Summarize coarse population synchrony statistics from binned activity.

    Args:
        spike_times: Per-neuron spike-time sequences in milliseconds.
        duration_ms: Recording duration in milliseconds.
        bin_width_ms: Width of each activity bin in milliseconds.

    Returns:
        A dictionary of coarse synchrony statistics derived from the population
        activity trace.
    """
    active_fraction = compute_population_activity_trace(spike_times, duration_ms, bin_width_ms=bin_width_ms)
    n_bins = len(active_fraction)
    return {
        "n_bins": int(n_bins),
        "max_active_fraction_100ms": float(active_fraction.max()) if n_bins > 0 else 0.0,
        "bins_ge_10pct": int(np.sum(active_fraction >= 0.10)),
        "bins_ge_25pct": int(np.sum(active_fraction >= 0.25)),
        "bins_ge_50pct": int(np.sum(active_fraction >= 0.50)),
    }


def detect_network_bursts(spike_times, duration_ms,
                          activity_bin_ms=100.0,
                          smooth_bins=3,
                          threshold_std=3.0,
                          min_active_fraction=0.10,
                          min_burst_duration_ms=100.0,
                          merge_gap_ms=150.0,
                          pad_before_ms=100.0,
                          pad_after_ms=250.0):
    """Detect network bursts from coarse population synchrony.

    Args:
        spike_times: Per-neuron spike-time sequences in milliseconds.
        duration_ms: Recording duration in milliseconds.
        activity_bin_ms: Width of the coarse activity bins in milliseconds.
        smooth_bins: Width of the moving-average smoothing kernel in bins.
        threshold_std: Threshold scale factor used in ``mean + std_factor * std``.
        min_active_fraction: Minimum active-neuron fraction required for a burst.
        min_burst_duration_ms: Minimum coarse burst duration retained in the final
            output.
        merge_gap_ms: Maximum gap used when merging nearby burst windows.
        pad_before_ms: Padding added before each detected burst window.
        pad_after_ms: Padding added after each detected burst window.

    Returns:
        A dictionary containing the detected burst windows and the intermediate
        thresholding parameters used to find them.
    """
    active_fraction = compute_population_activity_trace(
        spike_times,
        duration_ms,
        bin_width_ms=activity_bin_ms,
    )
    if active_fraction.size == 0:
        return {
            "detected_network_burst_count": 0,
            "detected_network_burst_windows_ms": [],
            "burst_threshold": float(min_active_fraction),
            "peak_active_fraction": 0.0,
            "peak_smoothed_active_fraction": 0.0,
            "activity_bin_ms": float(activity_bin_ms),
            "smooth_bins": int(max(1, smooth_bins)),
            "threshold_std": float(threshold_std),
            "min_active_fraction": float(min_active_fraction),
            "min_burst_duration_ms": float(min_burst_duration_ms),
            "merge_gap_ms": float(merge_gap_ms),
            "pad_before_ms": float(pad_before_ms),
            "pad_after_ms": float(pad_after_ms),
        }

    smooth_bins = max(1, int(smooth_bins))
    kernel = np.ones(smooth_bins, dtype=np.float32) / float(smooth_bins)
    smoothed = np.convolve(active_fraction, kernel, mode="same")
    threshold = max(
        float(np.mean(smoothed) + threshold_std * np.std(smoothed)),
        float(min_active_fraction),
    )

    coarse_segments = find_true_segments(smoothed >= threshold)
    burst_windows = []
    for start_idx, end_idx in coarse_segments:
        start_ms = float(start_idx) * float(activity_bin_ms)
        end_ms = min(float(duration_ms), float(end_idx) * float(activity_bin_ms))
        if end_ms - start_ms >= float(min_burst_duration_ms):
            burst_windows.append((start_ms, end_ms))

    burst_windows = merge_windows(burst_windows, merge_gap_ms)
    burst_windows = [
        (
            max(0.0, start_ms - float(pad_before_ms)),
            min(float(duration_ms), end_ms + float(pad_after_ms)),
        )
        for start_ms, end_ms in burst_windows
    ]
    burst_windows = merge_windows(burst_windows, 0.0)

    return {
        "detected_network_burst_count": int(len(burst_windows)),
        "detected_network_burst_windows_ms": [
            [float(start_ms), float(end_ms)] for start_ms, end_ms in burst_windows
        ],
        "burst_threshold": float(threshold),
        "peak_active_fraction": float(active_fraction.max()),
        "peak_smoothed_active_fraction": float(smoothed.max()),
        "activity_bin_ms": float(activity_bin_ms),
        "smooth_bins": int(smooth_bins),
        "threshold_std": float(threshold_std),
        "min_active_fraction": float(min_active_fraction),
        "min_burst_duration_ms": float(min_burst_duration_ms),
        "merge_gap_ms": float(merge_gap_ms),
        "pad_before_ms": float(pad_before_ms),
        "pad_after_ms": float(pad_after_ms),
    }


def plot_recording_raster(spike_times, duration_ms, cluster_assignments=None,
                          detected_burst_windows_ms=None,
                          title="Recording Raster",
                          output_path=None):
    """Create a raster plot with detected burst windows shaded.

    Args:
        spike_times: Per-neuron spike-time sequences in milliseconds.
        duration_ms: Recording duration in milliseconds.
        cluster_assignments: Optional cluster index for each neuron.
        detected_burst_windows_ms: Optional burst windows to shade.
        title: Figure title.
        output_path: Optional path where the figure should be saved.

    Returns:
        The saved output path when ``output_path`` is provided, otherwise the
        ``(fig, ax)`` tuple for the created raster plot.
    """
    if cluster_assignments is None:
        neuron_ids = list(range(len(spike_times)))
        sorted_assignments = None
    else:
        cluster_assignments = np.asarray(cluster_assignments)
        neuron_ids = sorted(range(len(spike_times)), key=lambda idx: int(cluster_assignments[idx]))
        sorted_assignments = cluster_assignments[neuron_ids]

    fig, ax = plt.subplots(figsize=(18, 8))

    if detected_burst_windows_ms:
        for window_idx, (start_ms, end_ms) in enumerate(detected_burst_windows_ms):
            ax.axvspan(
                float(start_ms) / 1000.0,
                float(end_ms) / 1000.0,
                color="tab:red",
                alpha=0.18,
                linewidth=0,
                label="Detected burst" if window_idx == 0 else None,
            )

    for plot_idx, neuron_id in enumerate(neuron_ids):
        spikes = np.asarray(spike_times[neuron_id], dtype=np.float64)
        if spikes.size == 0:
            continue
        ax.scatter(
            spikes / 1000.0,
            np.full(spikes.size, plot_idx, dtype=np.float64),
            s=2,
            c="black",
            marker="|",
            linewidths=0.6,
        )

    if sorted_assignments is not None and len(sorted_assignments) > 0:
        unique_clusters = np.unique(sorted_assignments)
        for cluster_id in unique_clusters[:-1]:
            boundary_idx = np.max(np.where(sorted_assignments == cluster_id)[0]) + 0.5
            ax.axhline(float(boundary_idx), color="gray", linewidth=0.3, alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron (sorted by cluster)")
    ax.set_title(title)
    ax.set_xlim(0.0, float(duration_ms) / 1000.0)
    ax.set_ylim(-1.0, len(neuron_ids))
    if detected_burst_windows_ms:
        ax.legend(loc="upper right")

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)
    return fig, ax


def load_recording_metrics(recording_path, burst_detection_config=None,
                           raster_output_path=None, raster_title=None):
    """Load one saved recording and compute burst, activity, and rate metrics.

    Args:
        recording_path: Path to a saved ``recordingXXX.npz`` bundle.
        burst_detection_config: Optional keyword arguments passed to
            ``detect_network_bursts``.
        raster_output_path: Optional output path for a saved raster figure.
        raster_title: Optional raster title override.

    Returns:
        A dictionary containing per-recording rate, synchrony, burst, and raster
        metadata.
    """
    with np.load(recording_path, allow_pickle=True) as recording_data:
        spike_times = recording_data["spike_times"]
        duration_ms = float(recording_data["duration"]) if "duration" in recording_data else 0.0
        n_neurons = len(spike_times)
        spike_counts = np.asarray([len(neuron_spikes) for neuron_spikes in spike_times], dtype=np.int64)
        duration_s = duration_ms / 1000.0 if duration_ms > 0 else 0.0
        firing_rates = spike_counts / duration_s if duration_s > 0 else np.zeros(n_neurons, dtype=np.float64)
        population_stats = compute_population_activity_stats(spike_times, duration_ms)

        cluster_assignments = None
        if "resampled_cluster_assignments" in recording_data:
            cluster_assignments = np.asarray(recording_data["resampled_cluster_assignments"], dtype=np.int64)

        saved_burst_count = 0
        if "burst_onset_times" in recording_data:
            saved_burst_count = int(len(recording_data["burst_onset_times"]))

        burst_detection = detect_network_bursts(
            spike_times,
            duration_ms,
            **(burst_detection_config or {}),
        )

        raster_path = None
        if raster_output_path is not None:
            raster_path = plot_recording_raster(
                spike_times,
                duration_ms,
                cluster_assignments=cluster_assignments,
                detected_burst_windows_ms=burst_detection["detected_network_burst_windows_ms"],
                title=raster_title or f"Raster: {Path(recording_path).stem}",
                output_path=raster_output_path,
            )

    return {
        "path": str(recording_path),
        "duration_ms": float(duration_ms),
        "n_neurons": int(n_neurons),
        "total_spikes": int(spike_counts.sum()),
        "spike_counts": spike_counts,
        "mean_rate_hz": float(firing_rates.mean()) if firing_rates.size > 0 else 0.0,
        "median_rate_hz": float(np.median(firing_rates)) if firing_rates.size > 0 else 0.0,
        "max_rate_hz": float(firing_rates.max()) if firing_rates.size > 0 else 0.0,
        "active_neuron_fraction": float(np.mean(spike_counts > 0)) if spike_counts.size > 0 else 0.0,
        "saved_burst_count": int(saved_burst_count),
        "cluster_assignments_available": bool(cluster_assignments is not None),
        "raster_path": raster_path,
        **burst_detection,
        **population_stats,
    }


def aggregate_recordings(metadata, recording_metrics):
    """Aggregate per-recording metrics into one session-level summary.

    Args:
        metadata: Parsed session metadata dictionary.
        recording_metrics: Per-recording metric dictionaries from
            ``load_recording_metrics``.

    Returns:
        A session-level summary dictionary spanning all successful recordings.
    """
    if not recording_metrics:
        raise ValueError("No successful recordings found for session")

    total_duration_ms = float(sum(item["duration_ms"] for item in recording_metrics))
    n_neurons = int(recording_metrics[0]["n_neurons"])
    total_spikes = int(sum(item["total_spikes"] for item in recording_metrics))
    per_neuron_counts = np.sum([item["spike_counts"] for item in recording_metrics], axis=0)
    total_duration_s = total_duration_ms / 1000.0 if total_duration_ms > 0 else 0.0
    firing_rates = per_neuron_counts / total_duration_s if total_duration_s > 0 else np.zeros(n_neurons, dtype=np.float64)
    total_bins = int(sum(item["n_bins"] for item in recording_metrics))

    return {
        "timestamp": metadata.get("timestamp"),
        "use_h_current": bool(metadata.get("use_h_current", False)),
        "h_current_mode": metadata.get("h_current_mode", "unknown"),
        "mode": metadata.get("mode", "unknown"),
        "num_clusters": int(metadata.get("num_clusters", 0)),
        "num_neurons": int(metadata.get("num_neurons", n_neurons)),
        "num_connections": int(metadata.get("num_connections", 0)),
        "n_recordings_successful": int(len(recording_metrics)),
        "recording_duration_ms_total": float(total_duration_ms),
        "total_spikes": int(total_spikes),
        "mean_rate_hz": float(firing_rates.mean()) if firing_rates.size > 0 else 0.0,
        "median_rate_hz": float(np.median(firing_rates)) if firing_rates.size > 0 else 0.0,
        "max_rate_hz": float(firing_rates.max()) if firing_rates.size > 0 else 0.0,
        "active_neuron_fraction": float(np.mean(per_neuron_counts > 0)) if per_neuron_counts.size > 0 else 0.0,
        "max_active_fraction_100ms": float(max(item["max_active_fraction_100ms"] for item in recording_metrics)),
        "bins_ge_10pct": int(sum(item["bins_ge_10pct"] for item in recording_metrics)),
        "bins_ge_25pct": int(sum(item["bins_ge_25pct"] for item in recording_metrics)),
        "bins_ge_50pct": int(sum(item["bins_ge_50pct"] for item in recording_metrics)),
        "total_bins_100ms": int(total_bins),
        "saved_burst_count": int(sum(item["saved_burst_count"] for item in recording_metrics)),
        "detected_network_burst_count": int(sum(item["detected_network_burst_count"] for item in recording_metrics)),
        "peak_detected_burst_threshold": float(max(item["burst_threshold"] for item in recording_metrics)),
        "peak_smoothed_active_fraction": float(max(item["peak_smoothed_active_fraction"] for item in recording_metrics)),
    }


def build_network_signature(session_dir, metadata):
    """Hash the saved network structure so matched ablation pairs can be compared.

    Args:
        session_dir: Directory containing the saved session outputs.
        metadata: Parsed session metadata dictionary.

    Returns:
        A dictionary containing the resolved network path and a SHA-256 hash of the
        saved topology, or ``None`` when no network bundle can be found.
    """
    network_path = resolve_saved_path(session_dir, metadata.get("network_file"))
    if network_path is None or not Path(network_path).exists():
        candidates = sorted(glob.glob(str(session_dir / "network_*.npz")))
        if not candidates:
            return None
        network_path = Path(candidates[0])

    digest = hashlib.sha256()
    with np.load(network_path, allow_pickle=True) as network_data:
        cluster_assignments = np.asarray(network_data["cluster_assignments"], dtype=np.int64)
        neuron_positions = np.asarray(network_data["neuron_positions"], dtype=np.float64)
        digest.update(cluster_assignments.tobytes())
        digest.update(neuron_positions.tobytes())
        for connection in network_data["connections"]:
            digest.update(
                f"{int(connection[0])},{int(connection[1])},{float(connection[2]):.12g},{str(connection[3])}\n".encode("utf-8")
            )

    return {
        "network_path": str(network_path),
        "sha256": digest.hexdigest(),
    }


def load_session_summary(path_arg, burst_detection_config=None, raster_output_dir=None):
    """Load one session and compute its per-recording and aggregate metrics.

    Args:
        path_arg: Session directory or metadata-file path.
        burst_detection_config: Optional keyword arguments forwarded to
            ``detect_network_bursts``.
        raster_output_dir: Optional directory where per-recording rasters are saved.

    Returns:
        A dictionary containing the resolved session paths, per-recording metrics,
        aggregate summary, and optional network signature.
    """
    metadata_path = resolve_metadata_path(path_arg)
    session_dir = metadata_path.parent
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    successful_recordings = [entry for entry in metadata.get("recordings", []) if entry.get("success")]
    recording_paths = []
    for entry in successful_recordings:
        resolved = resolve_saved_path(session_dir, entry.get("file"))
        if resolved is not None and Path(resolved).exists():
            recording_paths.append(Path(resolved))

    # Older sessions may omit recording paths in metadata, so fall back to filename discovery.
    if not recording_paths:
        recording_paths = [Path(path) for path in sorted(glob.glob(str(session_dir / "recording[0-9][0-9][0-9].npz")))]

    recording_metrics = []
    session_label = "with_h_current" if bool(metadata.get("use_h_current", False)) else "without_h_current"
    for recording_idx, recording_path in enumerate(recording_paths):
        raster_output_path = None
        if raster_output_dir is not None:
            raster_output_path = Path(raster_output_dir) / (
                f"{metadata.get('timestamp', 'session')}_{session_label}_recording{recording_idx:03d}_raster.png"
            )
        recording_metrics.append(
            load_recording_metrics(
                recording_path,
                burst_detection_config=burst_detection_config,
                raster_output_path=raster_output_path,
                raster_title=(
                    f"{metadata.get('timestamp', 'session')} | {session_label.replace('_', ' ')} | "
                    f"recording {recording_idx:03d}"
                ),
            )
        )
    summary = aggregate_recordings(metadata, recording_metrics)
    network_signature = build_network_signature(session_dir, metadata)

    return {
        "session_dir": str(session_dir),
        "metadata_path": str(metadata_path),
        "summary": summary,
        "recordings": recording_metrics,
        "network_signature": network_signature,
    }


def summarize_group(session_summaries, use_h_current):
    """Average key metrics across sessions sharing the same h-current setting.

    Args:
        session_summaries: Iterable of session-summary dictionaries.
        use_h_current: Target h-current condition used to select the group.

    Returns:
        A grouped summary dictionary, or ``None`` if no sessions match the
        requested condition.
    """
    group = [item["summary"] for item in session_summaries if item["summary"]["use_h_current"] == use_h_current]
    if not group:
        return None

    def mean_metric(name):
        """Return the group mean for one named summary metric.

        Args:
            name: Summary-field name to average across the selected sessions.

        Returns:
            The floating-point mean for the requested metric.
        """
        return float(np.mean([item[name] for item in group]))

    return {
        "n_sessions": int(len(group)),
        "mean_total_spikes": mean_metric("total_spikes"),
        "mean_rate_hz": mean_metric("mean_rate_hz"),
        "mean_active_neuron_fraction": mean_metric("active_neuron_fraction"),
        "mean_max_active_fraction_100ms": mean_metric("max_active_fraction_100ms"),
        "mean_detected_network_burst_count": mean_metric("detected_network_burst_count"),
    }


def build_pairwise_comparison(session_summaries):
    """Build a matched with-vs-without-h-current delta summary.

    Args:
        session_summaries: Iterable of loaded session-summary dictionaries.

    Returns:
        A pairwise comparison dictionary when exactly two sessions are supplied,
        otherwise ``None``.
    """
    if len(session_summaries) != 2:
        return None

    first, second = session_summaries
    if first["summary"]["use_h_current"] and not second["summary"]["use_h_current"]:
        with_h_current = first
        without_h_current = second
    elif second["summary"]["use_h_current"] and not first["summary"]["use_h_current"]:
        with_h_current = second
        without_h_current = first
    else:
        with_h_current = first
        without_h_current = second

    signature_a = with_h_current.get("network_signature")
    signature_b = without_h_current.get("network_signature")
    topology_match = None
    # Compare saved network hashes so matched ablation pairs can be checked for identical topology.
    if signature_a is not None and signature_b is not None:
        topology_match = signature_a["sha256"] == signature_b["sha256"]

    fields = [
        "total_spikes",
        "mean_rate_hz",
        "median_rate_hz",
        "max_rate_hz",
        "active_neuron_fraction",
        "max_active_fraction_100ms",
        "bins_ge_10pct",
        "bins_ge_25pct",
        "bins_ge_50pct",
        "saved_burst_count",
        "detected_network_burst_count",
    ]
    delta = {}
    for field in fields:
        delta[field] = with_h_current["summary"][field] - without_h_current["summary"][field]

    return {
        "with_h_current": with_h_current["summary"]["timestamp"],
        "without_h_current": without_h_current["summary"]["timestamp"],
        "topology_match": topology_match,
        "delta_with_minus_without": delta,
    }


def print_session_summary(session_summary):
    """Print a readable per-session summary for the ablation report.

    Args:
        session_summary: Session-summary dictionary returned by
            ``load_session_summary``.

    Returns:
        None. The function prints the key metrics for one session.
    """
    summary = session_summary["summary"]
    print(f"\nSession: {summary['timestamp']}")
    print(f"  Path: {session_summary['session_dir']}")
    print(f"  use_h_current: {summary['use_h_current']} ({summary['h_current_mode']})")
    print(f"  Mode: {summary['mode']}")
    print(
        f"  Neurons: {summary['num_neurons']} | Connections: {summary['num_connections']} | "
        f"Recordings: {summary['n_recordings_successful']}"
    )
    print(
        f"  Duration: {summary['recording_duration_ms_total'] / 1000.0:.1f} s | "
        f"Total spikes: {summary['total_spikes']} | "
        f"Detected bursts: {summary['detected_network_burst_count']}"
    )
    print(
        f"  Mean/median/max rate (Hz): {summary['mean_rate_hz']:.4f} / "
        f"{summary['median_rate_hz']:.4f} / {summary['max_rate_hz']:.4f}"
    )
    print(
        f"  Active neuron fraction: {summary['active_neuron_fraction']:.4f} | "
        f"Max active fraction 100 ms: {summary['max_active_fraction_100ms']:.4f}"
    )
    print(
        f"  100 ms bins >=10/25/50% active: {summary['bins_ge_10pct']} / "
        f"{summary['bins_ge_25pct']} / {summary['bins_ge_50pct']} of {summary['total_bins_100ms']}"
    )
    print(
        f"  Saved burst onsets: {summary['saved_burst_count']} | "
        f"Peak burst threshold: {summary['peak_detected_burst_threshold']:.4f} | "
        f"Peak smoothed active fraction: {summary['peak_smoothed_active_fraction']:.4f}"
    )
    raster_paths = [item["raster_path"] for item in session_summary["recordings"] if item.get("raster_path")]
    if raster_paths:
        print(f"  Raster outputs: {len(raster_paths)}")


def make_json_safe(obj):
    """Convert NumPy scalars and arrays into JSON-safe Python values.

    Args:
        obj: Arbitrary nested Python or NumPy object.

    Returns:
        The same data converted into plain Python container, scalar, and list
        types that ``json.dump`` can serialize.
    """
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    """Parse CLI inputs, summarize sessions, and optionally save outputs.

    Args:
        None.

    Returns:
        None. The function prints per-session summaries, optional grouped and
        pairwise comparisons, and may write a JSON report and raster images.
    """
    parser = argparse.ArgumentParser(description="Compare saved h-current ablation sessions.")
    parser.add_argument(
        "session_paths",
        nargs="*",
        help="Session directories or session_metadata.json paths. Defaults to all sessions under 'LIF data'.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save the comparison report as JSON.",
    )
    parser.add_argument(
        "--raster-out-dir",
        default=None,
        help="Optional directory where per-recording raster plots are saved.",
    )
    parser.add_argument("--burst-activity-bin-ms", type=float, default=100.0,
                        help="Burst detection population-activity bin width in ms.")
    parser.add_argument("--burst-smooth-bins", type=int, default=3,
                        help="Burst detection smoothing width in activity bins.")
    parser.add_argument("--burst-threshold-std", type=float, default=3.0,
                        help="Burst detection threshold = mean + std_factor * std.")
    parser.add_argument("--burst-min-active-frac", type=float, default=0.10,
                        help="Minimum active-neuron fraction for burst detection.")
    parser.add_argument("--burst-min-duration-ms", type=float, default=100.0,
                        help="Minimum detected burst duration in ms.")
    parser.add_argument("--burst-merge-gap-ms", type=float, default=150.0,
                        help="Merge nearby burst segments separated by at most this gap.")
    parser.add_argument("--burst-pad-before-ms", type=float, default=100.0,
                        help="Padding before each detected burst window in ms.")
    parser.add_argument("--burst-pad-after-ms", type=float, default=250.0,
                        help="Padding after each detected burst window in ms.")
    args = parser.parse_args()

    session_paths = args.session_paths
    # With no explicit inputs, scan every session under the default data root.
    if not session_paths:
        if not DEFAULT_DATA_DIR.exists():
            raise FileNotFoundError(f"No default data directory found at {DEFAULT_DATA_DIR}")
        session_paths = [str(path) for path in sorted(DEFAULT_DATA_DIR.iterdir()) if path.is_dir()]

    burst_detection_config = {
        "activity_bin_ms": args.burst_activity_bin_ms,
        "smooth_bins": args.burst_smooth_bins,
        "threshold_std": args.burst_threshold_std,
        "min_active_fraction": args.burst_min_active_frac,
        "min_burst_duration_ms": args.burst_min_duration_ms,
        "merge_gap_ms": args.burst_merge_gap_ms,
        "pad_before_ms": args.burst_pad_before_ms,
        "pad_after_ms": args.burst_pad_after_ms,
    }

    raster_output_dir = None
    if args.raster_out_dir:
        raster_output_dir = Path(args.raster_out_dir)
        raster_output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize each input into the same per-session summary structure for later comparisons.
    session_summaries = [
        load_session_summary(
            path_arg,
            burst_detection_config=burst_detection_config,
            raster_output_dir=raster_output_dir,
        )
        for path_arg in session_paths
    ]
    for session_summary in session_summaries:
        print_session_summary(session_summary)
        for recording in session_summary["recordings"]:
            if recording.get("raster_path"):
                print(f"    Raster: {recording['raster_path']}")
            print(
                f"    Recording {Path(recording['path']).stem}: "
                f"detected_bursts={recording['detected_network_burst_count']}, "
                f"threshold={recording['burst_threshold']:.4f}"
            )

    pairwise = build_pairwise_comparison(session_summaries)
    if pairwise is not None:
        print("\nPairwise comparison (with_h_current - without_h_current)")
        print(f"  with_h_current session: {pairwise['with_h_current']}")
        print(f"  without_h_current session: {pairwise['without_h_current']}")
        print(f"  Topology match: {pairwise['topology_match']}")
        for metric_name, value in pairwise["delta_with_minus_without"].items():
            if isinstance(value, float):
                print(f"  delta_{metric_name}: {value:.6f}")
            else:
                print(f"  delta_{metric_name}: {value}")

    grouped = {
        "with_h_current": summarize_group(session_summaries, True),
        "without_h_current": summarize_group(session_summaries, False),
    }
    if grouped["with_h_current"] is not None or grouped["without_h_current"] is not None:
        print("\nGrouped summary")
        for label, summary in grouped.items():
            if summary is None:
                continue
            print(
                f"  {label}: n_sessions={summary['n_sessions']}, "
                f"mean_total_spikes={summary['mean_total_spikes']:.2f}, "
                f"mean_rate_hz={summary['mean_rate_hz']:.6f}, "
                f"mean_active_neuron_fraction={summary['mean_active_neuron_fraction']:.6f}, "
                f"mean_max_active_fraction_100ms={summary['mean_max_active_fraction_100ms']:.6f}, "
                f"mean_detected_network_burst_count={summary['mean_detected_network_burst_count']:.6f}"
            )

    report = {
        "sessions": [
            {
                "session_dir": session_summary["session_dir"],
                "metadata_path": session_summary["metadata_path"],
                "summary": session_summary["summary"],
                "recordings": session_summary["recordings"],
                "network_signature": session_summary["network_signature"],
            }
            for session_summary in session_summaries
        ],
        "pairwise": pairwise,
        "grouped": grouped,
        "burst_detection_config": burst_detection_config,
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(make_json_safe(report), handle, indent=2)
        print(f"\nSaved JSON report to: {args.json_out}")


if __name__ == "__main__":
    main()