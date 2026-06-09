import json
import os

import numpy as np

from .analysis import organize_spike_data_by_cluster, resample_data
from .voltage_storage import resolve_recording_voltage


class LoadedRecording:
    """Lazy wrapper around one saved recording file and its filesystem path."""

    def __init__(self, path):
        self.path = path
        self._data = np.load(path, allow_pickle=True)

    @property
    def files(self):
        return self._data.files

    def __contains__(self, key):
        return key in self._data.files

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        if key in self:
            return self._data[key]
        return default

    def close(self):
        self._data.close()


def save_network_structure(connections, neuron_positions, cluster_info, weight_params, timestamp, save_dir):
    """Save the generated network topology and spatial metadata for a session.

    Args:
        connections: Connection table describing all generated synapses.
        neuron_positions: Spatial coordinates for every neuron.
        cluster_info: Cluster metadata returned by network construction.
        weight_params: ``NetworkWeightParameters`` instance used to sample weights.
        timestamp: Session timestamp used to name the output folder and file.
        save_dir: Root directory where session folders are stored.

    Returns:
        The path to the compressed ``.npz`` file containing the saved network
        structure.
    """
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    filename = os.path.join(session_dir, f"network_{timestamp}.npz")

    cluster_neuron_groups = np.array(cluster_info["cluster_neuron_groups"], dtype=object)
    save_dict = {
        "connections": connections,
        "neuron_positions": neuron_positions,
        "cluster_centers": cluster_info["cluster_centers"],
        "cluster_sizes": cluster_info["cluster_sizes"],
        "cluster_assignments": cluster_info["cluster_assignments"],
        "cluster_neuron_groups": cluster_neuron_groups,
        "weight_params": vars(weight_params),
    }

    if "hub_neuron_ids" in cluster_info:
        save_dict["hub_neuron_ids"] = np.array(cluster_info["hub_neuron_ids"])
        save_dict["hub_fraction"] = cluster_info.get("hub_fraction", 0.1)
        save_dict["hub_between_prob"] = cluster_info.get("hub_between_prob", 0.4)
        save_dict["hub_weight_scale"] = cluster_info.get("hub_weight_scale", 1.5)
        save_dict["hub_reciprocal_factor"] = cluster_info.get("hub_reciprocal_factor", 2.0)
        save_dict["n_hub_connections"] = cluster_info.get("n_hub_connections", 0)

    if "baseline_currents" in cluster_info:
        save_dict["baseline_currents"] = np.asarray(cluster_info["baseline_currents"], dtype=float)
        save_dict["baseline_drive_mean"] = cluster_info.get("baseline_drive_mean", 0.0)
        save_dict["baseline_drive_sd"] = cluster_info.get("baseline_drive_sd", 0.0)
        save_dict["baseline_drive_seed"] = cluster_info.get("baseline_drive_seed", 0)
        save_dict["baseline_excitatory_only"] = cluster_info.get("baseline_excitatory_only", True)
        save_dict["exc_weight_scale"] = cluster_info.get("exc_weight_scale", 1.0)
        save_dict["noise_sigma"] = np.asarray(cluster_info.get("noise_sigma", []), dtype=float)

    np.savez_compressed(filename, **save_dict)
    print(f"Network structure saved to: {filename}")
    if "hub_neuron_ids" in cluster_info:
        print(f"  - Hub neurons: {len(cluster_info['hub_neuron_ids'])}")
        print(f"  - Hub connections: {cluster_info.get('n_hub_connections', 0)}")
    return filename


def save_recording_data(
    spike_data,
    voltage_data,
    cluster_info,
    recording_idx,
    timestamp,
    save_dir,
    target_freq=10,
    duration=60000,
    burst_onset_times=None,
    burst_windows=None,
    interburst_windows=None,
):
    """Persist one recording's spike, voltage, and resampled analysis outputs.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        voltage_data: Optional full-resolution voltage bundle returned by the simulator.
        cluster_info: Saved cluster metadata used to organize outputs by cluster.
        recording_idx: Recording index within the session.
        timestamp: Session timestamp used in filenames.
        save_dir: Root output directory for session artifacts.
        target_freq: Frequency used to build saved resampled spike rasters.
        duration: Recording duration in milliseconds.
        burst_onset_times: Optional stimulation onset times to save with the recording.
        burst_windows: Optional detected network-burst windows for spontaneous recordings.
        interburst_windows: Optional complement windows between detected bursts.

    Returns:
        The path to the saved compressed recording file.
    """
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    filename = os.path.join(session_dir, f"recording{recording_idx:03d}.npz")

    spike_times_list = np.array([spike_data[i] for i in range(len(spike_data))], dtype=object)
    cluster_spike_data = organize_spike_data_by_cluster(spike_data, cluster_info)
    cluster_spike_data = np.array(
        [np.array(cluster, dtype=object) for cluster in cluster_spike_data],
        dtype=object,
    )

    resampled_spikes, resampled_time_points, resampled_spike_positions = resample_data(
        spike_data,
        cluster_info["cluster_assignments"],
        target_freq,
        duration,
    )

    save_dict = {
        "spike_times": spike_times_list,
        "cluster_spike_data": cluster_spike_data,
        "resampled_spikes": resampled_spikes,
        "resampled_time_points": resampled_time_points,
        "resampled_cluster_assignments": cluster_info["cluster_assignments"],
        "resampling_frequency": target_freq,
        "resampling_interval_ms": 1000.0 / target_freq,
        "resampled_spike_positions": resampled_spike_positions,
        "recording_index": recording_idx,
        "timestamp": timestamp,
        "duration": duration,
    }

    if voltage_data is not None:
        save_dict["voltage_sample_rate"] = voltage_data["sample_rate"]
        save_dict["voltage_times"] = voltage_data["times"]
        save_dict["voltage_storage_backend"] = voltage_data.get("storage_backend", "inline_npz")
        save_dict["voltage_units"] = voltage_data.get("units", "mV")
        if voltage_data.get("storage_backend") == "hdf5_external":
            save_dict["voltage_hdf5_file"] = os.path.relpath(voltage_data["voltage_file"], session_dir)
            save_dict["voltage_hdf5_dataset"] = voltage_data.get("voltage_dataset", "voltage_traces")
            save_dict["voltage_n_samples"] = int(voltage_data["n_samples"])
            save_dict["voltage_dtype"] = voltage_data.get("dtype", "float32")
        else:
            save_dict["voltage_traces"] = voltage_data["traces"]

    if burst_onset_times is not None:
        save_dict["burst_onset_times"] = np.array(burst_onset_times)

    if burst_windows is not None:
        save_dict["burst_windows"] = np.asarray(burst_windows, dtype=float).reshape(-1, 2)
    if interburst_windows is not None:
        save_dict["interburst_windows"] = np.asarray(interburst_windows, dtype=float).reshape(-1, 2)

    np.savez_compressed(filename, **save_dict)
    print(f"Recording {recording_idx} saved to: {filename}")
    print(f"  - Original spike times: {len(spike_times_list)} neurons")
    print(
        f"  - Resampled data: {resampled_spikes.shape[0]} neurons x "
        f"{resampled_spikes.shape[1]} time points"
    )
    if voltage_data is not None:
        if voltage_data.get("storage_backend") == "hdf5_external":
            print(
                "  - Voltage traces: external HDF5 "
                f"({len(spike_times_list)} neurons x {int(voltage_data['n_samples'])} time points)"
            )
        else:
            print(
                f"  - Voltage traces: {voltage_data['traces'].shape[0]} neurons x "
                f"{voltage_data['traces'].shape[1]} time points"
            )
    if burst_onset_times is not None:
        print(f"  - Burst onset times: {len(burst_onset_times)} stimuli")
    if burst_windows is not None:
        print(f"  - Detected burst windows: {len(burst_windows)}")
    if interburst_windows is not None:
        print(f"  - Inter-burst windows: {len(interburst_windows)}")
    return filename


def load_single_recording(filename):
    """Load one saved recording bundle from disk.

    Args:
        filename: Path to a saved ``recordingXXX.npz`` file.

    Returns:
        A lazy wrapper exposing the stored arrays, metadata, and recording path.
    """
    return LoadedRecording(filename)


def _resolve_metadata_file(session_source):
    """Normalize a session directory or file path to ``session_metadata.json``.

    Args:
        session_source: Session directory path or direct metadata-file path.

    Returns:
        A filesystem path pointing to the session metadata JSON file.
    """
    if os.path.isdir(session_source):
        return os.path.join(session_source, "session_metadata.json")
    return session_source


def load_session_recordings(metadata_file):
    """Load every successful recording referenced by one session metadata file.

    Args:
        metadata_file: Path to ``session_metadata.json`` or to its containing
            session directory.

    Returns:
        A tuple ``(recordings, metadata)`` where ``recordings`` is a list of loaded
        recording bundles and ``metadata`` is the parsed JSON session metadata.
    """
    metadata_file = _resolve_metadata_file(metadata_file)
    with open(metadata_file, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    recordings = []
    for rec_info in metadata["recordings"]:
        if rec_info["success"]:
            recordings.append(load_single_recording(rec_info["file"]))
    return recordings, metadata


def find_session_folders(base_dir="LIF data"):
    """Discover all saved session folders below the configured output root.

    Args:
        base_dir: Root directory containing timestamped session subfolders.

    Returns:
        A sorted list of ``(timestamp, session_dir, metadata_file)`` tuples.
    """
    sessions = []
    if not os.path.exists(base_dir):
        return sessions

    for root, dirs, files in os.walk(base_dir):
        if "session_metadata.json" in files:
            timestamp = os.path.basename(root)
            metadata_file = os.path.join(root, "session_metadata.json")
            sessions.append((timestamp, root, metadata_file))
            dirs[:] = []

    sessions.sort(key=lambda item: (item[0], item[1]))
    return sessions


def _record_has_key(record, key):
    """Check whether an ``npz``-like record exposes a requested field.

    Args:
        record: Mapping-like object returned from ``np.load`` or an equivalent dict.
        key: Field name to test for.

    Returns:
        ``True`` when the record contains ``key``, otherwise ``False``.
    """
    if hasattr(record, "files"):
        return key in record.files
    return key in record


def combine_session_data(recordings):
    """Concatenate multiple saved recordings into one continuous session view.

    Args:
        recordings: Sequence of loaded recording bundles from the same session.

    Returns:
        A combined dictionary containing concatenated spike, raster, and optional
        voltage data, or ``None`` when ``recordings`` is empty.
    """
    if not recordings:
        return None

    n_neurons = len(recordings[0]["spike_times"])
    all_spike_times = [[] for _ in range(n_neurons)]
    all_resampled_spikes = []
    all_resampled_times = []
    all_voltage_traces = []
    all_voltage_times = []
    recording_durations = []
    time_offset = 0.0
    voltage_sample_rate = None

    for rec in recordings:
        rec_duration = float(rec["duration"]) if _record_has_key(rec, "duration") else 60000.0
        recording_durations.append(rec_duration)

        for neuron_id, neuron_spikes in enumerate(rec["spike_times"]):
            all_spike_times[neuron_id].extend(float(t) + time_offset for t in neuron_spikes)

        all_resampled_spikes.append(rec["resampled_spikes"])
        all_resampled_times.append(rec["resampled_time_points"] + time_offset)

        voltage_bundle = resolve_recording_voltage(
            rec,
            recording_path=getattr(rec, "path", None),
            load_into_memory=True,
        )
        if voltage_bundle is not None and voltage_bundle["times"] is not None:
            all_voltage_traces.append(np.asarray(voltage_bundle["traces"], dtype=np.float32))
            all_voltage_times.append(np.asarray(voltage_bundle["times"], dtype=np.float64) + time_offset)
            if voltage_bundle["sample_rate"] is not None:
                voltage_sample_rate = float(voltage_bundle["sample_rate"])

        time_offset += rec_duration

    combined = {
        "spike_times": all_spike_times,
        "resampled_spikes": np.concatenate(all_resampled_spikes, axis=1),
        "resampled_time_points": np.concatenate(all_resampled_times),
        "n_recordings": len(recordings),
        "recording_durations": np.array(recording_durations),
        "total_duration": time_offset,
    }

    if all_voltage_traces:
        combined["voltage_traces"] = np.concatenate(all_voltage_traces, axis=1)
        combined["voltage_times"] = np.concatenate(all_voltage_times)
        combined["voltage_sample_rate"] = voltage_sample_rate

    return combined