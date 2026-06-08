import os
from pathlib import Path

import numpy as np

from .session_io import find_session_folders, load_session_recordings
from .voltage_storage import resolve_recording_voltage


def _npz_get(data, key, default=None):
    """Read a value from an ``npz``-like object with a default fallback.

    Args:
        data: Mapping-like object returned by ``np.load`` or an equivalent dict.
        key: Field name to look up.
        default: Value returned when the key is absent.

    Returns:
        The stored value for ``key`` when present, otherwise ``default``.
    """
    if key in data:
        return data[key]
    return default


def resolve_session_source(session_source=None, base_dir="LIF data"):
    """Resolve a user-facing session specifier into a concrete session triple.

    Args:
        session_source: ``latest``, a session directory, or a metadata-file path.
        base_dir: Root directory searched when ``session_source`` is omitted or set
            to ``latest``.

    Returns:
        A ``(timestamp, session_dir, metadata_file)`` tuple describing the chosen
        session.
    """
    if session_source is None or str(session_source).lower() == "latest":
        sessions = find_session_folders(base_dir)
        if not sessions:
            raise FileNotFoundError(f"No session folders found under {base_dir}")
        return sessions[-1]

    path = Path(session_source)
    if path.is_dir():
        metadata_file = path / "session_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"No session_metadata.json found in {path}")
        return path.name, str(path), str(metadata_file)

    if path.name != "session_metadata.json":
        raise ValueError("session_source must be a session directory, metadata file, or 'latest'.")

    return path.parent.name, str(path.parent), str(path)


def build_hub_cluster_info(network_data):
    """Reconstruct hub-related cluster metadata from a saved network bundle.

    Args:
        network_data: Loaded network ``.npz`` bundle containing cluster metadata.

    Returns:
        A cluster-info dictionary with hub metadata when the saved network includes
        hub fields, otherwise ``None``.
    """
    cluster_info = {
        "cluster_centers": network_data["cluster_centers"],
        "cluster_sizes": network_data["cluster_sizes"],
        "cluster_assignments": network_data["cluster_assignments"],
        "cluster_neuron_groups": list(network_data["cluster_neuron_groups"]),
    }

    if "hub_neuron_ids" not in network_data:
        return None

    cluster_info["hub_neuron_ids"] = list(network_data["hub_neuron_ids"])
    cluster_info["hub_fraction"] = float(_npz_get(network_data, "hub_fraction", 0.1))
    cluster_info["hub_between_prob"] = float(_npz_get(network_data, "hub_between_prob", 0.4))
    cluster_info["hub_weight_scale"] = float(_npz_get(network_data, "hub_weight_scale", 1.5))
    cluster_info["hub_reciprocal_factor"] = float(_npz_get(network_data, "hub_reciprocal_factor", 2.0))
    cluster_info["n_hub_connections"] = int(_npz_get(network_data, "n_hub_connections", 0))
    return cluster_info


def load_session_bundle(session_source=None, base_dir="LIF data", recording_index=0):
    """Load one session and expose a recording-centric bundle for plotting or analysis.

    Args:
        session_source: Session directory, metadata path, or `latest`.
        base_dir: Root directory containing saved sessions.
        recording_index: Which successful recording to load from the session.

    Returns:
        A dictionary containing network, recording, and derived plotting metadata.
    """
    timestamp, session_dir, metadata_file = resolve_session_source(session_source, base_dir)
    recordings, metadata = load_session_recordings(metadata_file)
    if not recordings:
        raise ValueError(f"No successful recordings found in {metadata_file}")
    if recording_index < 0 or recording_index >= len(recordings):
        raise IndexError(
            f"recording_index={recording_index} is out of range for {len(recordings)} recordings"
        )

    network_data = np.load(metadata["network_file"], allow_pickle=True)
    rec_data = recordings[recording_index]
    successful_recordings = [rec for rec in metadata["recordings"] if rec["success"]]
    recording_path = successful_recordings[recording_index]["file"]

    spike_times = rec_data["spike_times"]
    voltage_data = resolve_recording_voltage(rec_data, recording_path=recording_path, load_into_memory=False)

    burst_onset_times = rec_data["burst_onset_times"] if "burst_onset_times" in rec_data.files else None
    cluster_assignments = rec_data["resampled_cluster_assignments"]
    resampled_spikes = rec_data["resampled_spikes"]
    resampled_times = rec_data["resampled_time_points"]
    resampling_frequency = (
        float(rec_data["resampling_frequency"])
        if "resampling_frequency" in rec_data.files
        else float(metadata.get("target_freq", 10))
    )
    recording_duration = float(
        metadata.get("recording_duration", rec_data["duration"] if "duration" in rec_data.files else 60000)
    )

    network_cluster_info = {
        "cluster_centers": network_data["cluster_centers"],
        "cluster_neuron_groups": list(network_data["cluster_neuron_groups"]),
    }

    relative_session_path = os.path.relpath(session_dir, base_dir) if os.path.exists(base_dir) else session_dir
    return {
        "timestamp": timestamp,
        "session_dir": session_dir,
        "metadata_file": metadata_file,
        "relative_session_path": relative_session_path,
        "recordings": recordings,
        "metadata": metadata,
        "recording": rec_data,
        "network_data": network_data,
        "network_positions": network_data["neuron_positions"],
        "network_connections": network_data["connections"],
        "network_cluster_info": network_cluster_info,
        "hub_cluster_info": build_hub_cluster_info(network_data),
        "spike_times": spike_times,
        "cluster_spike_data": rec_data["cluster_spike_data"],
        "spike_data_dict": {idx: list(spikes) for idx, spikes in enumerate(spike_times)},
        "cluster_assignments": cluster_assignments,
        "voltage_data": voltage_data,
        "burst_onset_times": burst_onset_times,
        "stimulation_enabled": burst_onset_times is not None and len(burst_onset_times) > 0,
        "resampled_spikes": resampled_spikes,
        "resampled_times": resampled_times,
        "resampling_frequency": resampling_frequency,
        "recording_duration": recording_duration,
    }