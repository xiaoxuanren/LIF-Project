# LIF Simulation Data Structure

This document describes all files saved by the LIF network simulation pipeline.

---

## Folder Structure

```
LIF data/
└── {timestamp}/                          # e.g. 20260223_130621
    ├── network_{timestamp}.npz           # Network topology (1 per session)
    ├── recording000.npz                  # Individual recording files
  ├── recording000_voltage.h5           # Chunked full-dt voltage sidecar for recording000
    ├── recording001.npz                  # (n_recordings per session)
  ├── recording001_voltage.h5           # Optional sidecar per recording when using external HDF5 voltage storage
    ├── ...
    └── session_metadata.json             # Session config & recording manifest
```

---

## 1. `network_{timestamp}.npz` — Network Topology

Saved once per session by `save_network_structure()`.

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `connections` | ndarray (object) | `(n_connections, 4)` | Each row: `[pre_id, post_id, weight, type]` where type is `'exc'` or `'inh'` |
| `neuron_positions` | ndarray (float64) | `(n_neurons, 2)` | 2D spatial coordinates of each neuron |
| `cluster_centers` | ndarray (float64) | `(n_clusters, 2)` | 2D center positions of each cluster |
| `cluster_sizes` | ndarray (int) | `(n_clusters,)` | Number of neurons in each cluster |
| `cluster_assignments` | ndarray (int) | `(n_neurons,)` | Cluster ID for each neuron |
| `cluster_neuron_groups` | ndarray (object) | `(n_clusters,)` | Each element is a list of neuron IDs belonging to that cluster |
| `weight_params` | ndarray (object) | scalar dict | Serialized `NetworkWeightParameters`: `within_exc_range`, `between_exc_range`, `within_inh_range`, `between_inh_range`, `use_lognormal`, `lognormal_sigma` |
| `hub_neuron_ids` | ndarray (int) | `(n_hubs,)` | Global IDs of hub neurons |
| `hub_fraction` | float | scalar | Fraction of neurons per cluster designated as hubs (default 0.1) |
| `hub_between_prob` | float | scalar | Inter-cluster connection probability for hubs (default 0.4) |
| `hub_weight_scale` | float | scalar | Multiplicative weight scaling for hub synapses (default 1.5) |
| `hub_reciprocal_factor` | float | scalar | Reciprocal connection boost for hub-hub pairs (default 2.0) |
| `n_hub_connections` | int | scalar | Total number of added hub connections |

---

## 2. `recording{NNN}.npz` — Individual Recording

Saved per recording by `save_recording_data()`.

### Spike Data (original resolution)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `spike_times` | ndarray (object) | `(n_neurons,)` | Each element is a 1D array of spike times (ms) for that neuron |
| `cluster_spike_data` | ndarray (object) | `(n_clusters,)` | Nested: each element is an object array of per-neuron spike-time arrays for that cluster |

### Resampled Data (binned)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `resampled_spikes` | ndarray (int) | `(n_neurons, n_time_points)` | Binary spike matrix at `target_freq` Hz (default 10 Hz → 100 ms bins) |
| `resampled_time_points` | ndarray (float64) | `(n_time_points,)` | Time values for each resampled bin (ms) |
| `resampled_cluster_assignments` | ndarray (int) | `(n_neurons,)` | Cluster ID for each neuron |
| `resampling_frequency` | int | scalar | Target resampling frequency in Hz (default 10) |
| `resampling_interval_ms` | float | scalar | Bin width in ms (default 100.0) |
| `resampled_spike_positions` | ndarray (int) | `(n_spikes, 2)` | `[neuron_id, time_bin_idx]` pairs where resampled spikes occurred |

### Voltage Traces (optional, when `record_voltage=True`)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `voltage_times` | ndarray (float32) | `(n_voltage_samples,)` | Time values for voltage samples (ms). When raw full-dt storage is used, spacing equals `dt`. |
| `voltage_sample_rate` | float | scalar | Actual stored voltage step in ms. For current raw full-dt recordings this equals `dt` (default 0.1). |
| `voltage_storage_backend` | str | scalar | Storage backend used for the saved voltage trace. Current main-workflow default is `hdf5_external`; legacy sessions may omit this field or store `inline_npz`. |
| `voltage_units` | str | scalar | Physical units of the saved membrane voltage values. Current sessions store `mV`. |
| `voltage_traces` | ndarray (float32) | `(n_neurons, n_voltage_samples)` | Inline voltage array used by legacy sessions or smaller inline-save runs. No artificial spike waveform is stamped into the stored data. |
| `voltage_hdf5_file` | str | scalar | Relative path to the external HDF5 sidecar when `voltage_storage_backend = hdf5_external`. |
| `voltage_hdf5_dataset` | str | scalar | Dataset name inside the HDF5 sidecar. Current value: `voltage_traces`. |
| `voltage_n_samples` | int | scalar | Number of stored voltage samples when using external HDF5 voltage storage. |
| `voltage_dtype` | str | scalar | Stored voltage dtype for external HDF5 datasets. Current default: `float32`. |

When `voltage_storage_backend = hdf5_external`, the heavy full-dt voltage matrix is stored in `recordingNNN_voltage.h5`, while `recordingNNN.npz` retains spike data, resampled outputs, timing metadata, and the pointer to the voltage sidecar.

### Burst / Stimulation Data (optional)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `burst_onset_times` | ndarray (float64) | `(n_bursts,)` | Stimulus onset times in ms |

### Recording Metadata

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `recording_index` | int | scalar | 0-based index of this recording in the session |
| `timestamp` | str | scalar | Session timestamp string |
| `duration` | int | scalar | Recording duration in ms |

---

## 3. Combined Session View (in memory)

Produced in memory by `combine_session_data()` (exported from `lif_simulation`, defined in `lif_simulation/session_io.py`). All loaded recordings are merged with cumulative time offsets applied to spike times (`spike_time + cumulative_recording_offset`); resampled rasters and voltage traces are concatenated along the time axis. This function returns a dict and does **not** write a file — the legacy `recording_combined.npz` and its GNN preprocessing notebook are not part of this snapshot.

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `spike_times` | list of list | `(n_neurons,)` | Per-neuron spike times (ms) across all recordings with time offsets applied |
| `resampled_spikes` | ndarray (int) | `(n_neurons, total_time_points)` | Concatenated resampled binary spike matrix |
| `resampled_time_points` | ndarray (float64) | `(total_time_points,)` | Concatenated bin time axis with offsets applied |
| `n_recordings` | int | scalar | Number of merged recordings |
| `recording_durations` | ndarray (float64) | `(n_recordings,)` | Per-recording durations in ms |
| `total_duration` | float | scalar | Total combined duration in ms |
| `voltage_traces` | ndarray (float32) | `(n_neurons, total_voltage_samples)` | Concatenated raw membrane voltage *(included only if source recordings have voltage data)* |
| `voltage_times` | ndarray (float64) | `(total_voltage_samples,)` | Concatenated voltage time axis *(optional)* |
| `voltage_sample_rate` | float | scalar | Stored voltage step in ms *(optional)* |

---

## 4. `session_metadata.json` — Full Session Configuration

Saved by `sequential_simulation_individual_saves()`. Contains all simulation parameters and a manifest of recordings.

```json
{
  "timestamp": "20260223_130621",
  "session_dir": "LIF data\\20260223_130621",
  "n_recordings": 5,
  "recording_duration": 60000,
  "num_clusters": 20,
  "num_neurons": 299,
  "num_connections": 2808,
  "target_freq": 10,
  "dt": 0.1,
  "record_voltage": true,
  "voltage_sample_rate": 0.1,
  "requested_voltage_sample_rate": 1.0,
  "voltage_trace_mode": "raw_full_dt",
  "voltage_storage_backend": "hdf5_external",
  "voltage_chunk_samples": 4096,
  "space_size": 15,
  "max_connection_distance": 8.0,
  "network_file": "LIF data\\20260223_130621\\network_20260223_130621.npz",
  "mode": "stimulus_driven_bursting",
  "use_h_current": true,
  "h_current_mode": "enabled",
  "background_input": false,
  "burst_interval": 7000,
  "burst_interval_jitter": 1500,
  "cluster_fraction": 0.7,
  "neurons_per_cluster": 6,
  "stim_amplitude": [2.0, 3.5],
  "stim_duration": [10, 30],
  "hub_fraction": 0.1,
  "hub_between_prob": 0.4,
  "hub_weight_scale": 1.5,
  "hub_reciprocal_factor": 2.0,
  "n_hub_neurons": 20,
  "n_hub_connections": 1133,
  "recordings": [
    {
      "index": 0,
      "file": "LIF data\\20260223_130621\\recording000.npz",
      "success": true,
      "num_spikes": 9075
    }
  ]
}
```

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | str | Session identifier (format: `YYYYMMDD_HHMMSS`) |
| `recording_duration` | int | Per-recording duration in ms |
| `num_clusters` | int | Number of neuron clusters |
| `num_neurons` | int | Total neuron count |
| `num_connections` | int | Total synapse count |
| `target_freq` | int | Resampling frequency (Hz) |
| `dt` | float | Simulation time step (ms) |
| `voltage_sample_rate` | float | Actual stored voltage step in ms for saved traces |
| `requested_voltage_sample_rate` | float | Legacy requested voltage sampling interval retained for compatibility |
| `voltage_trace_mode` | str | Records whether saved voltage traces use raw full-dt storage or a legacy sampled mode |
| `voltage_storage_backend` | str | Records whether saved voltage traces live inline or in external chunked HDF5 sidecars |
| `use_h_current` | bool | Whether session neurons used slow h-current dynamics |
| `h_current_mode` | str | Records whether h-current was enabled or disabled by skipping the update path |
| `recordings` | list | Manifest of recording files with success status and spike counts |

---

## Typical Dimensions

| Quantity | Typical Value |
|----------|---------------|
| Neurons | ~300 |
| Clusters | 20 |
| Connections | ~2,800 |
| Hub neurons | ~20 (~7%) |
| Recordings per session | 1–20 |
| Per-recording duration | 60,000 ms (60 s) |
| Resampled time points per recording | 600 (at 10 Hz, 100 ms bins) |
| Voltage samples per recording | 600,000 (at 0.1 ms rate) |
