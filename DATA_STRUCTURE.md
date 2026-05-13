# LIF Simulation Data Structure

This document describes all files saved by the LIF network simulation pipeline and the GNN preprocessing notebook.

---

## Folder Structure

```
LIF data/
└── {timestamp}/                          # e.g. 20260223_130621
    ├── network_{timestamp}.npz           # Network topology (1 per session)
    ├── recording000.npz                  # Individual recording files
    ├── recording001.npz                  # (n_recordings per session)
    ├── ...
    ├── recording_combined.npz            # All recordings merged (created by process_existing_for_gnn.ipynb)
    ├── session_metadata.json             # Session config & recording manifest
    └── session_gnn_metadata.json         # Simplified metadata for GNN pipeline
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
| `resampled_spikes` | ndarray (int) | `(n_neurons, n_time_points)` | Binary spike matrix at `target_freq` Hz (default 20 Hz → 50 ms bins) |
| `resampled_time_points` | ndarray (float64) | `(n_time_points,)` | Time values for each resampled bin (ms) |
| `resampled_cluster_assignments` | ndarray (int) | `(n_neurons,)` | Cluster ID for each neuron |
| `resampling_frequency` | int | scalar | Target resampling frequency in Hz (default 20) |
| `resampling_interval_ms` | float | scalar | Bin width in ms (default 50.0) |
| `resampled_spike_positions` | ndarray (int) | `(n_spikes, 2)` | `[neuron_id, time_bin_idx]` pairs where resampled spikes occurred |

### Voltage Traces (optional, when `record_voltage=True`)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `voltage_traces` | ndarray (float64) | `(n_neurons, n_voltage_samples)` | Membrane voltage time series for each neuron (mV). Includes artificial +20 mV spike peaks for visualization. |
| `voltage_times` | ndarray (float64) | `(n_voltage_samples,)` | Time values for voltage samples (ms) |
| `voltage_sample_rate` | float | scalar | Voltage sampling interval in ms (default 1.0) |

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

## 3. `recording_combined.npz` — Combined Recording for GNN

Created by `process_existing_for_gnn.ipynb`. All individual recordings are merged with time offsets (`spike_time + rec_index × recording_duration`). Voltage traces are concatenated along the time axis.

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `spike_times` | ndarray (object) | `(n_neurons,)` | Each element is a sorted 1D array of spike times (ms) across all recordings with time offsets applied |
| `duration_ms` | int | scalar | Total combined duration in ms (`n_recordings × recording_duration`) |
| `n_neurons` | int | scalar | Number of neurons |
| `voltage_traces` | ndarray (float64) | `(n_neurons, total_voltage_samples)` | Concatenated membrane voltage across all recordings *(included if source recordings have voltage data)* |
| `voltage_times` | ndarray (float64) | `(total_voltage_samples,)` | Time points for concatenated voltage *(optional)* |
| `voltage_sample_rate` | float | scalar | Voltage sampling interval in ms *(optional)* |

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
  "target_freq": 20,
  "dt": 0.1,
  "record_voltage": true,
  "voltage_sample_rate": 1.0,
  "space_size": 15,
  "max_connection_distance": 8.0,
  "network_file": "LIF data\\20260223_130621\\network_20260223_130621.npz",
  "mode": "stimulus_driven_bursting",
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
| `recordings` | list | Manifest of recording files with success status and spike counts |

---

## 5. `session_gnn_metadata.json` — GNN Pipeline Metadata

Created by `process_existing_for_gnn.ipynb`. Points to the combined recording file.

```json
{
  "timestamp": "20260223_130621",
  "n_recordings_combined": 5,
  "recording_duration": 300000,
  "num_clusters": 20,
  "num_neurons": 299,
  "num_connections": 2808,
  "network_file": "LIF data/20260223_130621/network_20260223_130621.npz",
  "recordings": [
    {
      "index": 0,
      "file": "LIF data/20260223_130621/recording_combined.npz",
      "success": true,
      "num_spikes": 41392
    }
  ]
}
```

**Key difference from `session_metadata.json`:** The GNN metadata wraps all recordings into a single entry pointing to `recording_combined.npz`, with `recording_duration` set to the **total combined time** (not per-recording). Paths use forward slashes for cross-platform compatibility.

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
| Resampled time points per recording | 1,200 (at 20 Hz) |
| Voltage samples per recording | 60,000 (at 1.0 ms rate) |
