# CLAUDE.md — LIF Neuron Network with Clustered Bursting

## Project Goal

Spiking neural network simulation using Leaky Integrate-and-Fire (LIF) neurons in a clustered topology. Generates stimulus-driven network bursts with inter-burst spontaneous activity. Data is used for downstream GNN-based connectivity prediction. Default: ~300 neurons in ~20 clusters over 60 seconds. Neuron count must be easily scalable.

## Key Files

### Main Simulation
- `LIF_network_simulation_network_burst.ipynb` — Primary notebook. Defines LIF neurons, synapses, clustered network creation, periodic stimulation, simulation engine, data saving/loading, and visualization. **This is the main entry point.**
- `LIF_network_simulation_self_sustained_random_firing.ipynb` — Alternative variant with self-sustained activity.
- `11-15_2025_simulate_network.ipynb` — Earlier simulation notebook.
- `11-15-2025 combine individual recorings.ipynb` — Combines individual recording files.

### Analysis Scripts
- `analyze_firing_rate.py` — Firing rate analysis utilities.
- `analyze_gnn_feasibility.py` — Assesses feasibility of GNN-based connectivity prediction.
- `gnn_connectivity_prediction.py` — GNN model for predicting network connectivity from spike data.
- `gnn_cross_network_prediction.py` — Cross-network GNN prediction.
- `predict_new_network.py` — Run trained GNN on new network data.
- `visualize_connectivity_prediction.py` — Visualize GNN prediction results.
- `process_existing_for_gnn.ipynb` — Preprocess existing simulation data for GNN input.

### Connectivity Inference Baselines
- `perceptron_connectivity.py` — Perceptron-based weight learning baseline (Ren, Bok, Vareberg, Hai 2023). Learns synaptic weights per postsynaptic neuron from binary spike trains at 1ms using the perceptron update rule. Outputs: `perceptron_outputs/`.
- `spike_train_connectivity.py` — CNN/LSTM/Perceptron models for connectivity inference from raw spike trains. For each postsynaptic neuron, takes K nearest spatial neighbors as candidates, feeds paired spike trains through a temporal model. Learned activation threshold. Outputs: `spike_cnn_outputs/`.
- `learned_lif_connectivity.py` — Differentiable LIF model for connectivity inference. For each postsynaptic neuron, simulates membrane dynamics step-by-step: weighted sum of delayed presynaptic inputs → learnable threshold → predict spike. Learned weights directly reveal connectivity. Learnable parameters: synaptic weights, membrane leak, threshold, delays. Outputs: `learned_lif_outputs/`.

### Utility Scripts
- `resample_hires.py` — Resamples saved recordings in-place: replaces `resampled_spikes` with 0.1ms resolution binary spike trains (dt=0.1ms, 10kHz). Overwrites existing recording .npz files.
- `create_presentation.py` — Generates `LIF_Network_GNN_Presentation.pptx` summarizing the project (simulation design, GNN architecture, prediction pipeline).

### Output Directories
- `LIF data/` — Simulation output in timestamped session folders:
  ```
  LIF data/<timestamp>/
  ├── network_<timestamp>.npz    # Network structure (connections, positions, clusters)
  ├── recording000.npz           # Individual recordings (spikes, voltages, resampled data)
  ├── recording001.npz
  └── session_metadata.json      # Session parameters and metadata
  ```
- `gnn_outputs/` — GNN model outputs (trained models, figures, prediction results).

## Environment
- Python 3.9 (`.venv/` in project root)
- Key dependencies: numpy, matplotlib, json, pathlib
- GNN scripts additionally require: torch, torch_geometric

## Code Style
- Jupyter notebook cells organized by numbered Parts (0-11)
- Functions use descriptive names with docstring-style inline comments
- Parameters use SI-adjacent units: mV, ms, nA, MOhm, Hz
- Data saved as compressed `.npz` (numpy) and `.json` (metadata)

## Common Tasks
- **Running a simulation**: Execute all cells in the main notebook. Adjustable params in Part 10.
- **Modifying neuron parameters**: Edit `LIFNeuron.__init__` in Part 1. Separate blocks for exc vs inh.
- **Modifying connectivity**: Edit `create_clustered_network()` in Part 2 and `NetworkWeightParameters` in Part 1.
- **Adding new analysis**: Add functions after Part 5 or in separate `.py` files.

---

## EXISTING NOTEBOOK STRUCTURE

The simulation lives in a single Jupyter notebook (`LIF_network_simulation_network_burst_copy.ipynb`) organized in 11 parts:

### Classes:
- **`LIFNeuron`** — Leaky integrate-and-fire neuron with spike-frequency adaptation (SFA). Has exc/inh subtypes with different tau_m, adaptation parameters. State: membrane voltage `v`, adaptation current `i_adapt`, synaptic current `i_syn`, external current `i_ext`. Key method: `update(t, dt)` does Euler integration with Gaussian noise, refractory period, voltage floor, and adaptation decay. `reset_state()` reinitializes for new recordings.
- **`ExpSynapse`** — Current-based exponential synapse with delay queue. AMPA (tau=3ms) and GABA (tau=10ms). `receive_spike(t)` adds to pending queue, `update(t, dt)` processes arrivals and decays current.
- **`NetworkWeightParameters`** — Stores weight ranges for within/between cluster exc/inh connections. Supports lognormal distribution.

### Network creation (Part 2):
- `generate_non_overlapping_cluster_positions()` — Places cluster centers in 2D space
- `calculate_connection_probability()` — Distance-based with Gaussian decay, different base prob for within vs between cluster
- `get_connection_weight()` — Draws from lognormal distribution, clipped to range
- **`create_clustered_network()`** — Main builder. Creates neurons, assigns to clusters, creates synapses based on distance/probability. Returns neurons list, synapses list, connections array, positions, cluster_info dict.

### Stimulation (Part 3):
- `create_initial_stimulation()` — Single pulse at t=0 (not used in current mode)
- **`create_periodic_cluster_stimulation()`** — Generates periodic stimulus events. At each `burst_interval`, selects `cluster_fraction` of clusters, picks `neurons_per_cluster` random neurons, creates current injection events with small jitter. Returns sorted list of `(time, neuron_id, amplitude, duration)` tuples.

### Simulation engine (Part 4):
- **`simulate_network()`** — Main loop. Iterates `num_steps = duration/dt`. Each step: applies/expires stimulations, resets synaptic currents, updates all synapses, updates all neurons, propagates spikes to postsynaptic synapses. Records spike times and optionally voltage traces. **Performance bottleneck**: iterates ALL synapses per step to update, and iterates ALL synapses again per spike to find matching pre_neuron_id.

### Data processing (Part 5):
- `organize_spike_data_by_cluster()` — Groups spike times by cluster
- **`resample_data(target_freq, duration)`** — Bins spike trains into time windows at given frequency. Returns binary matrix `(n_neurons, n_timebins)`. **Use `target_freq=10` to match reference recording rate.**
- `report_network_statistics()` — Prints firing rates, connection counts, weight stats

### Saving/loading (Parts 6, 8):
- `save_network_structure()` / `save_recording_data()` — Saves to `.npz` in timestamped subfolder
- `load_single_recording()` / `load_session_recordings()` / `find_session_folders()` — Loading utilities
- Session structure: `LIF data/{timestamp}/network_{ts}.npz`, `recording000.npz`, `session_metadata.json`

### Sequential simulation (Part 7):
- **`sequential_simulation_individual_saves()`** — Top-level function. Creates network once, then runs N recordings with neuron state reset between each. Creates periodic stimulation per recording. Saves everything with metadata.

### Plotting (Part 9):
- `plot_raster()` — Spike raster sorted by cluster
- `plot_voltage_traces()` — Individual neuron membrane potential with spike markers
- `plot_voltage_heatmap()` — Population voltage activity as heatmap
- `plot_firing_rates()` — Histogram + per-neuron bar chart

### Analysis (Part 11):
- `analyze_spike_trains()` — Comprehensive analysis: firing rates, ISI distribution, CV of ISI, temporal dynamics (population rate in 100ms bins), cluster-level rates, connectivity stats, quality verdict with ✓/✗ checks.

### Current execution parameters (Part 10):
```python
sequential_simulation_individual_saves(
    n_recordings=1, recording_duration=20000, num_clusters=20,
    neurons_per_cluster_range=(15, 20), inhibitory_probability=0.2,
    within_cluster_prob=0.5, between_cluster_prob=0.25,
    target_freq=20, dt=0.1, record_voltage=True, voltage_sample_rate=1.0,
    space_size=12, burst_interval=6000, cluster_fraction=0.8,
    neurons_per_cluster=5, stim_amplitude_range=(2.0, 3.5),
    stim_duration_range=(5, 15)
)
```

60-second recording, ~700 neurons (we target the same pattern at ~300 neurons):

- **Network bursts** at ~5s, 10s, 18s, 23s, 28s, 35s, 40s, 45s, 50s — roughly every **5-8s with jitter**
- **Burst shape**: Sharp vertical columns, ~200-500ms wide, recruiting **most neurons** simultaneously
- **Between bursts**: Sparse scattered individual spikes — a few dots per neuron, NOT a haze, NOT silence
- **Inter-burst firing rate**: ~0.1-0.5 Hz per neuron (very low, just occasional spikes)
- **Burst participation**: Most (70-90%) neurons fire during each burst, but not identically every time
- **Burst peak rate**: Very high — nearly all neurons fire within a short window

### CRITICAL: Reference figure was recorded at 10 Hz sampling frequency
The reference data comes from a **10 Hz recording** (one sample every 100ms). This is the actual acquisition rate, not a post-hoc resampling. Consequences:
- Each time bin is **100ms wide** — any spike(s) within a 100ms window appear as a single mark
- Multiple spikes within one bin are **collapsed** into one event (no sub-bin temporal resolution)
- Each spike marker represents "at least one spike occurred in this 100ms window"
- Bursts that last ~200-500ms span only **2-5 bins** in the recording
- Inter-burst spikes appear as visible dashes spaced ≥100ms apart

**Our simulation** runs at dt=0.1ms (10 kHz effective resolution). To produce a comparable raster:
1. Simulate at full dt=0.1ms resolution for accurate dynamics
2. **For comparison plots**: Bin the output spike trains into 100ms windows (10 Hz) — mark a bin as active if ≥1 spike occurred. Use the existing `resample_data()` function with `target_freq=10`
3. **For analysis**: Work with the raw spike times (full precision) for ISI, firing rate, burst detection etc.
4. **For saving data that matches the reference format**: Save the 10 Hz binned binary matrix alongside the raw spike times

The existing notebook's `resample_data()` function does exactly this binning. Set `target_freq=10` to match the reference recording rate.

### What this means mechanistically:
The network is **NEAR-CRITICAL** — it maintains continuous low-rate asynchronous firing between bursts, but cannot generate synchronized network bursts on its own. Bursts happen ONLY when an external periodic stimulus triggers coordinated firing. Between stimuli, neurons fire sporadically due to noise + weak recurrent excitation.

**The inter-burst spikes are real and necessary** — they are NOT an artifact to be eliminated.

---

## DIAGNOSED PROBLEMS IN CURRENT OUTPUT (v1)

Your current raster (20s, ~350 neurons) shows:
1. **~20 bursts in 20s** — the network auto-bursts every ~1s, ignoring the 6000ms stimulus
2. **Inter-burst activity exists** (good!) but bursts are self-generated (bad!)
3. **Uneven cluster recruitment** — neurons 170-270 form dense blocks while others are lighter

**Root cause**: Excitatory weights are too strong → recurrent excitation doesn't just sustain low-rate firing, it also triggers synchronized bursts without external stimulus → the 6s periodic stimulus is invisible because the network is auto-bursting.

---

## FIX STRATEGY

### Principle: NEAR-CRITICAL network with stimulus-driven bursts
The network must have **two distinct activity modes**:
1. **Between bursts**: Low-rate asynchronous firing (~0.1-0.5 Hz per neuron). Neurons fire occasionally from noise + weak recurrent excitation. This is a steady drizzle of spikes — NOT silence.
2. **During bursts**: High-rate synchronized firing triggered by external stimulus. The stimulus pushes enough neurons past threshold simultaneously to recruit the full network via recurrent excitation.

The inter-burst activity comes from:
- **Noise** (noise_sigma) pushing neurons close to threshold occasionally past it
- **Weak recurrent excitation** — when one neuron fires, it gives a small nudge to its neighbors (not enough to trigger a cascade, but enough to slightly elevate network activity above pure noise)

The key tuning challenge: excitatory weights must be **strong enough** to sustain inter-burst firing and propagate stimulus-driven bursts, but **weak enough** that spontaneous spikes don't cascade into self-generated bursts.

### Validation test (run this FIRST):
Run 10 seconds with NO stimulation (`stimulation_events = []`).
- ✅ PASS: Scattered asynchronous spikes (~0.1-0.5 Hz per neuron) but **NO synchronized bursts** (no vertical columns)
- ❌ FAIL (too active): Self-generated bursts visible → reduce excitatory weights
- ❌ FAIL (too quiet): Near silence, < 0.05 Hz per neuron → increase noise_sigma or v_rest

---

## PARAMETER CHANGES (from current notebook values)

### 1. Neuron parameters

```python
class LIFNeuron:
    def __init__(self, neuron_id, is_inhibitory=False):
        # ... (existing code) ...

        # === CHANGED PARAMETERS ===
        self.v_rest = -62.0        # Was -60.0. Slightly farther from threshold (12mV gap).
                                    # NOT -65 — that kills inter-burst spikes.

        self.noise_sigma = 1.0     # Was 1.2. Slight reduction only.
                                    # NOT 0.5 — need enough noise for inter-burst activity.

        if is_inhibitory:
            self.adaptation_increment = 0.01   # Was 0.005
            self.tau_adaptation = 100.0        # Was 50.0
        else:
            self.adaptation_increment = 0.04   # Was 0.02. Stronger burst termination.
            self.tau_adaptation = 400.0        # Was 200.0. Longer recovery between bursts.
```

### 2. Weight parameters

```python
class NetworkWeightParameters:
    def __init__(self):
        # Moderate reduction in excitation — prevent auto-bursting but keep inter-burst firing
        self.within_exc_range = (0.25, 0.45)    # Was (0.45, 0.70). ~40% reduction.
        self.between_exc_range = (0.20, 0.35)   # Was (0.40, 0.60). ~45% reduction.

        # Slight increase in inhibition — helps terminate bursts
        self.within_inh_range = (0.35, 0.55)    # Was (0.30, 0.50). ~15% increase.
        self.between_inh_range = (0.28, 0.45)   # Was (0.25, 0.40). ~12% increase.
```

### 3. Stimulation parameters

```python
# Main execution call:
burst_interval = 7000,                 # Was 6000. ~7s between bursts.
cluster_fraction = 0.7,               # Was 0.8. Stimulate 70% of clusters.
neurons_per_cluster = 6,              # Was 5. Slightly more neurons seeded.
stim_amplitude_range = (2.0, 3.5),    # Keep similar — network is near-critical, doesn't need huge stimulus.
stim_duration_range = (5, 20),         # Was (5, 15). Slightly longer.
```

### 4. Add jitter to burst_interval

```python
# In create_periodic_cluster_stimulation(), replace:
#   burst_time = burst_idx * burst_interval
# With:
jitter = np.random.uniform(-1500, 1500)
burst_time = burst_idx * burst_interval + jitter
burst_time = max(0, burst_time)
```

### 5. Network connectivity

```python
between_cluster_prob = 0.15,           # Was 0.25. Reduce to prevent self-sustaining bursts.
max_connection_distance = 8.0,         # Was 5.0. Wider reach for uniform recruitment.
space_size = 15,                       # Was 12. More room for clusters.
```

### 6. Keep neuron count at ~300 (scalable)

```python
neurons_per_cluster_range = (12, 18),  # Was (15, 20). ~300 total with 20 clusters.
```

---

## SUMMARY TABLE: ALL CHANGES

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| **v_rest** | -60 mV | **-62 mV** | Slightly farther from threshold, still allows inter-burst spikes |
| **noise_sigma** | 1.2 mV | **1.0 mV** | Slight reduction, preserves inter-burst activity |
| **exc adaptation_increment** | 0.02 nA | **0.04 nA** | Stronger burst termination |
| **exc tau_adaptation** | 200 ms | **400 ms** | Longer recovery between bursts |
| **inh adaptation_increment** | 0.005 nA | **0.01 nA** | Moderate increase |
| **inh tau_adaptation** | 50 ms | **100 ms** | Moderate increase |
| **within_exc_range** | (0.45, 0.70) | **(0.25, 0.45)** | Prevent auto-bursting, keep inter-burst firing |
| **between_exc_range** | (0.40, 0.60) | **(0.20, 0.35)** | Prevent auto-bursting, keep inter-burst firing |
| **within_inh_range** | (0.30, 0.50) | **(0.35, 0.55)** | Faster burst termination |
| **between_inh_range** | (0.25, 0.40) | **(0.28, 0.45)** | Faster burst termination |
| **between_cluster_prob** | 0.25 | **0.15** | Less recurrent excitation |
| **max_connection_distance** | 5.0 | **8.0** | Wider reach for uniform burst recruitment |
| **space_size** | 12 | **15** | Room for 20 clusters |
| **burst_interval** | 6000 ms | **7000 ms** | ~7s between bursts |
| **burst_interval_jitter** | 0 | **±1500 ms** | Natural timing variation |
| **cluster_fraction** | 0.8 | **0.7** | Variable burst participation |
| **neurons_per_cluster** | 5 | **6** | Slightly more seeds |
| **stim_amplitude_range** | (2.0, 3.5) | **(2.0, 3.5)** | Keep — near-critical network is responsive |
| **stim_duration_range** | (5, 15) | **(5, 20)** | Slightly longer |
| **neurons_per_cluster_range** | (15, 20) | **(12, 18)** | ~300 neurons default |

---

## SCALING GUIDE

To change neuron count, adjust ONLY `neurons_per_cluster_range`. Everything else auto-scales:

| Target neurons | neurons_per_cluster_range | Clusters | Notes |
|---------------|--------------------------|----------|-------|
| ~200 | (8, 12) | 20 | Small network, fast simulation |
| **~300** | **(12, 18)** | **20** | **Default** |
| ~500 | (22, 28) | 20 | Medium network |
| ~700 | (30, 40) | 20 | Matches reference figure neuron count |
| ~1000 | (45, 55) | 20 | Large network, slow in pure Python |

When scaling up, you may need to:
- Slightly **reduce** `within_exc_range` (larger clusters have more recurrent connections → easier to auto-burst)
- Slightly **increase** `stim_amplitude_range` (more neurons to recruit)
- The `neurons_per_cluster` for stimulation should scale: use `max(5, cluster_size // 3)`

---

## PERFORMANCE TIP

Use pre-computed synapse lookup dict instead of iterating all synapses per spike:
```python
synapse_lookup = {}
for syn in synapses:
    synapse_lookup.setdefault(syn.pre_neuron_id, []).append(syn)
```

---

## CONFIG (config.yaml)

```yaml
network:
  num_clusters: 20
  neurons_per_cluster_range: [12, 18]  # ~300 neurons. Scale this for more/fewer.
  inhibitory_probability: 0.2
  cluster_radius: 1.0
  space_size: 15
  within_cluster_prob: 0.50
  between_cluster_prob: 0.15
  max_connection_distance: 8.0
  distance_decay_sigma: 1.0

neuron:
  v_rest: -62.0
  v_thresh: -50.0
  v_reset: -70.0
  v_floor: -80.0
  R_m: 100.0
  tau_ref: 2.0
  noise_sigma: 1.0
  exc:
    tau_m: 20.0
    adaptation_increment: 0.04
    tau_adaptation: 400.0
  inh:
    tau_m: 10.0
    adaptation_increment: 0.01
    tau_adaptation: 100.0

weights:
  within_exc_range: [0.25, 0.45]
  between_exc_range: [0.20, 0.35]
  within_inh_range: [0.35, 0.55]
  between_inh_range: [0.28, 0.45]
  use_lognormal: true
  lognormal_sigma: 0.5

synapse:
  exc_tau: 3.0
  inh_tau: 10.0
  delay: 1.0

stimulation:
  burst_interval: 7000
  burst_interval_jitter: 1500
  cluster_fraction: 0.70
  neurons_per_cluster: 6
  stim_amplitude_range: [2.0, 3.5]
  stim_duration_range: [5, 20]
  per_neuron_jitter: 1.0  # ±ms

simulation:
  dt: 0.1
  duration: 60000
  record_voltage: true
  voltage_sample_rate: 1.0
  n_recordings: 1

analysis:
  resampling_frequency: 10  # Hz — matches reference recording sampling rate
  burst_detection_threshold_factor: 5.0

output:
  save_dir: "LIF_data"
```

---

## ITERATIVE TUNING

### Step 1: Near-criticality test
Run 10s with NO stimulation. Must see scattered spikes but NO synchronized bursts.
- ❌ Auto-bursting → reduce `within_exc_range` and `between_exc_range`
- ❌ Completely silent → increase `noise_sigma` or `v_rest`

### Step 2: Stimulus-driven bursts
Run 20s with stimulation.
- ❌ No bursts → increase `stim_amplitude_range` (try 5-8 nA), or `neurons_per_cluster` to 10-12
- ❌ Bursts only in some clusters → increase `between_cluster_prob` to 0.20, or `max_connection_distance` to 10

### Step 3: Burst termination
- ❌ Bursts last > 1s → increase `adaptation_increment` (try 0.08 exc), increase `within_inh_range`

### Step 4: Inter-burst activity level
- ❌ Too much activity between bursts → reduce `noise_sigma` to 0.7, reduce `v_rest` to -64
- ❌ Completely silent between bursts → increase `noise_sigma` to 1.2, increase `v_rest` to -60

### Step 5: Timing
- ❌ Too periodic → increase `burst_interval_jitter` to ±2000ms
- ❌ Too many bursts → increase `burst_interval` to 8000-10000ms

---

## QUICK TROUBLESHOOTING

| Symptom | Fix |
|---------|-----|
| Self-sustaining bursts (auto-bursting without stimulus) | ↓↓ within_exc_range, ↓↓ between_exc_range, ↑ adaptation_increment |
| Too much inter-burst activity | ↓ noise_sigma, ↓ v_rest |
| No bursts at all | ↑ stim_amplitude, ↑ neurons_per_cluster, ↑ stim_duration |
| Bursts only in some clusters | ↑ between_cluster_prob, ↑ max_connection_distance, ↑ cluster_fraction |
| Bursts don't stop | ↑ adaptation_increment, ↑ inh weights |
| Dead silent between bursts | ↑ noise_sigma, ↑ v_rest |
| Too periodic | ↑ burst_interval_jitter |