# LIF Network Simulation — Code Analysis

## File

`LIF_network_simulation.ipynb` — Jupyter notebook, 29 cells (Parts 0–11).

## Purpose

Simulates a spatially-embedded, clustered spiking neural network using Leaky Integrate-and-Fire (LIF) neurons with exponential current-based synapses. The network is self-sustaining: a single initial current pulse at t=0 kicks off activity, and recurrent connections maintain it with no background input.

---

## Architecture

### Neuron Model (`LIFNeuron`, cell 4)

Current-based LIF with Euler-Maruyama integration and Spike-Frequency Adaptation (SFA).

- Excitatory: `tau_m = 20 ms` (pyramidal cell dynamics)
- Inhibitory: `tau_m = 10 ms` (fast-spiking interneuron dynamics)
- `v_rest = -62 mV`, `v_thresh = -50 mV` (12 mV gap)
- `v_reset = -70 mV`, `v_floor = -80 mV`
- `R_m = 100 MOhm`, `tau_ref = 2 ms`, `noise_sigma = 1.5 mV`
- Noise scaled by `sqrt(dt)` for correct stochastic integration

**Spike-Frequency Adaptation (SFA):**
- `tau_adapt = 2000 ms` (slow decay for inter-burst intervals)
- `adapt_increment = 0.008 nA` (hyperpolarizing current added per spike)
- Models Ca²⁺-activated K⁺ channels (SK channels) for burst termination

Update equation: `dV = ((v_rest - V) + R_m * (I_syn + I_ext - I_adapt)) * dt / tau_m + noise * sqrt(dt)`

### Synapse Model (`ExpSynapse`, cell 5)

Exponential current-based synapses with axonal delay.

- AMPA (excitatory): `tau = 3 ms`
- GABA (inhibitory): `tau = 10 ms`
- Default axonal delay: `1 ms`
- Spike queue for delayed delivery
- Decay: `i_syn *= exp(-dt / tau)` each step

### Network Topology (`create_clustered_network`, cell 9)

Spatially-embedded clustered network built by:

1. Placing cluster centers in 2D space with non-overlap constraint
2. Distributing neurons randomly within each cluster radius
3. Connecting neurons with distance-dependent probability (Gaussian decay)

Parameters:
- Target: 20 clusters, 15–20 neurons each (~300–400 total)
- 20% inhibitory probability per neuron
- Within-cluster connection probability: 0.5
- Between-cluster connection probability: 0.1
- Max connection distance: 5.0 units

### Synaptic Weight Ranges (`NetworkWeightParameters`, cell 6)

All weights drawn from lognormal distributions (sigma=0.5), clipped to range.

| Connection Type         | Weight Range (nA) |
|------------------------|-------------------|
| Within-cluster exc     | 0.50 – 0.80      |
| Between-cluster exc    | 0.30 – 0.50      |
| Within-cluster inh     | 0.35 – 0.55      |
| Between-cluster inh    | 0.20 – 0.35      |

### Stimulation (`create_initial_stimulation`, cell 11)

Single pulse at t=0 to 30% of neurons (randomly selected across all clusters).
- Amplitude: 0.8–1.2 nA
- Duration: 50–100 ms
- No ongoing background input after initial pulse

### Simulation Engine (`simulate_network`, cell 13)

Main loop: Euler integration at dt=0.1 ms for configurable duration (default 60s).

Loop order per timestep:
1. Apply/expire stimulation currents
2. Reset all `i_syn` to zero
3. Update all synapses (accumulate currents onto postsynaptic neurons)
4. Update all neurons (integrate voltage, detect spikes)
5. Propagate spikes to outgoing synapses
6. Optionally record voltage traces

### Data Output

Each recording saves two representations:
- **Full-resolution spike times**: per-neuron lists of spike times in ms
- **Resampled binary matrix**: spikes binned at 20 Hz (50 ms bins), shape (n_neurons, n_time_bins)
- **Voltage traces** (optional): sampled at 1 ms, shape (n_neurons, n_time_points)

Session structure on disk:
```
LIF data/
  <timestamp>/
    network_<timestamp>.npz    # topology, positions, weights
    recording000.npz           # spike + voltage data
    recording001.npz
    session_metadata.json      # run parameters
```

---

## Key Functions

| Function | Cell | Purpose |
|----------|------|---------|
| `LIFNeuron.__init__` | 4 | Initialize neuron with E/I-specific parameters |
| `LIFNeuron.update` | 4 | Single timestep: integrate voltage, check spike |
| `ExpSynapse.receive_spike` | 5 | Queue spike with axonal delay |
| `ExpSynapse.update` | 5 | Deliver queued spikes, decay current |
| `create_clustered_network` | 9 | Build full network topology |
| `create_initial_stimulation` | 11 | Generate t=0 current pulse events |
| `simulate_network` | 13 | Run the simulation loop |
| `sequential_simulation_individual_saves` | 19 | Orchestrate multiple recordings |
| `save_recording_data` | 17 | Save spike, voltage, and resampled data |
| `load_session_recordings` | 21 | Load all recordings from a session |
| `analyze_spike_trains` | 28 | Compute firing rates, ISI, CV, population rate |
| `plot_raster` | 23 | Raster plot of spike data |
| `plot_voltage_traces` | 23 | Individual neuron voltage traces |
| `plot_voltage_heatmap` | 23 | Population voltage heatmap |
| `plot_network_positions` | 23 | Spatial layout with connections |

---

## Known Issues

### Performance: O(N*S) spike delivery (cell 13)

When a neuron spikes, all synapses are iterated to find outgoing connections:
```python
for syn in synapses:
    if syn.pre_neuron_id == neuron.neuron_id:
        syn.receive_spike(t)
```
A pre-indexed dictionary (`pre_id -> [synapse_list]`) would reduce this from O(S) to O(K) where K is the neuron's out-degree.

### Performance: O(N^2) connection creation (cell 9)

All neuron pairs are checked. Works for ~200 neurons but does not scale. A KD-tree or grid-based spatial index would help for larger networks.

### Performance: list `pop(0)` in synapse queue (cell 5)

`pending_spikes.pop(0)` is O(N) on a Python list. Using `collections.deque` would make it O(1).

### Resampling window too wide (cell 15)

The resampling function marks bins at `[time_idx-1, time_idx, time_idx+1]` for each spike, spanning up to 3 bins (150 ms at 20 Hz). This inflates spike counts in the resampled binary matrix.

---

## Simulation Output (from most recent run)

Configuration: 218 neurons, 13 clusters (of 20 target), 1430 connections, space_size=8, dt=0.1 ms, 60s duration, 2 recordings.

| Metric | Recording 1 | Recording 2 |
|--------|------------|------------|
| Mean firing rate | 7.73 Hz | 7.98 Hz |
| Std firing rate | 11.08 Hz | 11.97 Hz |
| Max firing rate | 68.18 Hz | 73.40 Hz |
| Total spikes | 101,128 | 104,342 |
| Active neurons | 218 / 218 | 218 / 218 |
| Exc / Inh neurons | 177 / 41 | 177 / 41 |
| Exc / Inh connections | 1149 / 281 | 1149 / 281 |

Assessment:
- Mean ~8 Hz is biologically plausible for cortical networks
- High std and max (~70 Hz) indicate skewed distribution with some hyperactive neurons
- All neurons remain active for full 60s, confirming self-sustaining design works
- Only 13/20 clusters placed due to tight space_size=8
- E/I weight ratio ~3:1, within-cluster connections dominate (1279 vs 151 between)
