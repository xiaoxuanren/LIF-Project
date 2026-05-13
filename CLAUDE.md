# CLAUDE.md — LIF Simulation Workspace

## Project Goal

This repository centers on a clustered spiking neural network simulation built from Leaky Integrate-and-Fire neurons, together with downstream connectivity-inference pipelines.

The current primary simulation surface is the conductance-based notebook with a slow h-current. It is used to generate near-critical spontaneous activity, stimulus-driven bursting, saved network structure, spike recordings, and voltage traces for later analysis and model fitting.

## Current Main Entry Points

### Main Simulation
- `LIF_network_simulation_network_burst_conductance.ipynb` — Main notebook. Conductance-based recurrent synapses with legacy weight scaling, slow h-current, no-stimulation validation cell, sag/rebound probe, saving/loading, and visualization.
- `LIF_network_simulation_network_burst.ipynb` — Legacy current-based reference notebook retained for comparison.
- `LIF_network_simulation_network_burst_voltage_traces.ipynb` — Legacy raw-voltage variant retained for reference.
- `process_existing_for_gnn.ipynb` — Converts saved simulation sessions into `session_gnn_metadata.json` and combined recordings for downstream inference.

### Connectivity Inference
- `learned_lif_connectivity.py` — Differentiable LIF connectivity inference from spike trains.
- `voltage_augmented_learned_lif_connectivity.py` — Learned-LIF model with masked subthreshold voltage supervision.
- `perceptron_connectivity.py` — Classical perceptron baseline.
- `spike_train_connectivity.py` — CNN/LSTM/perceptron baselines on raw spike trains.
- `correlation_connectivity.py` — Correlation-based connectivity baseline and visual summaries.
- `gnn_cross_network_prediction.py` — Cross-network GraphSAGE training pipeline.
- `predict_new_network.py` — Loads a trained GNN checkpoint and predicts connectivity on new sessions.

### Documentation / Presentation
- `README.md` — Public repo landing-page overview.
- `SCRIPTS_SUMMARY.md` — Current file inventory and parameter reference.
- `create_simulation_learned_lif_presentation.py` — Builds the learned-LIF presentation deck from tracked assets plus local outputs.
- `presentation_oral_script.txt` — Oral presentation notes.

### Local Outputs (Usually Ignored by Git)
- `LIF data/` — Saved network structures, recordings, and session metadata.
- `gnn_outputs/`, `learned_lif_outputs/`, `correlation_outputs/`, `perceptron_outputs/`, `voltage_augmented_learned_lif_outputs/`, `voltage_connectivity_outputs/` — Generated analysis and model output directories.

## Current Notebook Structure

The conductance notebook is still organized by numbered parts, but the important implementation details are now:

### Core Classes
- `LIFNeuron` — Conductance-based LIF neuron with spike-frequency adaptation and a slow h-current.
  - Excitatory and inhibitory neurons keep separate `tau_m`, adaptation, and `g_h_max` / `tau_h` values.
  - State includes membrane voltage `v`, conductances `g_exc` and `g_inh`, external current `i_ext`, adaptation current `i_adapt`, and h-current state `h_gate` / `i_h`.
- `ExpSynapse` — Exponential conductance synapse.
  - This is a minimal conversion from the current-based notebook.
  - Stored legacy weights are converted to conductance increments through nominal driving-force constants so the old tuning stays approximately usable.
- `NetworkWeightParameters` — Weight ranges for within/between cluster excitatory and inhibitory connections.

### Network / Simulation Flow
- `create_clustered_network()` — Builds neurons, assigns clusters, designates hub neurons, and creates distance-dependent recurrent synapses.
- `create_periodic_cluster_stimulation()` — Generates burst-triggering stimulus events with burst jitter, per-cluster jitter, and per-neuron jitter.
- `simulate_network()` — Main simulation loop.
  - Uses synapse lookup for spike propagation.
  - Records spike times and sampled voltage traces.
  - Overlays a short spike waveform only in the recorded visualization traces so plots look more biophysical while the model itself remains LIF.
- `sequential_simulation_individual_saves()` — Runs one network across one or more recordings, resetting neuron state between recordings and saving outputs to `LIF data/<timestamp>/`.

### Validation / Analysis Cells
- Zero-stimulation validation cell:
  - Sets `stimulation_events_test = []`
  - Plots network structure, raster, and example voltage traces
  - Reports firing-rate and active-fraction statistics to assess near-critical spontaneous activity
- h-current sag test:
  - Runs the same neuron with and without `g_h_max`
  - Quantifies sag depth and rebound from a hyperpolarizing step

## Current Default Operating Point

These reflect the current conductance notebook defaults and supporting summary docs:

- `num_clusters = 20`
- `neurons_per_cluster_range = (12, 18)`
- `inhibitory_probability = 0.2`
- `within_cluster_prob = 0.5`
- `between_cluster_prob = 0.15`
- `max_connection_distance = 8.0`
- `space_size = 15`
- `burst_interval = 7000 ms`
- `cluster_fraction = 0.7`
- `neurons_per_cluster = 6`
- `stim_amplitude_range = (2.0, 3.5)`
- `stim_duration_range = (10, 30)`
- `target_freq = 10 Hz`
- `dt = 0.1 ms`
- `record_voltage = True`
- `voltage_sample_rate = 1.0 ms`

## Environment

- Python 3.9 in `.venv/` locally
- Core packages: `numpy`, `matplotlib`, `jupyter`, `Pillow`, `python-pptx`
- Learning / prediction packages: `torch`, `torch_geometric`, `scikit-learn`

## Common Tasks

### Run a Simulation
Open `LIF_network_simulation_network_burst_conductance.ipynb` and execute the cells in order. Adjust execution parameters in the main run cell near the end of the notebook.

### Validate Spontaneous Activity
Run the zero-stimulation validation cell in the conductance notebook. The target regime is sparse spontaneous firing with no synchronized auto-bursts.

### Probe h-current Behavior
Run the sag/rebound probe cell in the conductance notebook to compare the same neuron with and without the h-current.

### Prepare Data for Inference
Run `process_existing_for_gnn.ipynb` after simulation output exists under `LIF data/`.

### Fit Connectivity Models
- Learned-LIF: `learned_lif_connectivity.py`
- Voltage-augmented learned-LIF: `voltage_augmented_learned_lif_connectivity.py`
- Cross-network GNN: `gnn_cross_network_prediction.py`
- Predict with trained GNN: `predict_new_network.py`

### Build the Presentation Deck
Run `create_simulation_learned_lif_presentation.py` after the expected local result files and figure assets exist.

## Git / Sharing Notes

The public GitHub repository intentionally excludes local environments, raw simulation data, learned-model outputs, generated plots, and derivative presentation exports. Anyone cloning the repo should expect to generate those outputs locally.