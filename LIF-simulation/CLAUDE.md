# CLAUDE.md — LIF Simulation Workspace

## Project Goal

This repository centers on a clustered spiking neural network simulation built from Leaky Integrate-and-Fire neurons, together with the supported learned-LIF connectivity-inference workflows.

The current primary simulation surface is the conductance-based notebook with a slow h-current. It is used to generate near-critical spontaneous activity, stimulus-driven bursting, saved network structure, spike recordings, and raw full-dt membrane-voltage traces for later analysis and model fitting. The notebook now keeps h-current as an explicit toggle so the same code path can be run with or without it for ablation and validation.

## Current Main Entry Points

### Main Simulation
- `LIF_network_simulation_network_burst_conductance_modular.ipynb` — Recommended simulation notebook. Imports `lif_simulation/` instead of embedding the full pipeline inline.
- `scripts/run_conductance_simulation.py` — Thin CLI for saved simulation sessions.
- `scripts/run_no_stim_validation.py` — Thin CLI for the spontaneous near-criticality validation.
- `scripts/run_h_current_sag_probe.py` — Thin CLI for the h-current sag/rebound probe.
- `scripts/plot_saved_session.py` — Thin CLI for loading and plotting saved sessions.
- `scripts/compare_h_current_ablation.py` — Utility script for comparing saved h-current on/off sessions.
- `scripts/inspect_data.py` — Quick raw-voltage inspection helper.

### Connectivity Inference
- `lif_inference/` — Packaged learned-LIF implementation modules.
- `learned_lif_connectivity_modular.ipynb` — Recommended inference notebook. Imports `lif_inference/` directly.
- `lif_inference/learned_lif_connectivity.py` — Packaged spike-only learned-LIF CLI module.
- `lif_inference/voltage_augmented_learned_lif_connectivity.py` — Packaged voltage-augmented learned-LIF CLI module. In addition to the legacy event-window path, it now exposes opt-in per-neuron surrogate-FDR thresholding, reduced intrinsic slow states, and continuous-state chunk training.
- `scripts/run_voltage_lambda_sweep.py` — Sweep helper for voltage-augmented learned-LIF runs.

### Documentation
- `README.md` — Public repo landing-page overview.
- `SCRIPTS_SUMMARY.md` — Current file inventory and parameter reference.
- `CLAUDE.md` — Repo-local project notes and current workflow summary.
- `docs/` — Reference notes moved out of the root execution surface.

The longer supported-only per-script reference now lives in `docs/SCRIPT_DETAILS.md`.

### Local Outputs (Usually Ignored by Git)
- `LIF data/` — Saved network structures, recordings, and session metadata.
- `learned_lif_outputs/`, `voltage_augmented_learned_lif_outputs/`, `modular_validation_outputs/` — Generated analysis and model output directories.

## Current Notebook Structure

The conductance notebook is still organized by numbered parts, but the important implementation details are now:

### Core Classes
- `LIFNeuron` — Conductance-based LIF neuron with spike-frequency adaptation and an optional slow h-current.
  - Excitatory and inhibitory neurons keep separate `tau_m`, adaptation, and `g_h_max` / `tau_h` values.
  - State includes membrane voltage `v`, conductances `g_exc` and `g_inh`, external current `i_ext`, adaptation current `i_adapt`, and h-current state `h_gate` / `i_h`.
  - `set_h_current_enabled()` makes the toggle explicit; when disabled, the update path is skipped and `i_h` remains zero.
- `ExpSynapse` — Exponential conductance synapse.
  - This is a minimal conversion from the current-based notebook.
  - Stored legacy weights are converted to conductance increments through nominal driving-force constants so the old tuning stays approximately usable.
- `NetworkWeightParameters` — Weight ranges for within/between cluster excitatory and inhibitory connections.

### Network / Simulation Flow
- `create_clustered_network()` — Builds neurons, assigns clusters, designates hub neurons, and creates distance-dependent recurrent synapses. Accepts `use_h_current` so network instantiation matches the intended ablation setting.
- `create_periodic_cluster_stimulation()` — Generates burst-triggering stimulus events with burst jitter, per-cluster jitter, and per-neuron jitter.
- `simulate_network()` — Main simulation loop.
  - Uses synapse lookup for spike propagation.
  - Records spike times and raw membrane voltage traces at the full simulation step by default.
  - Does not stamp a synthetic spike waveform into the saved `voltage_traces` array.
- `sequential_simulation_individual_saves()` — Runs one network across one or more recordings, resetting neuron state between recordings and saving outputs to `LIF data/<timestamp>/`.
  - Accepts `use_h_current` and records `use_h_current`, `h_current_mode`, and raw-voltage metadata in `session_metadata.json`.

### Validation / Analysis Cells
- Zero-stimulation validation cell:
  - Exposes `use_h_current_test`
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
- Saved voltage step defaults to `dt = 0.1 ms` for raw full-dt traces
- `voltage_sample_rate = 1.0 ms` is retained only as a legacy requested value for compatibility notes in metadata

## Environment

- Python 3.9 in `.venv/` locally
- Core packages: `numpy`, `matplotlib`, `jupyter`
- Learning / prediction packages: `torch`, `scikit-learn`

## Common Tasks

### Run a Simulation
Open `LIF_network_simulation_network_burst_conductance_modular.ipynb` and execute the cells in order. Adjust execution parameters in the configuration cell, including `use_h_current` when you want an ablation run.

### Validate Spontaneous Activity
Run the zero-stimulation validation cell in the conductance notebook. The target regime is sparse spontaneous firing with no synchronized auto-bursts. Use `use_h_current_test` when you want a matched comparison with h-current disabled.

### Compare h-current Ablations
Run `python -m scripts.compare_h_current_ablation` on two or more saved session folders or `session_metadata.json` files to summarize firing-rate and population-activity differences across h-current on/off runs. The script can also detect network bursts from coarse population synchrony and save per-recording raster plots with detected burst windows shaded. For a true matched pair, reset the same random seed before each session and only change `use_h_current`.

### Probe h-current Behavior
Run the sag/rebound probe cell in the conductance notebook to compare the same neuron with and without the h-current.

### Fit Connectivity Models
- Learned-LIF: `python -m lif_inference.learned_lif_connectivity`
- Voltage-augmented learned-LIF: `python -m lif_inference.voltage_augmented_learned_lif_connectivity`
- For real-data-style non-leaky thresholding in the voltage path, prefer `--connectivity-threshold-mode surrogate_fdr` or the row-calibrated `--connectivity-threshold-mode surrogate_fdr_per_neuron`; `oracle_f1` remains a retrospective simulated-data benchmark.
- Use `--slow-state-mode adaptation_h` to add reduced adaptation and h-like intrinsic states to the inference LIF, and `--training-mode continuous_state` when you want those states carried across ordered recording chunks rather than reset for every event window.

### Archived Materials
Out-of-scope GNN, baseline, presentation, scratch, and legacy notebook files have been removed from this cleaned repo snapshot.

## Git / Sharing Notes

The public GitHub repository intentionally excludes local environments, raw simulation data, learned-model outputs, generated plots, and derivative presentation exports. Anyone cloning the repo should expect to generate those outputs locally.