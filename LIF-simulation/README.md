# LIF Simulation and Learned-LIF Connectivity Inference

This repository contains a clustered spiking neural network simulation built from Leaky Integrate-and-Fire neurons, together with the spike-only and voltage-augmented learned-LIF connectivity inference workflows.

The main simulation surface is a conductance-based LIF notebook with a slow hyperpolarization-activated h-current. It is used to generate near-critical spontaneous activity, stimulus-driven bursting, saved network structure, spike recordings, and voltage traces that feed later analysis and prediction scripts. Saved voltage traces now default to raw full-dt membrane voltage written into chunked external HDF5 sidecars, so the main simulation can keep full-dt traces without building the entire voltage matrix in RAM first. The h-current can be enabled or disabled from the notebook workflow for ablation and validation comparisons.

## Main Entry Points

- `lif_simulation/` — Shared simulation package extracted from the conductance notebook. Contains the model, network creation, stimulation, simulation loop, save/load helpers, analysis, hub validation, and plotting code.
- `lif_inference/` — Shared inference package for the spike-only and voltage-augmented learned-LIF connectivity pipelines.
- `LIF_network_simulation_network_burst_conductance_modular.ipynb` — Import-based orchestration notebook for the modular package.
- `learned_lif_connectivity_modular.ipynb` — Import-based orchestration notebook for the learned-LIF inference package.
- `scripts/` — CLI helpers for validation, saved simulation runs, saved-session plotting, ablation comparison, raw-voltage inspection, and voltage-lambda sweeps.
- `LIF_network_simulation_network_burst_conductance.ipynb` — Main notebook for simulation, validation, saving, loading, and visualization.
- `docs/` — Reference notes moved out of the root workflow surface.

For a longer supported-only script reference, see `docs/SCRIPT_DETAILS.md`.

## What Is in Scope

- Clustered recurrent LIF network simulation with excitatory, inhibitory, and hub neurons
- Conductance-based recurrent synapses with legacy weight scaling
- Slow h-current for sag, rebound, and richer subthreshold dynamics
- No-stimulation validation for near-critical spontaneous firing
- Learned-LIF connectivity inference from spikes
- Voltage-augmented learned-LIF connectivity inference with masked subthreshold voltage supervision

## First-Time Path

If you are opening this repository for the first time, use this order:

1. Set up the environment and confirm the package imports work.
2. Open `LIF_network_simulation_network_burst_conductance_modular.ipynb`.
3. Run the validation and h-current probe cells before running a long saved simulation.
4. Enable `execute_main_simulation = True` only when the validation outputs look correct.
5. After a session is saved under `LIF data/<timestamp>`, inspect it with `python -m scripts.plot_saved_session latest` or the later notebook cells.
6. Open `learned_lif_connectivity_modular.ipynb` only after you have a saved session to train on.

Use the monolithic `LIF_network_simulation_network_burst_conductance.ipynb` notebook only when you need the original all-in-one reference workflow for parity checks or extraction history. For new runs, prefer the modular notebook plus the shared `lif_simulation/` package.

## Quick Start

### 1. Create an Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib jupyter scikit-learn torch
pip install h5py
```

### 2. Run the Main Simulation

Recommended modular workflow:

```text
LIF_network_simulation_network_burst_conductance_modular.ipynb
```

or run the thin scripts directly:

```powershell
python -m scripts.run_no_stim_validation
python -m scripts.run_h_current_sag_probe
python -m scripts.run_conductance_simulation --mode spontaneous
python -m scripts.plot_saved_session latest
```

For a first run in the modular notebook:

- Leave `execute_main_simulation = False` and run the import, configuration, no-stimulation validation, and sag-probe cells first.
- Review the printed `verdict`, raster, and voltage plots from the validation cell.
- When those look reasonable, set `execute_main_simulation = True` and rerun the main simulation cell.
- Keep `burst_interval` very large together with `cluster_fraction = 0.0` and `neurons_per_cluster = 0` for spontaneous runs.
- Use finite `burst_interval` plus nonzero cluster stimulation settings for stimulus-driven burst experiments.

The original extraction-reference notebook remains available as:

```text
LIF_network_simulation_network_burst_conductance.ipynb
```

The modular notebook and thin scripts call shared plotting code in `lif_simulation/plotting.py`, so raster plots, voltage traces, heatmaps, network layouts, resampled rasters, and hub-analysis figures all come from a single implementation path.

The notebook workflow includes:

- the main sequential simulation run cell
- a zero-stimulation validation cell
- a sag/rebound h-current probe
- notebook-level `use_h_current` controls for saved runs and zero-stimulation validation
- plotting utilities for raster, voltage traces, and heatmaps

### 3. Run Learned-LIF Inference

Recommended learned-LIF workflow:

```text
learned_lif_connectivity_modular.ipynb
```

The notebook imports `lif_inference/` directly and exposes gated cells for both the spike-only and voltage-augmented learned-LIF pipelines.

For a first inference run:

- Set `session_source` to `latest` or to a specific saved session folder under `LIF data/`.
- Leave both execution toggles off while you inspect the resolved session path.
- Turn on `execute_spike_only` first if you want the simpler spike-only baseline.
- Turn on `execute_voltage_augmented` only when the saved session includes voltage traces; the current conductance simulation pipeline writes raw full-dt voltage by default and stores it in chunked external HDF5 sidecars referenced by each recording file.
- Keep `use_all_recordings = True` unless you are debugging on a small subset.

Equivalent CLI entry points remain available through thin wrappers:

```powershell
python -m lif_inference.learned_lif_connectivity --session "LIF data/<timestamp>" --k 100 --epochs 40 --batch 128 --max-delay 8 --exclude-detected-bursts
python -m lif_inference.voltage_augmented_learned_lif_connectivity --session "LIF data/<timestamp>" --k 50 --epochs 40 --batch 128 --max-delay 8
python -m lif_inference.voltage_augmented_learned_lif_connectivity --session "LIF data/<timestamp>" --training-mode continuous_state --slow-state-mode adaptation_h --connectivity-threshold-mode surrogate_fdr_per_neuron
python -m scripts.run_voltage_lambda_sweep --session "LIF data/<timestamp>" --lambdas 0.25,0.5,1.0,2.0
```

The voltage-augmented learned-LIF CLI and the voltage-lambda sweep now infer `dt` from `session_metadata.json` when `--dt` is omitted, so full-dt voltage sessions no longer require a manual override in the common case. When you explicitly pass a coarser `dt` that is an integer multiple of the saved voltage sample rate, the loader now keeps spike masking at the native voltage resolution and then downsamples the cleaned voltage targets into the requested bins.

The voltage-augmented defaults now use `mask_pre_ms=0.0`, `mask_post_ms=2.0`, and `warmup=100`: the pre-spike depolarization ramp remains supervised, while the spike bin and brief post-spike reset region stay masked. The default training mode remains `event_window` for parity with earlier runs, but `--training-mode continuous_state` carries membrane, adaptive-threshold, and optional slow-state values through ordered recording chunks. `--slow-state-mode adaptation|h|adaptation_h` adds reduced intrinsic slow states to the inference model, and `--connectivity-threshold-mode surrogate_fdr_per_neuron` calibrates non-leaky row-specific edge cutoffs for real-data-style thresholding without ground-truth labels.

The package-level import surface is:

```python
from lif_inference import run_learned_lif_pipeline, run_voltage_augmented_pipeline
```

Out-of-scope GNN, baseline, presentation, scratch, and legacy helper files have been removed from this cleaned repository snapshot so the supported root workflow stays focused.

Reference-only markdown notes have been moved under `docs/` so the repository root stays centered on runnable workflows.

## Repository Layout

- `lif_simulation/` — Modular simulation package extracted from the conductance notebook
- `lif_inference/` — Modular learned-LIF inference package
- `scripts/` — CLI helpers and utilities for simulation validation, saved-session analysis, raw-voltage inspection, ablation comparison, and voltage-lambda sweeps
- `LIF_network_simulation_network_burst_conductance_modular.ipynb` — Import-based modular notebook entry point
- `learned_lif_connectivity_modular.ipynb` — Import-based learned-LIF inference notebook
- `scripts/run_no_stim_validation.py` — Standalone near-criticality validation
- `scripts/run_h_current_sag_probe.py` — Standalone h-current sag/rebound probe
- `scripts/run_conductance_simulation.py` — Standalone conductance simulation runner
- `scripts/plot_saved_session.py` — Standalone saved-session plotting and analysis
- `scripts/compare_h_current_ablation.py` — Saved-session h-current comparison and burst summary
- `scripts/inspect_data.py` — Saved raw-voltage inspection helper for inline or external-HDF5 recordings
- `scripts/run_voltage_lambda_sweep.py` — Sweep script for the packaged voltage-augmented learned-LIF pipeline
- `lif_inference/learned_lif_connectivity.py` — Packaged spike-only learned-LIF CLI module
- `lif_inference/voltage_augmented_learned_lif_connectivity.py` — Packaged voltage-augmented learned-LIF CLI module
- `docs/` — Reference notes and project summaries not needed for day-to-day execution
- `LIF_network_simulation_network_burst_conductance.ipynb` — Main conductance + optional h-current simulation notebook
- `SCRIPTS_SUMMARY.md` — Current file inventory and parameter summary
- `CLAUDE.md` — Repo-local project notes and current workflow summary

If you want the old style of longer per-script descriptions without re-expanding the root summary, use `docs/SCRIPT_DETAILS.md`.

## Data and Outputs Not Stored in Git

The public repo intentionally does not include large local artifacts such as:

- `LIF data/`
- `.venv/`
- learned model output directories
- generated plots and analysis artifacts

If you clone this repository, expect to generate simulation data and trained-model outputs locally.

## Current Status

The supported root workflow is now intentionally limited to the conductance-based simulation surface plus the learned-LIF inference surface. The original conductance notebook is retained as the extraction/parity reference, while the modular simulation and learned-LIF notebooks are the cleaner orchestration paths for new runs.

Inside `lif_inference/`, the stable public entry points remain `learned_lif_connectivity.py` and `voltage_augmented_learned_lif_connectivity.py`, while shared internals are now split across focused helper modules for candidate selection, burst exclusion, event windows, event-window training, surrogate thresholding, and voltage-side orchestration.