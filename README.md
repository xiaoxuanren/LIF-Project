# LIF Simulation and Connectivity Inference

This repository contains a clustered spiking neural network simulation built from Leaky Integrate-and-Fire neurons, together with several downstream connectivity-inference pipelines.

The main simulation surface is a conductance-based LIF notebook with a slow hyperpolarization-activated h-current. It is used to generate near-critical spontaneous activity, stimulus-driven bursting, saved network structure, spike recordings, and voltage traces that feed later analysis and prediction scripts.

## Main Entry Points

- `LIF_network_simulation_network_burst_conductance.ipynb` — Main notebook for simulation, validation, saving, loading, and visualization.
- `process_existing_for_gnn.ipynb` — Builds combined recordings and `session_gnn_metadata.json` from saved simulation sessions.
- `learned_lif_connectivity.py` — Differentiable LIF connectivity inference from spike trains.
- `voltage_augmented_learned_lif_connectivity.py` — Learned-LIF model with masked subthreshold voltage supervision.
- `gnn_cross_network_prediction.py` — Cross-network GraphSAGE training pipeline.
- `predict_new_network.py` — Runs a trained GNN on new sessions.
- `create_simulation_learned_lif_presentation.py` — Builds the current presentation deck from tracked assets and local outputs.

## What Is in Scope

- Clustered recurrent LIF network simulation with excitatory, inhibitory, and hub neurons
- Conductance-based recurrent synapses with legacy weight scaling
- Slow h-current for sag, rebound, and richer subthreshold dynamics
- No-stimulation validation for near-critical spontaneous firing
- Connectivity inference baselines: perceptron, spike-train CNN/LSTM, correlation, learned-LIF, voltage-augmented learned-LIF, and cross-network GNN

## Quick Start

### 1. Create an Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib jupyter pillow python-pptx scikit-learn torch
```

Install a matching `torch-geometric` build separately for your PyTorch and CUDA setup if you plan to run the GNN pipeline.

### 2. Run the Main Simulation

Open and execute:

```text
LIF_network_simulation_network_burst_conductance.ipynb
```

The notebook includes:

- the main sequential simulation run cell
- a zero-stimulation validation cell
- a sag/rebound h-current probe
- plotting utilities for raster, voltage traces, and heatmaps

### 3. Prepare Sessions for Prediction

After simulation output exists under `LIF data/`, run:

```text
process_existing_for_gnn.ipynb
```

This creates combined recordings and `session_gnn_metadata.json` files used by later scripts.

### 4. Run Inference

Example learned-LIF run:

```powershell
python learned_lif_connectivity.py --session "LIF data/<timestamp>" --k 50 --epochs 40 --batch 128 --max-delay 8
```

Example GNN prediction on a processed session:

```powershell
python predict_new_network.py "LIF data/<timestamp>/session_gnn_metadata.json"
```

## Repository Layout

- `LIF_network_simulation_network_burst_conductance.ipynb` — Main conductance + h-current simulation notebook
- `LIF_network_simulation_network_burst.ipynb` — Legacy current-based reference notebook
- `LIF_network_simulation_network_burst_voltage_traces.ipynb` — Legacy raw-voltage variant
- `plot_voltage_traces.ipynb` and `raw_subthreshold_voltage_saving_patch.ipynb` — Auxiliary / legacy helper notebooks
- `presentation_assets/` — Tracked source assets used by the presentation script
- `SCRIPTS_SUMMARY.md` — Current file inventory and parameter summary
- `CLAUDE.md` — Repo-local project notes and current workflow summary

## Data and Outputs Not Stored in Git

The public repo intentionally does not include large local artifacts such as:

- `LIF data/`
- `.venv/`
- learned model output directories
- generated plots and exported presentation files
- derivative slide-export image folders

If you clone this repository, expect to generate simulation data and trained-model outputs locally.

## Current Status

The conductance notebook is the current source of truth for simulation work. The legacy current-based and raw-voltage variants are kept for comparison and migration history, but new changes should generally start from `LIF_network_simulation_network_burst_conductance.ipynb`.