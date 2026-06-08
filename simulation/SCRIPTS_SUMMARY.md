# Scripts Summary — Supported LIF Simulation + Learned-LIF Inference

This document tracks the supported top-level workflow only.

Out-of-scope GNN, classical baseline, presentation, scratch, and legacy helper materials have been removed from this cleaned repo and are intentionally excluded from the active workflow below.

## Supported Root Surface

```
lif_simulation/                                    Shared simulation package
lif_inference/                                     Shared learned-LIF inference package
scripts/                                           CLI helpers and analysis utilities
LIF_network_simulation_network_burst_conductance.ipynb          Monolithic conductance reference notebook
LIF_network_simulation_network_burst_conductance_modular.ipynb  Recommended modular simulation notebook
learned_lif_connectivity_modular.ipynb                          Recommended modular learned-LIF notebook
scripts/run_conductance_simulation.py              Thin CLI for simulation runs
scripts/run_no_stim_validation.py                  Thin CLI for spontaneous-activity validation
scripts/run_h_current_sag_probe.py                 Thin CLI for h-current sag/rebound probing
scripts/plot_saved_session.py                      Thin CLI for saved-session plotting
scripts/plot_voltage_score_separation.py           Plot ground-truth edge/non-edge score separation for saved spike-only or voltage checkpoints
scripts/compare_h_current_ablation.py              Saved-session h-current comparison and burst analysis
scripts/inspect_data.py                            Raw-voltage inspection helper
scripts/run_voltage_lambda_sweep.py                Voltage-augmented learned-LIF sweep helper
README.md                                          Public workflow overview
SCRIPTS_SUMMARY.md                                 Supported-surface summary
CLAUDE.md                                          Repo-local workflow notes
docs/                                              Reference-only project notes moved out of root
```

## Execution Order

### 1. Simulate or Validate

Recommended notebook path:

- `LIF_network_simulation_network_burst_conductance_modular.ipynb`

Equivalent CLI entry points:

- `run_no_stim_validation.py`
- `run_h_current_sag_probe.py`
- `run_conductance_simulation.py`
- `plot_saved_session.py`
- `scripts/run_no_stim_validation.py`
- `scripts/run_h_current_sag_probe.py`
- `scripts/run_conductance_simulation.py`
- `scripts/plot_saved_session.py`

Reference notebook retained for parity and extraction history:

- `LIF_network_simulation_network_burst_conductance.ipynb`

### 2. Fit Learned-LIF Connectivity Models

Recommended notebook path:

- `learned_lif_connectivity_modular.ipynb`

Equivalent CLI entry points:

- `lif_inference/learned_lif_connectivity.py`
- `lif_inference/voltage_augmented_learned_lif_connectivity.py`
- `scripts/run_voltage_lambda_sweep.py`

Current inference behavior:

- Both learned-LIF CLIs support `--threshold-mode adaptive|shared`; adaptive mode uses a per-neuron baseline plus spike-triggered threshold increments, while shared mode uses one learned threshold for all neurons.
- Both learned-LIF CLIs now also support `--connectivity-threshold-mode oracle_f1|surrogate_fdr`; `oracle_f1` keeps the retrospective benchmark cutoff tied to ground-truth labels, while `surrogate_fdr` fits circular-shift null surrogates and chooses a binary edge cutoff without using ground-truth connectivity. The current default surrogate target is `FDR=0.005` unless overridden with `--surrogate-fdr`.
- The spike-only learned-LIF CLI also supports `--event-anchor-mode post|pre|both`; `post` keeps the original post-spike-centered event windows, `pre` anchors windows on candidate presynaptic spikes, and `both` concatenates the two samplers for mixed event training.
- The spike-only learned-LIF pipeline now exports connectivity figures for both oracle-F1 and surrogate-FDR thresholds from the same learned model using explicit `*_oracle_f1.png` and `*_surrogate_fdr.png` filenames so the retrospective and non-leaky edge-call views can be compared side by side.
- Both inference paths can exclude saved stimulation-onset bins; the voltage-augmented CLI now also mirrors the spike-only optional detected-burst exclusion controls for candidate scoring and event-window extraction.

Current simulation behavior:

- No-stimulation validation and `run_conductance_simulation --mode spontaneous` now force `noise_sigma = 0.0` and use frozen per-neuron excitatory baseline drive (`mean=0.11`, `sd=0.05` by default) plus recurrent excitatory scaling (`0.65` by default). The baseline is stored on `LIFNeuron.i_baseline`, remains separate from stimulation `i_ext`, persists across `reset_state()`, and is drawn once per network seed.
- Spontaneous saved recordings can include detected `burst_windows` and `interburst_windows` from population activity segmentation. These windows are downstream analysis metadata, not ground-truth structural connectivity.

### 3. Optional Saved-Session Analysis

- `scripts/compare_h_current_ablation.py` compares matched saved sessions and detects network bursts from saved spike trains.
- `scripts/inspect_data.py` inspects raw full-dt voltage traces from saved recordings, including chunked external HDF5 sidecars.
- `scripts/plot_voltage_score_separation.py` plots the log10 learned-weight distributions for true edges vs non-edges from a saved spike-only or voltage checkpoint and can overlay circular-shift surrogate null scores plus a surrogate-FDR cutoff on the same axis.

## Key Workflow Notes

- The active simulation model is conductance-based and supports an optional slow h-current.
- Clustered connectivity semantics remain: within-cluster probability exceeds between-cluster probability, connectivity is distance-limited, and hub neurons preferentially strengthen long-range inter-cluster links.
- Saved voltage traces now default to raw membrane voltage at the full simulation step, with the main sequential workflow storing those traces in chunked external HDF5 sidecars referenced by each `recordingNNN.npz` file.
- Shared simulation logic lives in `lif_simulation/`; shared learned-LIF inference logic lives in `lif_inference/`.
- All shared plotting helpers live in `lif_simulation/plotting.py` so notebook and script outputs stay aligned.

## Main Outputs

- `LIF data/<timestamp>/` — saved network structures, recordings, and session metadata.
- `learned_lif_outputs/` — spike-only learned-LIF outputs, including threshold-mode metadata, connectivity-threshold metadata, learned threshold summaries in saved checkpoints, and explicitly named `*_oracle_f1.png` plus `*_surrogate_fdr.png` connectivity figures.
- `voltage_augmented_learned_lif_outputs/` — voltage-augmented learned-LIF outputs, including threshold-mode metadata, connectivity-threshold metadata, learned threshold summaries, saved stimulation/burst exclusion metadata, and dedicated `*_score_separation.png` diagnostics for true-edge vs non-edge score separation.
- `modular_validation_outputs/` — validation artifacts from modular checks.

## Supporting Reference Material

- `docs/SCRIPT_DETAILS.md` contains the longer script-by-script explanation for the supported simulation and learned-LIF surface.
- `docs/` contains reference notes moved out of the root to keep the supported workflow surface small.
