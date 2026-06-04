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
scripts/plot_voltage_validation_predictions.py     Rebuild a saved voltage checkpoint's held-out validation set and plot inferred vs actual post voltage
scripts/compare_h_current_ablation.py              Saved-session h-current comparison and burst analysis
scripts/inspect_data.py                            Raw-voltage inspection helper
scripts/analyze_burst_voltage.py                   Burst-aligned raw and spike-masked voltage analysis for saved sessions
scripts/analyze_connected_pair_psp.py              Connected-pair presynaptic-spike-triggered voltage analysis
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
- Both learned-LIF CLIs support `--connectivity-threshold-mode oracle_f1|surrogate_fdr`; `oracle_f1` keeps the retrospective benchmark cutoff tied to ground-truth labels, while `surrogate_fdr` fits circular-shift null surrogates and chooses a binary edge cutoff without using ground-truth connectivity. The voltage-augmented CLI additionally supports `surrogate_fdr_per_neuron`, which calibrates a separate non-leaky surrogate cutoff for each postsynaptic neuron so high-null or high-rate rows do not dominate one global FP budget. The current default surrogate target is `FDR=0.005` unless overridden with `--surrogate-fdr`.
- The public learned-LIF entry points remain `lif_inference/learned_lif_connectivity.py` and `lif_inference/voltage_augmented_learned_lif_connectivity.py`, while shared internals are now split across focused helper modules for candidate selection, burst exclusion, event windows, event-window training, surrogate thresholding, and voltage-side orchestration.
- The spike-only learned-LIF CLI now defaults to `K=100` and detected-burst exclusion on, so standard all-recordings runs start from the higher-coverage burst-excluded configuration; use `--include-detected-bursts` if you want the older burst-included behavior.
- The spike-only learned-LIF CLI also supports `--event-anchor-mode post|pre|both`; `post` keeps the original post-spike-centered event windows, `pre` anchors windows on candidate presynaptic spikes, and `both` concatenates the two samplers for mixed event training.
- The spike-only learned-LIF pipeline now exports connectivity figures for both oracle-F1 and surrogate-FDR thresholds from the same learned model using explicit `*_oracle_f1.png` and `*_surrogate_fdr.png` filenames so the retrospective and non-leaky edge-call views can be compared side by side.
- Both inference paths can exclude saved stimulation-onset bins; the voltage-augmented CLI still exposes optional detected-burst exclusion controls for candidate scoring and event-window extraction, while the spike-only CLI now enables detected-burst exclusion by default.
- `lif_inference/voltage_augmented_learned_lif_connectivity.py` and `scripts/run_voltage_lambda_sweep.py` now infer `dt` from `session_metadata.json` when `--dt` is omitted, and they also accept explicit coarser `dt` values that are integer multiples of the saved voltage sample rate by masking at native resolution and downsampling the cleaned voltage targets afterward.
- The voltage-augmented defaults now set `mask_pre_ms=0.0` (down from `1.0`) so the pre-spike depolarization ramp toward threshold is kept as a supervised voltage target, while the spike bin itself and `mask_post_ms=2.0` of post-spike reset stay masked; this restores the spike-timing signal that the earlier pre-spike masking removed.
- The voltage-augmented event-window `warmup` default is now `100` bins (up from `30`) so each window simulates a longer causal lead-in, giving slow membrane and adaptation state more time to settle before the scored region; `pre_context=50` and `post_context=10` are unchanged.
- The voltage-augmented inference LIF now exposes optional intrinsic slow states through `--slow-state-mode none|adaptation|h|adaptation_h`. These are reduced inference states, not a full ion-channel model: `adaptation` adds a spike-triggered slow outward current, `h` adds a reduced h-like inward state activated by low voltage, and `adaptation_h` enables both. The default remains `none` for checkpoint parity.
- The voltage-augmented CLI now supports `--training-mode event_window|continuous_state`. `event_window` preserves the legacy shuffled window workflow. `continuous_state` processes ordered all-neuron chunks, carries membrane/adaptive-threshold/slow-state values across chunks, detaches state between chunks for truncated BPTT, and resets only at train/validation segment boundaries; `--continuous-chunk-len` controls the chunk length.
- Voltage checkpoints now also persist the training-mode settings used for fitting, including event-window reconstruction fields or continuous-state chunk length, so downstream validation/plotting code can inspect how the voltage supervision was constructed.

### 3. Optional Saved-Session Analysis

- `scripts/compare_h_current_ablation.py` compares matched saved sessions and detects network bursts from saved spike trains.
- `scripts/inspect_data.py` inspects raw full-dt voltage traces from saved recordings, including chunked external HDF5 sidecars.
- `scripts/analyze_burst_voltage.py` streams raw voltage sidecars around saved stimulation onsets and detected network-burst windows, reporting raw and spike-masked baseline/during/after voltage summaries plus burst-aligned figures.
- `scripts/analyze_connected_pair_psp.py` streams raw voltage sidecars around presynaptic spikes for true connected pairs and matched unconnected controls, reporting PSP timing, amplitudes, and mean-pooling-vs-striding downsample diagnostics.
- `scripts/plot_voltage_score_separation.py` plots the log10 learned-weight distributions for true edges vs non-edges from a saved spike-only or voltage checkpoint and can overlay circular-shift surrogate null scores plus a surrogate-FDR cutoff on the same axis.
- `scripts/plot_voltage_validation_predictions.py` rebuilds the held-out validation split from a saved voltage checkpoint, runs the saved model on those windows plus the held-out validation timeline, saves both a dedicated spike-train-style raster over concatenated validation recordings and a composite window-level summary figure, and calibrates inferred spike calls for those plots with a label-free surrogate-FDR threshold rather than a label-aware validation-bin sweep.

## Key Workflow Notes

- The active simulation model is conductance-based and supports an optional slow h-current.
- Clustered connectivity semantics remain: within-cluster probability exceeds between-cluster probability, connectivity is distance-limited, and hub neurons preferentially strengthen long-range inter-cluster links.
- Saved voltage traces now default to raw membrane voltage at the full simulation step, with the main sequential workflow storing those traces in chunked external HDF5 sidecars referenced by each `recordingNNN.npz` file.
- Shared simulation logic lives in `lif_simulation/`; shared learned-LIF inference logic lives in `lif_inference/`.
- All shared plotting helpers live in `lif_simulation/plotting.py` so notebook and script outputs stay aligned.

## Main Outputs

- `LIF data/<timestamp>/` — saved network structures, recordings, and session metadata.
- `learned_lif_outputs/` — spike-only learned-LIF outputs, including threshold-mode metadata, connectivity-threshold metadata, learned threshold summaries in saved checkpoints, and explicitly named `*_oracle_f1.png` plus `*_surrogate_fdr.png` connectivity figures.
- `voltage_augmented_learned_lif_outputs/` — voltage-augmented learned-LIF outputs, including threshold-mode metadata, event-window reconstruction metadata, connectivity-threshold metadata, learned threshold summaries, saved stimulation/burst exclusion metadata, dedicated `*_score_separation.png` diagnostics, optional `*_validation_voltage.png` and `*_validation_spike_raster.png`, burst/PSP voltage-analysis subfolders, plus matching NPZ/JSON exports that now also include spike-threshold metadata, thresholded spike calls, per-window spike-quality scores, and validation-timeline spike timing summaries.
- `modular_validation_outputs/` — validation artifacts from modular checks.

## Supporting Reference Material

- `docs/SCRIPT_DETAILS.md` contains the longer script-by-script explanation for the supported simulation and learned-LIF surface.
- `docs/` contains reference notes moved out of the root to keep the supported workflow surface small.
