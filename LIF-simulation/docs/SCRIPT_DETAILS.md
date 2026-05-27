# Supported Script Details

This file restores the old long-form `SCRIPTS_SUMMARY.md` style, but only for the supported conductance-based simulation and learned-LIF inference workflow.

Archived GNN, classical baseline, presentation, scratch, and legacy helper materials are intentionally excluded and are not present in this cleaned repo snapshot.

## Current Supported Surface

```
LIF_network_simulation_network_burst_conductance.ipynb            (Step 1: main conductance-based simulation notebook)
LIF_network_simulation_network_burst_conductance_modular.ipynb    (Step 1a: modular simulation notebook over lif_simulation/)
scripts/run_conductance_simulation.py                             (Step 1b: CLI simulation entry point)
scripts/run_no_stim_validation.py                                 (Validation: spontaneous near-criticality check)
scripts/run_h_current_sag_probe.py                                (Validation: single-neuron h-current sag/rebound probe)
scripts/plot_saved_session.py                                     (Visualization: load saved sessions and generate shared plots)
scripts/compare_h_current_ablation.py                             (Analysis: compare saved h-current on/off sessions and detect bursts)
scripts/inspect_data.py                                           (Utility: inspect raw saved voltage traces)

learned_lif_connectivity_modular.ipynb                            (Step 2a: modular learned-LIF notebook over lif_inference/)
lif_inference/learned_lif_connectivity.py                         (Step 2b: packaged spike-only learned-LIF CLI module)
lif_inference/voltage_augmented_learned_lif_connectivity.py       (Step 2c: packaged voltage-augmented learned-LIF CLI module)
scripts/run_voltage_lambda_sweep.py                               (Experiment helper: sweep voltage-loss weight)

lif_simulation/                                                   (Shared simulation package)
lif_inference/                                                    (Shared learned-LIF inference package)
```

## Pipeline Execution Order

### Step 1: Simulate

**`LIF_network_simulation_network_burst_conductance_modular.ipynb`**
- Recommended notebook entry point for new simulation work.
- Current notebook structure is a 7-cell import-based workflow: intro, imports, config, no-stimulation validation, h-current sag probe, optional simulation, and saved-session load/analysis.
- Uses `lif_simulation/` for the actual model, network construction, simulation loop, save/load helpers, analysis, and plotting.
- Best when you want notebook-level control without duplicating the simulation implementation.

**`LIF_network_simulation_network_burst_conductance.ipynb`**
- Source-of-truth monolithic conductance notebook retained for parity checks and direct scientific inspection.
- Implements the clustered conductance-based LIF network, optional slow h-current, raw full-dt voltage saving, no-stimulation validation, and h-current sag/rebound probing.
- Output: `LIF data/<timestamp>/` folders containing `network_<ts>.npz`, `recording000.npz`, ..., and `session_metadata.json`.
- This remains the best reference when you need to inspect the full scientific workflow in one place.

**Thin simulation-side scripts**
- `scripts/run_conductance_simulation.py` — non-notebook saved-session generation.
- `scripts/run_no_stim_validation.py` — near-critical spontaneous firing validation.
- `scripts/run_h_current_sag_probe.py` — matched with/without-h-current sag probe.
- `scripts/plot_saved_session.py` — standard saved-session visualization and analysis.

### Step 2: Fit Learned-LIF Connectivity Models

**`learned_lif_connectivity_modular.ipynb`**
- Recommended notebook entry point for learned-LIF inference.
- Current notebook structure is a 7-cell import-based workflow: intro, imports, config, session resolution, spike-only run, voltage-augmented run, and summary.
- Imports `lif_inference/` directly instead of keeping monolithic inference code inside the notebook.

**`lif_inference/learned_lif_connectivity.py`**
- Active packaged spike-only learned-LIF implementation and CLI entry point.
- Keeps the stable spike-only public surface while delegating shared internals to helper modules such as `shared_data.py`, `candidate_selection.py`, `burst_exclusion.py`, `event_windows.py`, `event_training.py`, and `surrogate_thresholding.py`.
- Preserves compatibility exports for repo scripts that still import older helper names from this module.

**`lif_inference/voltage_augmented_learned_lif_connectivity.py`**
- Active packaged voltage-augmented learned-LIF implementation and CLI entry point.
- Shares candidate selection, burst exclusion, event windows, connectivity evaluation, and `dt` inference behavior with the spike-only and sweep paths.
- Delegates voltage-specific surrogate calibration and high-level training orchestration to `voltage_surrogate_thresholding.py` and `voltage_training_orchestration.py` while preserving the existing CLI surface.

**`scripts/run_voltage_lambda_sweep.py`**
- Shared-data sweep helper for the voltage-augmented learned-LIF model.
- Reuses one loaded session and one set of datasets while scanning multiple `voltage_lambda` values.
- Intended for tuning the strength of the masked voltage-supervision term.

**`lif_inference/` internal learned-LIF helpers**
- `shared_data.py` holds shared spike conversion, ground-truth construction, recording-boundary normalization, and circular-shift surrogate data helpers.
- `connectivity_metrics.py` holds flattened score extraction, threshold selection, binary classification metrics, and shared connectivity evaluation.
- `candidate_selection.py` and `burst_exclusion.py` hold shared candidate-proposal and burst/window exclusion logic used by both inference paths.
- `event_windows.py` and `event_training.py` hold the spike-only event-window extraction, dataset building, event loss, and event-window train/eval helpers.
- `surrogate_thresholding.py`, `voltage_surrogate_thresholding.py`, and `voltage_training_orchestration.py` isolate surrogate-threshold calibration and the voltage-side high-level training loop.

### Step 3: Optional Saved-Session Analysis

**`scripts/compare_h_current_ablation.py`**
- Offline comparison tool for saved h-current on/off sessions.
- Detects network bursts from coarse population synchrony and can save raster plots with detected burst windows shaded.

**`scripts/inspect_data.py`**
- Quick raw-voltage inspection helper for an individual recording file.
- Useful when the full notebook or saved-session plotting path is more than you need.

---

## `LIF_network_simulation_network_burst_conductance.ipynb` — Core Parameter Reference

These values reflect the current conductance-notebook defaults documented in the maintained repo notes.

### `LIFNeuron` — Shared Membrane / Conductance Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v_rest` | -61.5 | mV | Resting potential |
| `v_thresh` | -50.0 | mV | Spike threshold |
| `v_reset` | -70.0 | mV | Reset voltage after spike |
| `v_floor` | -80.0 | mV | Minimum voltage clamp |
| `R_m` | 100.0 | MOhm | Membrane resistance |
| `tau_ref` | 2.0 | ms | Absolute refractory period |
| `noise_sigma` | 1.05 | mV | Gaussian noise amplitude for sparse spontaneous firing |
| `e_exc` | 0.0 | mV | Excitatory reversal potential |
| `e_inh` | -75.0 | mV | Inhibitory reversal potential |
| `e_h` | -35.0 | mV | h-current reversal potential |
| `v_half_h` | -75.0 | mV | h-current half-activation voltage |
| `k_h` | 5.5 | mV | h-current activation slope |
| `use_h_current` | True | - | Enables slow h-current dynamics |

### `LIFNeuron` — Excitatory Defaults

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `tau_m` | 20.0 | ms | Membrane time constant |
| `adaptation_increment` | 0.025 | nA | Spike-triggered adaptation increment |
| `tau_adaptation` | 300.0 | ms | Adaptation decay time constant |
| `g_h_max` | 0.0023 | relative | Maximum h-current conductance scale |
| `tau_h` | 140.0 | ms | h-current activation time constant |

### `LIFNeuron` — Inhibitory Defaults

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `tau_m` | 10.0 | ms | Membrane time constant |
| `adaptation_increment` | 0.003 | nA | Spike-triggered adaptation increment |
| `tau_adaptation` | 40.0 | ms | Adaptation decay time constant |
| `g_h_max` | 0.0011 | relative | Maximum h-current conductance scale |
| `tau_h` | 100.0 | ms | h-current activation time constant |

### `ExpSynapse`

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `tau` (inhibitory) | 10.0 | ms | GABA conductance decay time |
| `tau` (excitatory) | 3.0 | ms | AMPA conductance decay time |
| `delay` | 1.0 | ms | Synaptic transmission delay |
| `reference_driving_force` (inhibitory) | 15.0 | mV | Legacy inhibitory weight-to-conductance conversion scale |
| `reference_driving_force` (excitatory) | 60.0 | mV | Legacy excitatory weight-to-conductance conversion scale |

### `create_clustered_network()` Defaults

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_clusters` | 20 | Number of neuron clusters |
| `neurons_per_cluster_range` | (12, 18) | Min/max neurons per cluster |
| `inhibitory_probability` | 0.2 | Fraction of inhibitory neurons |
| `within_cluster_prob` | 0.5 | Connection probability within a cluster |
| `between_cluster_prob` | 0.15 | Connection probability between clusters |
| `max_connection_distance` | 8.0 | Maximum Euclidean connection distance |
| `space_size` | 15 | 2D placement extent |
| `hub_fraction` | 0.1 | Fraction of hub neurons |
| `hub_between_prob` | 0.4 | Hub inter-cluster connection probability |
| `hub_weight_scale` | 1.5 | Hub synaptic weight multiplier |
| `hub_reciprocal_factor` | 2.0 | Reciprocal connection boost for hub-hub pairs |

### `create_periodic_cluster_stimulation()` Defaults

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `burst_interval` | 7000 | ms | Time between stimulus bursts |
| `cluster_fraction` | 0.7 | - | Fraction of clusters stimulated per burst |
| `neurons_per_cluster` | 6 | - | Neurons stimulated per selected cluster |
| `stim_amplitude_range` | (2.0, 3.5) | nA | Stimulus current amplitude range |
| `stim_duration_range` | (10, 30) | ms | Stimulus duration range |
| `burst_interval_jitter` | 1500 | ms | Timing jitter between bursts |

### `simulate_network()` Defaults

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `dt` | 0.1 | ms | Simulation time step |
| `duration` | 60000 | ms | Recording duration |
| `record_voltage` | True | - | Record membrane voltage traces |
| `voltage_sample_rate` | 1.0 | ms | Legacy requested voltage sampling interval kept in metadata |
| `target_freq` | 10 | Hz | Reference-compatible spike resampling frequency |

Saved sessions now store raw membrane voltage at full `dt` resolution rather than a display-oriented resampled trace. Session metadata records the raw-voltage mode explicitly.

### Validation / Test Cells

- The zero-stimulation validation path sets external stimulation to none, then reports spontaneous firing statistics together with a network structure plot, a raster, and example voltage traces.
- The h-current probe compares the same neuron with and without h-current to quantify sag depth and rebound.

---

## `scripts/run_conductance_simulation.py` — CLI Reference

**Usage**

```text
python -m scripts.run_conductance_simulation --mode spontaneous
python -m scripts.run_conductance_simulation --mode stimulus_driven --n-recordings 5 --recording-duration 60000
```

**Purpose**
- Thin CLI over `lif_simulation.workflows.sequential_simulation_individual_saves()`.
- Creates saved sessions under `LIF data/<timestamp>/`.
- Prints the final `session_metadata.json` path for downstream scripts.

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | None | Optional random seed |
| `--mode` | `spontaneous` | Session mode: `spontaneous` or `stimulus_driven` |
| `--save-dir` | `LIF data` | Output directory |
| `--n-recordings` | 20 | Number of recordings per session |
| `--recording-duration` | 60000.0 | Duration per recording in ms |
| `--num-clusters` | 30 | Number of clusters for CLI-generated sessions |
| `--min-neurons-per-cluster` | 12 | Lower bound on neurons per cluster |
| `--max-neurons-per-cluster` | 18 | Upper bound on neurons per cluster |
| `--inhibitory-probability` | 0.2 | Inhibitory neuron fraction |
| `--within-cluster-prob` | 0.3 | Within-cluster connection probability |
| `--between-cluster-prob` | 0.15 | Between-cluster connection probability |
| `--target-freq` | 10.0 | Resampling frequency in Hz |
| `--dt` | 0.1 | Simulation step in ms |
| `--voltage-sample-rate` | 1.0 | Requested voltage sampling interval retained in metadata |
| `--space-size` | 15.0 | Spatial extent for neuron placement |
| `--max-connection-distance` | 6.0 | Distance limit for candidate connections |
| `--no-h-current` | off | Disable h-current dynamics |
| `--burst-interval` | 7000.0 | Stimulus burst interval in ms |
| `--cluster-fraction` | 0.7 | Fraction of clusters stimulated per burst |
| `--neurons-per-cluster` | 6 | Neurons stimulated per selected cluster |
| `--stim-amplitude-min` | 2.0 | Minimum stimulation amplitude |
| `--stim-amplitude-max` | 3.5 | Maximum stimulation amplitude |
| `--stim-duration-min` | 10.0 | Minimum stimulation duration in ms |
| `--stim-duration-max` | 30.0 | Maximum stimulation duration in ms |
| `--burst-interval-jitter` | 1500.0 | Burst timing jitter in ms |
| `--hub-fraction` | 0.1 | Fraction of hub neurons |
| `--hub-between-prob` | 0.25 | Hub inter-cluster connection probability |
| `--hub-weight-scale` | 1.5 | Hub synaptic weight scaling |
| `--hub-reciprocal-factor` | 2.0 | Reciprocal hub connection boost |

---

## `scripts/run_no_stim_validation.py` — No-Stimulation Validation

**Usage**

```text
python -m scripts.run_no_stim_validation
python -m scripts.run_no_stim_validation --no-h-current --test-duration-ms 20000 --output-dir modular_validation_outputs/no_stim
```

**Purpose**
- Runs the spontaneous near-criticality validation without external stimulation.
- Prints a compact `NO_STIM_TEST` metrics block including firing-rate and active-fraction statistics.
- Generates or saves three figures: network structure, raster, and raw voltage traces.

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--test-duration-ms` | 10000.0 | Validation duration |
| `--requested-voltage-sample-rate` | 0.25 | Requested voltage sampling interval in ms |
| `--no-h-current` | off | Disable h-current for the validation run |
| `--output-dir` | None | Save figures instead of showing them interactively |

**Saved figure names when `--output-dir` is used**
- `no_stim_network_structure.png`
- `no_stim_raster.png`
- `no_stim_voltage_traces.png`

---

## `scripts/run_h_current_sag_probe.py` — h-Current Probe

**Usage**

```text
python -m scripts.run_h_current_sag_probe
python -m scripts.run_h_current_sag_probe --step-current-na -0.10 --duration-ms 1500 --output-dir modular_validation_outputs/sag
```

**Purpose**
- Runs a matched current-step probe with h-current enabled and disabled.
- Prints sag-depth and rebound summaries.
- Shows or saves a single comparison figure.

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--step-current-na` | -0.12 | Hyperpolarizing step current |
| `--duration-ms` | 1200.0 | Probe duration |
| `--dt` | 0.1 | Simulation step in ms |
| `--output-dir` | None | Save the figure instead of showing it |

**Saved figure name when `--output-dir` is used**
- `h_current_sag_probe.png`

---

## `scripts/plot_saved_session.py` — Saved Session Visualization

**Usage**

```text
python -m scripts.plot_saved_session latest
python -m scripts.plot_saved_session "LIF data/<timestamp>" --recording-index 0 --output-dir session_plots
```

**Purpose**
- Standard plotting and analysis entry point for saved sessions.
- Accepts a session directory, a `session_metadata.json` path, or `latest`.
- Loads a unified session bundle through `lif_simulation.session_views.load_session_bundle()`.

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `session_source` | `latest` | Session directory, metadata file, or `latest` |
| `--base-dir` | `LIF data` | Base directory for session lookup |
| `--recording-index` | 0 | Recording index within the session |
| `--output-dir` | None | Save figures instead of showing them |

**Generated figures**
- `session_raster.png`
- `session_voltage_traces.png`
- `session_voltage_heatmap.png`
- `session_firing_rates.png`
- `session_network_layout.png`
- `session_resampled_raster.png`
- `session_spike_train_analysis.png`
- `session_hub_network.png` when hub metadata is present
- `session_hub_degree_distributions.png` when hub stats are available
- `session_hub_firing_rates.png` when hub stats are available

---

## `scripts/compare_h_current_ablation.py` — Saved-Session Ablation Comparison

**Usage**

```text
python -m scripts.compare_h_current_ablation "LIF data/session_a" "LIF data/session_b"
python -m scripts.compare_h_current_ablation "LIF data/session_a" --json-out report.json --raster-out-dir ablation_rasters
```

**Purpose**
- Reads one or more session folders or `session_metadata.json` files.
- Computes per-session activity summaries, detects network bursts from coarse population synchrony, and optionally saves per-recording raster plots with detected burst windows shaded.
- For a matched h-current on/off pair, reports topology match and deltas such as total spikes, firing rate, active-neuron fraction, and detected burst count.

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `session_paths` | all sessions under `LIF data` | Session directories or metadata files |
| `--json-out` | None | Optional JSON report output path |
| `--raster-out-dir` | None | Optional directory for saved rasters |
| `--burst-activity-bin-ms` | 100.0 | Population-activity bin width for burst detection |
| `--burst-smooth-bins` | 3 | Smoothing width in activity bins |
| `--burst-threshold-std` | 3.0 | Threshold = mean + std factor * std |
| `--burst-min-active-frac` | 0.10 | Minimum active-neuron fraction for burst detection |
| `--burst-min-duration-ms` | 100.0 | Minimum burst duration |
| `--burst-merge-gap-ms` | 150.0 | Merge nearby burst segments separated by at most this gap |
| `--burst-pad-before-ms` | 100.0 | Pre-padding for detected burst windows |
| `--burst-pad-after-ms` | 250.0 | Post-padding for detected burst windows |

**Outputs**
- Optional JSON report via `--json-out`
- Optional per-recording raster plots via `--raster-out-dir`
- Console summaries for each session, each recording, grouped h-current on/off aggregates, and pairwise deltas when exactly two sessions are supplied

---

## `scripts/inspect_data.py` — Raw Voltage Inspection Helper

**Usage**

```text
python -m scripts.inspect_data
python -m scripts.inspect_data "LIF data/<timestamp>/recording000.npz"
```

**Purpose**
- Loads one saved recording file and inspects the raw stored voltage traces.
- Selects a high-spiking neuron, a median-firing neuron, and a silent neuron when available.
- Plots the first 5 seconds of raw voltage and overlays spike markers.

**Defaults and outputs**
- Default input: `LIF data/h_current_ablation_60s_raw_voltage/20260519_123510/recording000.npz`
- Time window: first 5000 ms
- Saved figure: `example_raw_voltage_traces_5s.png` in the same folder as the recording

---

## `lif_inference/learned_lif_connectivity.py` — Differentiable Learned-LIF Connectivity

Fits a differentiable LIF model separately for each postsynaptic neuron using only its candidate presynaptic set. Compared with classical baselines, this path tries to recover connectivity by learning membrane dynamics directly: delayed presynaptic input drives a soft LIF neuron, and the learned synaptic weights become the connectivity estimate.

By default it trains on all recordings in a session concatenated along time, while preserving recording boundaries so event windows never cross from one recording into the next.

**Usage**

```text
python -m lif_inference.learned_lif_connectivity --k 50 --epochs 40 --batch 128 --max-delay 8
python -m lif_inference.learned_lif_connectivity --subsample 10000
python -m lif_inference.learned_lif_connectivity --single-recording --recording 0 --k 30
```

**Recommended working configuration after tuning**
- `K=50`
- `epochs=40`
- `batch=128`
- `max-delay=8`
- `l1=0.01`
- `val-fraction=0.2`
- `candidate-mode=hybrid`
- `candidate-spatial-frac=0.8`

**Outputs**
- `learned_lif_outputs/learned_lif_<output_name>.png` — training curves, score distributions, PR curve, connectivity views, and learned membrane summaries
- `learned_lif_outputs/learned_lif_<output_name>.pt` — checkpoint with model state, validation summary, candidate info, neighbor indices, connectivity matrix, and histories
- `learned_lif_outputs/connectivity_<output_name>.npz` — compressed connectivity export with threshold, neighbor indices, and neuron positions

### Data Pipeline

1. Load one recording or all `recordingNNN.npz` files from a session.
2. Convert spike times into a binary spike matrix `[n_neurons, T]` at resolution `dt` ms.
3. Build the ground-truth adjacency from the saved `connections` array.
4. For each postsynaptic neuron, build a candidate presynaptic set. The current default is a hybrid proposal: nearest spatial neighbors plus causal temporal candidates.
5. Build train/validation event sets for the same neurons: if multiple recordings are available, hold out recordings; otherwise hold out a fraction of event windows per neuron.

### Event-Window Training Design

- **Positive windows:** centered on real postsynaptic spikes
- **Negative windows:** sampled far enough from any postsynaptic spike
- **Window length:** `warmup + pre_context + post_context`
- **Warmup region:** no loss is applied during the first `warmup` bins so the membrane can settle
- **Boundary handling:** windows never cross recording boundaries
- **Class balance:** controlled by `--neg-ratio`

Validation follows the same-neuron principle required by the model design: the held-out data comes from the same fitted neurons, not unseen neurons, because each postsynaptic neuron owns a separate learned weight row.

### Candidate Proposal Stage

The spike-only pipeline supports two candidate modes:

- **Spatial:** pure K-nearest neighbors in physical space
- **Hybrid:** nearest spatial neighbors plus causal temporal candidates extracted from training recordings only

The hybrid mode is the current default because it improves true-edge coverage while still keeping the fitting problem local.

### Model Design: `PerNeuronLIF`

For each postsynaptic neuron `j`, the model learns a private candidate-synapse bank and a small shared membrane parameter set.

**Per-neuron learned parameters**
- `W[j, :]` — one learned weight per candidate presynaptic neuron
- `delay_logits[j, :, :]` — a learned discrete delay distribution over `0..max_delay-1` bins for each candidate synapse

**Shared learned parameters**
- `alpha_logit` — membrane leak factor after sigmoid mapping
- `threshold` — firing threshold
- `beta` — spike nonlinearity sharpness
- `reset_strength` — post-spike reset term after softplus mapping

Conceptually the model uses:

```text
I_syn(t) = sum_i W[j, i] * pre_i(t - delay_ji)
V(t) = alpha * V(t-1) + I_syn(t)
spike_prob(t) = sigmoid(beta * (V(t) - threshold))
V(t) = V(t) - reset_strength * spike_prob(t)
```

The final connectivity score for a candidate edge is `abs(W[j, i])`.

### Training Procedure

- Uses event-window batches, where `--batch` means windows per batch
- Computes BCE spike-prediction loss only on the non-warmup portion of each window
- Adds L1 sparsity on learned weights through `--l1`
- Upweights positive spike bins through `--pos-weight`
- Clips gradients to norm 1.0
- Uses Adam plus `ReduceLROnPlateau`
- Uses early stopping based on held-out window loss while still tracking connectivity AUC for monitoring

### Evaluation and Saved Connectivity

After each epoch, the pipeline measures held-out spike-prediction loss and then evaluates the assembled connectivity matrix over all fitted neurons.

- Score per candidate edge: `abs(learned_weight)`
- Metrics: ROC AUC, average precision, precision, recall, F1
- Threshold selection: best F1 point from the validation precision-recall curve
- Saved artifacts: figure, checkpoint, and compressed connectivity export

### CLI Reference

**Session and data selection**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--session` | interactive prompt | Session folder to load |
| `--output-tag` | None | Optional suffix for saved artifact names |
| `--single-recording` | off | Use one recording instead of all recordings in the session |
| `--recording` | 0 | Recording index when using `--single-recording` |
| `--subsample` | None | Keep only the first `N` ms for quick tests |
| `--dt` | 1.0 | Bin width in ms for spike binarization |
| `--val-fraction` | 0.2 | Held-out validation fraction |

**Model and optimization**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--k` | 50 | Candidate presynaptic neurons per postsynaptic neuron |
| `--max-delay` | 8 | Number of discrete delay bins per candidate synapse |
| `--epochs` | 40 | Maximum training epochs |
| `--lr` | 0.001 | Adam learning rate |
| `--batch` | 128 | Event windows per batch |
| `--patience` | 20 | Early-stopping patience |
| `--l1` | 0.01 | L1 sparsity penalty |
| `--pos-weight` | 5.0 | Positive-class weight in BCE loss |
| `--device` | `cpu` | Compute device: `cpu` or `cuda` |
| `--candidate-mode` | `hybrid` | Candidate proposal mode |
| `--candidate-spatial-frac` | 0.8 | In hybrid mode, fraction of K reserved for spatial neighbors |
| `--candidate-min-lag` | 1 | Minimum causal lag in bins for temporal candidates |
| `--candidate-max-lag` | uses `--max-delay` | Maximum causal lag in bins for temporal candidates |

**Event-window extraction**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pre-context` | 50 | Causal presynaptic history before the event |
| `--post-context` | 10 | Bins after the event for spike and reset dynamics |
| `--warmup` | 30 | Warmup bins before the loss region |
| `--neg-ratio` | 1.0 | Negative windows per positive window |
| `--neg-min-dist` | 100 | Minimum distance in bins from any postsynaptic spike for negatives |

**Detected-burst exclusion**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--exclude-detected-bursts` | off | Exclude detected network-burst windows from candidate scoring and event extraction |
| `--burst-activity-bin-ms` | 100.0 | Burst detection activity bin width |
| `--burst-smooth-bins` | 3 | Burst detection smoothing width |
| `--burst-threshold-std` | 3.0 | Threshold = mean + std factor * std |
| `--burst-min-active-frac` | 0.10 | Minimum active-neuron fraction for burst detection |
| `--burst-min-duration-ms` | 100.0 | Minimum detected burst duration |
| `--burst-merge-gap-ms` | 150.0 | Merge nearby burst segments separated by at most this gap |
| `--burst-pad-before-ms` | 100.0 | Pre-padding for detected burst windows |
| `--burst-pad-after-ms` | 250.0 | Post-padding for detected burst windows |

---

## `lif_inference/voltage_augmented_learned_lif_connectivity.py` — Voltage-Augmented Learned-LIF

This pipeline extends the spike-only learned-LIF model with masked subthreshold voltage supervision from the saved voltage traces.

**Usage**

```text
python -m lif_inference.voltage_augmented_learned_lif_connectivity --session "LIF data/<timestamp>" --k 50 --epochs 40 --batch 128 --max-delay 8
python -m lif_inference.voltage_augmented_learned_lif_connectivity --single-recording --recording 0 --voltage-lambda 0.5
```

**Outputs**
- `voltage_augmented_learned_lif_outputs/voltage_augmented_learned_lif_<output_name>.png` — visualization summary
- `voltage_augmented_learned_lif_outputs/voltage_augmented_learned_lif_<output_name>.pt` — checkpoint with model state, validation summary, candidate info, recording summaries, and voltage-cleaning settings
- `voltage_augmented_learned_lif_outputs/connectivity_<output_name>.npz` — compressed connectivity export

### What Changes Relative to the Spike-Only Path

- Loads both spike trains and voltage traces from saved recordings
- Masks out windows around peaks so the voltage loss focuses on subthreshold dynamics
- Adds a voltage reconstruction term weighted by `--voltage-lambda`
- Reports additional quality metrics such as sign accuracy and weight correlation

### Data Pipeline

1. Load all `recordingNNN.npz` files from a session and the saved `network_*.npz` file.
2. Convert spike times into a binary spike matrix `[n_neurons, T]` at resolution `dt` ms.
3. Resolve the stored voltage array from `voltage_traces_raw` when available, otherwise `voltage_traces`.
4. When `--dt` is omitted, infer it from `session_metadata.json` or the first recording's saved voltage sample rate.
	If an explicit `dt` is coarser than the saved voltage sample rate, it must be an integer multiple; the pipeline masks spike neighborhoods at native voltage resolution and then downsamples the cleaned voltage targets into the requested bins.
5. Normalize each neuron's voltage trace after masking spike neighborhoods and any legacy high-voltage peaks.
6. Concatenate all recordings across time while preserving recording boundaries for later train/validation splitting.
7. Build candidate presynaptic sets, then create matched spike-and-voltage event windows for the same postsynaptic neurons.

### Voltage Preprocessing and Masking

The preprocessing stage is implemented by `preprocess_voltage_recording()` and is the main difference between this pipeline and the spike-only path.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask_pre_ms` | 1.0 | Mask duration before each spike-aligned sample neighborhood |
| `mask_post_ms` | 2.0 | Mask duration after each spike-aligned sample neighborhood |
| `peak_threshold_mv` | 15.0 | Mask any legacy saved voltage samples above this threshold |

For each neuron:

- spike times are mapped into sample bins using the stored voltage sampling step
- a boolean validity mask is built by excluding spike neighborhoods and high-voltage peaks
- voltage is normalized by subtracting the median and dividing by the standard deviation of the remaining valid samples
- masked samples are set to zero in the normalized target array

The loader keeps per-recording summary fields such as:

- `voltage_source_key`
- `sample_rate_ms`
- `duration_ms`
- `mean_valid_fraction`
- `min_valid_fraction`
- `max_valid_fraction`

### Voltage Event-Window Dataset

The training dataset is `VoltageEventWindowDataset`, which extends the spike-only event-window idea by returning both the postsynaptic spike target and the cleaned postsynaptic voltage target.

Each sample returns:

- candidate presynaptic spike window `[K, T]`
- postsynaptic spike window `[T]`
- normalized postsynaptic voltage window `[T]`
- postsynaptic voltage-validity mask `[T]`
- postsynaptic neuron id
- positive/negative event label

### Event-Window Extraction Parameters

These are the internal dataset controls used by `build_train_val_voltage_datasets()`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pre_context` | 50 | Causal presynaptic history before the event |
| `post_context` | 10 | Bins after the event for spike and reset dynamics |
| `warmup` | 30 | Warmup bins before the loss region starts |
| `neg_ratio` | 1.0 | Negative windows per positive window |
| `neg_min_distance` | 100 | Minimum distance in bins from any postsynaptic spike for negatives |
| `val_fraction` | 0.2 | Validation fraction |

Validation follows the same-neuron principle as the spike-only path:

- if multiple recordings exist, the preferred strategy is held-out recordings
- otherwise the pipeline splits event windows from the same fitted neurons into train and validation subsets

### Model Design: `VoltageAugmentedPerNeuronLIF`

`VoltageAugmentedPerNeuronLIF` keeps the same delayed-synapse structure as the spike-only model, but adds a learned postsynaptic bias term and supervises the latent membrane trace directly.

**Per-neuron learned parameters**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `W` | `[n_neurons, K]` | One learned weight per candidate presynaptic neuron |
| `delay_logits` | `[n_neurons, K, max_delay]` | Learned discrete delay distribution for each candidate synapse |
| `bias` | `[n_neurons]` | Learned postsynaptic bias current term |

**Shared learned parameters**

| Parameter | Initial value | Description |
|-----------|---------------|-------------|
| `alpha_logit` | 3.0 | Leak factor after sigmoid mapping |
| `threshold` | 1.0 | Shared firing threshold |
| `beta` | 5.0 | Shared spike nonlinearity sharpness |
| `reset_strength` | 2.0 | Shared post-spike reset term after softplus |

The forward path builds delayed presynaptic inputs, combines them with learned weights plus the learned bias, then updates a soft LIF membrane in short chunks.

Conceptually:

```text
I_syn(t) = sum_i W[j, i] * pre_i(t - delay_ji) + bias_j
V(t) = alpha * V(t-1) + I_syn(t)
spike_prob(t) = sigmoid(beta * (V(t) - threshold))
V(t) = V(t) - reset_strength * spike_prob(t)
```

### Combined Loss Function

The main loss is implemented by `compute_voltage_augmented_event_loss()`.

| Loss term | Description |
|-----------|-------------|
| `spike_loss` | BCE spike-prediction loss on the post-warmup region only |
| `voltage_loss` | Smooth L1 loss between predicted and target voltage on valid masked points only |
| `l1_loss` | Sparsity penalty on learned candidate weights |
| `total` | `spike_loss + voltage_lambda * voltage_loss + l1_loss` |

The function also tracks `n_voltage_points`, the number of valid voltage targets that contributed to the voltage term.

### Training Procedure

- Uses `VoltageEventWindowDataset` batches, where `--batch` means event windows per batch
- Runs full-window BPTT for each short event window rather than long-sequence optimization
- Computes spike loss only after the warmup region
- Computes voltage loss only where the voltage-validity mask is true
- Adds L1 sparsity to the learned candidate weights
- Clips gradients to norm 1.0 every step
- Uses Adam plus `ReduceLROnPlateau`
- Uses early stopping on held-out total loss while also tracking connectivity AUC

### Evaluation and Saved Connectivity

After training, the pipeline evaluates both held-out event windows and the assembled connectivity matrix over all fitted neurons.

**Held-out window metrics**

- `loss`
- `spike_loss`
- `voltage_loss`
- `l1_loss`
- `n_windows`
- `n_voltage_points`

**Connectivity metrics**

- ROC AUC
- average precision
- precision
- recall
- F1
- sign accuracy on true connected edges
- weight correlation on true connected edges
- predicted true-positive sign accuracy

Threshold selection still uses the best F1 point from the precision-recall curve, and the final connectivity export remains the full `[post, pre]` matrix assembled from the learned candidate weights.

### Voltage Cleaning / Masking Controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mask-pre-ms` | 1.0 | Mask duration before detected peaks |
| `--mask-post-ms` | 5.0 | Mask duration after detected peaks |
| `--peak-threshold-mv` | 15.0 | Voltage threshold for peak masking |
| `--voltage-lambda` | 1.0 | Weight on masked subthreshold voltage loss |

### Shared Learned-LIF Controls

The voltage-augmented pipeline keeps the same main controls as the spike-only path:

- candidate mode and candidate lag controls
- event-window extraction controls
- single-recording vs all-recordings selection
- standard optimization controls (`--epochs`, `--lr`, `--batch`, `--patience`, `--l1`, `--pos-weight`)

In the current packaged implementation, `run_pipeline()` also stores the voltage-cleaning configuration and per-recording validity summaries inside the saved checkpoint so later sweeps and notebook analysis can recover how the voltage supervision was constructed.

### CLI Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--session` | interactive prompt | Session folder to load |
| `--output-tag` | None | Optional suffix for saved artifact names |
| `--k` | 50 | Candidate presynaptic neurons per postsynaptic neuron |
| `--epochs` | 40 | Maximum training epochs |
| `--lr` | 0.001 | Adam learning rate |
| `--batch` | 128 | Event windows per batch |
| `--patience` | 20 | Early-stopping patience |
| `--max-delay` | 8 | Number of discrete delay bins per candidate synapse |
| `--l1` | 0.01 | L1 sparsity penalty |
| `--pos-weight` | 5.0 | Positive-class weight in BCE spike loss |
| `--voltage-lambda` | 1.0 | Weight on masked voltage loss |
| `--dt` | inferred from session metadata | Optional spike/voltage bin width override in ms |
| `--recording` | 0 | Recording index when using a single recording |
| `--single-recording` | off | Use only one recording instead of all recordings |
| `--subsample` | None | Keep only the first `N` ms for quick tests |
| `--device` | `cpu` | Compute device: `cpu` or `cuda` |
| `--candidate-mode` | `hybrid` | Candidate proposal mode |
| `--candidate-spatial-frac` | 0.8 | Fraction of K reserved for spatial neighbors |
| `--candidate-min-lag` | 1 | Minimum causal lag in bins |
| `--candidate-max-lag` | uses `--max-delay` | Maximum causal lag in bins |
| `--pre-context` | 50 | Event-window presynaptic history |
| `--post-context` | 10 | Event-window post-spike context |
| `--warmup` | 30 | Warmup bins before loss starts |
| `--neg-ratio` | 1.0 | Negative windows per positive window |
| `--neg-min-dist` | 100 | Minimum distance from any postsynaptic spike for negatives |
| `--val-fraction` | 0.2 | Validation fraction |
| `--mask-pre-ms` | 1.0 | Mask duration before peaks |
| `--mask-post-ms` | 5.0 | Mask duration after peaks |
| `--peak-threshold-mv` | 15.0 | Peak masking threshold |

---

## `scripts/run_voltage_lambda_sweep.py` — Voltage-Lambda Sweep

**Usage**

```text
python -m scripts.run_voltage_lambda_sweep --session "LIF data/<timestamp>"
python -m scripts.run_voltage_lambda_sweep --session "LIF data/<timestamp>" --lambdas 0,0.25,0.5,1,4 --device cpu
```

**Purpose**
- Reuses one loaded session, one candidate set, and one train/validation split across multiple `voltage_lambda` values.
- Saves per-lambda checkpoints and connectivity exports plus a JSON summary, a CSV summary, and a sweep plot.
- Infers `dt` from `session_metadata.json` by default for voltage-enabled sessions, while still allowing `--dt` to override it.

### Main Outputs

- Per-lambda checkpoints: `voltage_augmented_learned_lif_outputs/voltage_augmented_learned_lif_<output_name>.pt`
- Per-lambda connectivity exports: `voltage_augmented_learned_lif_outputs/connectivity_<output_name>.npz`
- Sweep summary JSON: `voltage_augmented_learned_lif_outputs/voltage_lambda_sweep_<base_stem>.json`
- Sweep summary CSV: `voltage_augmented_learned_lif_outputs/voltage_lambda_sweep_<base_stem>.csv`
- Sweep plot: `voltage_augmented_learned_lif_outputs/voltage_lambda_sweep_<base_stem>.png`

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--session` | required | Session folder to analyze |
| `--lambdas` | `0,0.25,0.5,1,4` | Comma-separated voltage-loss weights |
| `--k` | 100 | Candidate presynaptic neurons per postsynaptic neuron |
| `--epochs` | 10 | Training epochs per lambda |
| `--batch` | 256 | Event windows per batch |
| `--lr` | 0.001 | Learning rate |
| `--max-delay` | 8 | Number of delay bins |
| `--l1` | 0.01 | L1 sparsity penalty |
| `--pos-weight` | 5.0 | Positive-class weight |
| `--dt` | inferred from session metadata | Optional bin width override in ms |
| `--device` | `cuda` | Compute device |
| `--candidate-mode` | `hybrid` | Candidate proposal mode |
| `--candidate-spatial-frac` | 0.8 | Fraction of K reserved for spatial neighbors |
| `--candidate-min-lag` | 1 | Minimum causal lag in bins |
| `--candidate-max-lag` | None | Maximum causal lag in bins |
| `--pre-context` | 50 | Event-window presynaptic history |
| `--post-context` | 10 | Event-window post-spike context |
| `--warmup` | 30 | Warmup bins before loss starts |
| `--neg-ratio` | 1.0 | Negative windows per positive window |
| `--neg-min-dist` | 100 | Minimum distance from any postsynaptic spike for negatives |
| `--val-fraction` | 0.2 | Validation fraction |
| `--mask-pre-ms` | 1.0 | Mask duration before peaks |
| `--mask-post-ms` | 5.0 | Mask duration after peaks |
| `--peak-threshold-mv` | 15.0 | Peak masking threshold |

---

## Output Locations

- `LIF data/<timestamp>/` — saved simulation sessions, network structures, recordings, and metadata
- `learned_lif_outputs/` — spike-only learned-LIF figures, checkpoints, and compressed connectivity exports
- `voltage_augmented_learned_lif_outputs/` — voltage-augmented learned-LIF figures, checkpoints, connectivity exports, and sweep summaries
- `modular_validation_outputs/` — optional saved outputs from validation or probe runs

## Relationship to the Other Docs

- `SCRIPTS_SUMMARY.md` stays short and only maps the supported workflow.
- `docs/SCRIPT_DETAILS.md` holds the longer supported-only script explanations.
