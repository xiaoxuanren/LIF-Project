# Scripts Summary — LIF Neuron Network + GNN Connectivity Prediction

## Current Scripts

```
LIF_network_simulation_network_burst_conductance.ipynb   (Step 1: simulate; main conductance + h-current notebook)
LIF_network_simulation_network_burst.ipynb               (Legacy reference: original current-based notebook)
process_existing_for_gnn.ipynb                           (Step 2: preprocess)
gnn_cross_network_prediction.py                          (Step 3: train GNN)
predict_new_network.py                                   (Step 4: predict on new data)

perceptron_connectivity.py                   (Baseline: perceptron weight learning)
spike_train_connectivity.py                  (Baseline: CNN/LSTM/Perceptron on raw spike trains)
learned_lif_connectivity.py                  (Learned LIF: differentiable LIF for connectivity)
voltage_augmented_learned_lif_connectivity.py (Learned LIF + masked subthreshold voltage supervision)
correlation_connectivity.py                  (Baseline: correlation heatmap + spatial connectivity map)
analyze_predicted_connectivity_distribution.py (Analysis: predicted edge sign/strength and false-negative distributions)
resample_hires.py                            (Utility: resample recordings to 0.1ms)
create_simulation_learned_lif_presentation.py (Utility: generate learned-LIF presentation deck)
presentation_oral_script.txt                 (Utility: oral presentation notes)
raw_subthreshold_voltage_saving_patch.ipynb  (Notebook patch: save raw sampled membrane voltage alongside visualization traces)
```

## Pipeline Execution Order

### Step 1: Simulate
**`LIF_network_simulation_network_burst_conductance.ipynb`**
- Main entry point. Conductance-based copy of the clustered LIF network notebook. Recurrent synapses now use voltage-dependent conductance drive with legacy weight scaling so the original tuning remains usable.
- Adds a slow hyperpolarization-activated h-current, a zero-stimulation validation cell, a sag/rebound probe, and more biophysically realistic rendered spike traces for voltage plots.
- Output: `LIF data/<timestamp>/` folders containing `network_<ts>.npz`, `recording000.npz`, ..., `session_metadata.json`
- See full parameter reference below.

The original current-based reference notebook remains available as `LIF_network_simulation_network_burst.ipynb`.

### Step 2: Preprocess for GNN
**`process_existing_for_gnn.ipynb`**
- Combines multiple recordings per session into a single `recording_combined.npz` with time offsets.
- Creates `session_gnn_metadata.json` in each session folder.
- No tunable parameters (processes all sessions in `LIF data/`).

### Step 3: Train GNN (cross-network)
**`gnn_cross_network_prediction.py`**
- Main GNN training pipeline. Trains an inductive GNN (GraphSAGE) across multiple networks.
- **Requires at least 2 session folders** with `session_gnn_metadata.json`.
- Saves model to `gnn_outputs/models/trained_gnn_model.pt`.
- See full parameter reference below.

### Step 4 (optional): Predict on new networks
**`predict_new_network.py`**
- Loads the pre-trained model from Step 3 (no re-training).
- Predicts connectivity on any network, generates visualizations, exports as NPZ and CSV.
- Uses the same `CrossNetworkConfig` saved in the model checkpoint.
- Additional parameter: `model_path` (default: `gnn_outputs/models/trained_gnn_model.pt`).

---

## LIF_network_simulation_network_burst_conductance.ipynb — Full Parameter Reference

### LIFNeuron — Shared Membrane / Conductance Parameters (Part 1)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `v_rest` | -61.5 | mV | Resting potential (11.5 mV below threshold) |
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

### LIFNeuron — Excitatory (pyramidal)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `tau_m` | 20.0 | ms | Membrane time constant |
| `adaptation_increment` | 0.025 | nA | Spike-triggered adaptation increment |
| `tau_adaptation` | 300.0 | ms | Adaptation decay time constant |
| `g_h_max` | 0.0023 | relative | Maximum h-current conductance scale |
| `tau_h` | 140.0 | ms | h-current activation time constant |

### LIFNeuron — Inhibitory (interneuron)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `tau_m` | 10.0 | ms | Membrane time constant |
| `adaptation_increment` | 0.003 | nA | Spike-triggered adaptation increment |
| `tau_adaptation` | 40.0 | ms | Adaptation decay time constant |
| `g_h_max` | 0.0011 | relative | Maximum h-current conductance scale |
| `tau_h` | 100.0 | ms | h-current activation time constant |

### ExpSynapse (Part 1)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `tau` (inhibitory) | 10.0 | ms | GABA conductance decay time |
| `tau` (excitatory) | 3.0 | ms | AMPA conductance decay time |
| `delay` | 1.0 | ms | Synaptic transmission delay |
| `reference_driving_force` (inhibitory) | 15.0 | mV | Nominal driving force for legacy weight-to-conductance conversion |
| `reference_driving_force` (excitatory) | 60.0 | mV | Nominal driving force for legacy weight-to-conductance conversion |
| `g_increment` | `abs(weight) / reference_driving_force` | — | Conductance increment added when a delayed spike arrives |

### NetworkWeightParameters (Part 1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `within_exc_range` | (0.25, 0.45) | Within-cluster excitatory weight range |
| `between_exc_range` | (0.20, 0.35) | Between-cluster excitatory weight range |
| `within_inh_range` | (0.35, 0.55) | Within-cluster inhibitory weight range |
| `between_inh_range` | (0.28, 0.45) | Between-cluster inhibitory weight range |
| `lognormal_sigma` | 0.5 | Log-normal distribution spread for weight sampling |

### create_clustered_network() (Part 2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_clusters` | 20 | Number of neuron clusters |
| `neurons_per_cluster_range` | (12, 18) | Min/max neurons per cluster (~300 total) |
| `inhibitory_probability` | 0.2 | Fraction of neurons that are inhibitory |
| `cluster_radius` | 1.0 | Spatial radius of each cluster |
| `within_cluster_prob` | 0.5 | Connection probability within a cluster |
| `between_cluster_prob` | 0.15 | Connection probability between clusters |
| `max_connection_distance` | 8.0 | Max Euclidean distance for any connection |
| `space_size` | 15 | 2D space extent for cluster placement |
| `hub_fraction` | 0.1 | Fraction of neurons per cluster designated as hubs |
| `hub_between_prob` | 0.4 | Inter-cluster connection probability for hub neurons |
| `hub_weight_scale` | 1.5 | Multiplicative weight scaling for hub synapses |
| `hub_reciprocal_factor` | 2.0 | Reciprocal connection probability boost for hub-hub pairs |

### create_periodic_cluster_stimulation() (Part 3)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `burst_interval` | 7000 | ms | Time between stimulus bursts |
| `cluster_fraction` | 0.7 | — | Fraction of clusters stimulated per burst |
| `neurons_per_cluster` | 6 | — | Neurons stimulated per selected cluster |
| `stim_amplitude_range` | (2.0, 3.5) | nA | Stimulus current amplitude range |
| `stim_duration_range` | (10, 30) | ms | Stimulus duration range |
| `burst_interval_jitter` | 1500 | ms | +/- timing jitter between bursts |
| `per_cluster_jitter` | 75 | ms | +/- stagger between clusters within a burst |
| `per_neuron_jitter` | 20 | ms | +/- jitter per individual neuron |

### simulate_network() (Part 4)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `dt` | 0.1 | ms | Simulation time step (10 kHz effective resolution) |
| `duration` | 60000 | ms | Total simulation duration |
| `record_voltage` | True | — | Whether to record membrane potential traces |
| `voltage_sample_rate` | 1.0 | ms | Interval between voltage samples |

The conductance notebook records continuous subthreshold voltage and overlays a short action-potential waveform only for visualization, so the saved plots look more biophysical while the neuron model remains LIF.

### resample_data() (Part 5)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `target_freq` | 10 | Hz | Reference-compatible binning frequency (100 ms bins) |

### Execution Call (Part 10)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_recordings` | 1 | Number of sequential recordings per session |
| `recording_duration` | 60000 | Duration per recording (ms) |
| `target_freq` | 10 | Saved resampling frequency |
| `save_dir` | 'LIF data' | Output directory |

### Validation / Test Cells

- The zero-stimulation validation cell sets `stimulation_events_test = []`, plots the network structure, a first-minute raster, and example voltage traces, and reports a near-criticality verdict from the spontaneous firing statistics.
- The h-current probe cell compares the same neuron with and without `g_h_max` to quantify sag depth and rebound.

---

## gnn_cross_network_prediction.py — Full Parameter Reference

All parameters live in the `CrossNetworkConfig` dataclass (line 132).
The `__main__` block (line 1420) sets the values actually used at runtime.

### Feature Extraction

| Parameter | Default | Runtime | Description |
|-----------|---------|---------|-------------|
| `bin_size_ms` | 5.0 | 5.0 | Spike binning resolution (ms). Finer than 20ms to capture synaptic timing |
| `correlation_lags` | [-4..4] | [-4..4] | Cross-correlation lag offsets (in bins). At 5ms bins = -20ms to +20ms window |

### Spatial / Connectivity

| Parameter | Default | Runtime | Description |
|-----------|---------|---------|-------------|
| `connection_radius` | 2.0 | 2.0 | Max Euclidean distance to consider a candidate connection |
| `neighbor_radius` | 1.0 | 1.0 | Radius for computing local neighborhood node features |

### Sliding Window Prediction

| Parameter | Default | Runtime | Description |
|-----------|---------|---------|-------------|
| `use_sliding_window` | True | True | Enable 2-stage sliding window prediction (memory efficient) |
| `window_size` | 3.0 | 3.0 | Spatial size of each sliding window |
| `window_overlap` | 0.5 | 0.5 | Overlap ratio between windows (0 = no overlap, 1 = full overlap) |
| `edges_per_batch` | 2000 | 2000 | Max edges predicted per batch within a window |

### Negative Sampling

| Parameter | Default | Runtime | Description |
|-----------|---------|---------|-------------|
| `neg_ratio` | 3.0 | 3.0 | Ratio of negative to positive edges in training set |
| `max_edges_per_graph` | 5000 | 5000 | Memory cap: max total edges per graph during prediction |
| `filter_by_radius` | True | True | Only consider edges within `connection_radius` |
| `hard_negative_ratio` | 0.5 | 0.5 | Fraction of negatives that are "hard" (nearby but unconnected) |
| `hard_negative_radius` | 1.0 | 1.0 | Distance threshold defining hard negatives |

### Model Architecture

| Parameter | Default | Runtime | Description |
|-----------|---------|---------|-------------|
| `hidden_dim` | 64 | 64 | Hidden layer dimension for GNN encoder and edge predictor |
| `num_layers` | 3 | 3 | Number of GNN message-passing layers |
| `dropout` | 0.3 | 0.3 | Dropout rate in all layers |
| `conv_type` | 'SAGE' | 'SAGE' | GNN conv type. SAGE = GraphSAGE (inductive). Also supports 'GAT', 'GCN' |

### Training

| Parameter | Default | Runtime | Description |
|-----------|---------|---------|-------------|
| `learning_rate` | 0.001 | 0.001 | Adam optimizer learning rate |
| `weight_decay` | 1e-4 | 1e-4 | L2 regularization weight |
| `num_epochs` | 300 | 300 | Maximum training epochs |
| `patience` | 30 | 30 | Early stopping: epochs without val AUC improvement |
| `batch_size` | 1 | 1 | Graphs per training batch (1 for variable-sized graphs) |
| `pos_weight` | 5.0 | 5.0 | BCEWithLogitsLoss weight for positive class (handles class imbalance) |

### Runtime-Only Parameters (set in `__main__`, not in config)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_test` | 1 | Number of networks held out for testing |
| `test_indices` | random | Which network indices to use for test (randomly selected) |
| `metadata_files` | `LIF data/*/session_gnn_metadata.json` | Glob pattern for finding session data |

---

## Node Features Extracted (per neuron)

Computed in `GeneralizableFeatureExtractor.extract_node_features()` (line 209).
No cluster info needed — all features are relative/statistical.

| Feature | Description |
|---------|-------------|
| Firing rate (Hz) | Spikes per second |
| Spike count | Total number of spikes |
| Mean ISI | Mean inter-spike interval (ms) |
| CV of ISI | Coefficient of variation of ISI (std/mean) |
| Burstiness | Fraction of ISIs < 10ms |
| Local density | Number of neurons within `neighbor_radius` |
| Avg neighbor distance | Mean distance to spatial neighbors |
| Neighbor firing rate mean | Mean firing rate of spatial neighbors |
| Neighbor firing rate std | Std of firing rate of spatial neighbors |

All features are z-score normalized via `StandardScaler` (fit on training data).

## Edge Features Extracted (per candidate edge)

Computed in `GeneralizableFeatureExtractor.extract_edge_features()` (line 301).

| Feature | Description |
|---------|-------------|
| Euclidean distance | Raw distance between neuron positions |
| Normalized distance | distance / connection_radius |
| Within radius | Binary: 1 if distance <= connection_radius |
| Cross-correlations (x9) | Pearson correlation at each lag in `correlation_lags` |
| Transfer entropy i->j | TE measuring causal influence from source to target |
| Transfer entropy j->i | TE measuring causal influence from target to source |
| TE asymmetry | TE(i->j) - TE(j->i) |
| TE max | max(TE(i->j), TE(j->i)) |
| Firing rate difference | abs(FR_i - FR_j) |
| Firing rate product | FR_i * FR_j |

Total edge feature dimension: 3 + len(correlation_lags) + 4 + 2 = **18** (with default 9 lags).

---

## predict_new_network.py — Parameters

This script uses the saved `CrossNetworkConfig` from the model checkpoint.
No separate config needed — it loads everything from `trained_gnn_model.pt`.

| Parameter | Source | Description |
|-----------|--------|-------------|
| `model_path` | CLI or default | Path to saved model (default: `gnn_outputs/models/trained_gnn_model.pt`) |
| `metadata_file` | CLI arg or interactive | Path to session's `session_gnn_metadata.json` |
| `optimal_threshold` | Computed | F1-maximizing threshold from precision-recall curve |

---

## perceptron_connectivity.py — Perceptron Baseline

Based on Ren, Bok, Vareberg, Hai (2023) "Stimulation-mediated reverse engineering of silent neural networks".

For each postsynaptic neuron, learns input weights from all other neurons using the perceptron learning rule on binary spike trains at 1ms resolution.

**Usage:**
```
python perceptron_connectivity.py [session_folder]
python perceptron_connectivity.py   # interactive selection
```

**Output:** `perceptron_outputs/perceptron_<session>.png` + `perceptron_weights_<session>.npz`

### Parameters (in `run_inference()`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iterations` | 50 | Number of passes through the data |
| `lr` | 0.01 | Perceptron learning rate |
| `threshold` | 20.0 | Firing threshold (should be tuned to match network weight scale) |
| `delay_ms` | 1 | Synaptic delay in ms (1 bin at 1ms resolution) |
| `recording_idx` | 0 | Which recording to use |

### Key Functions

| Function | Description |
|----------|-------------|
| `spike_times_to_binary()` | Convert spike times to binary matrix at given dt |
| `learn_weights_perceptron()` | Learn input weights for a single postsynaptic neuron |
| `learn_all_weights()` | Learn connectivity for all neurons |
| `evaluate_predictions()` | Compute AUC, AP, F1, weight correlation, sign accuracy |
| `plot_results()` | 6-panel visualization (weight recovery, PR curve, connections) |

### Known Limitations

- Poor performance on recurrent networks (designed for feedforward; A→B→C confounds)
- Requires dense firing (struggles with sparse ~0.5 Hz inter-burst activity)
- No regularization → spreads weight across many false candidates
- Useful as a **baseline** to benchmark against GNN and CNN approaches

---

## spike_train_connectivity.py — CNN/LSTM/Perceptron on Raw Spike Trains

For each postsynaptic neuron, selects K nearest spatial neighbors as candidate presynaptic neurons. Feeds their paired spike trains (2-channel: [pre, post]) through a temporal model to predict connectivity. The activation threshold is a **learnable parameter**.

**Usage:**
```
python spike_train_connectivity.py --model cnn --k 50 --epochs 100
python spike_train_connectivity.py --model lstm --k 30
python spike_train_connectivity.py --model perceptron --k 50
```

**Output:** `spike_cnn_outputs/<model>_<session>.png` + `<model>_<session>.pt`

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | cnn | Model type: `cnn`, `lstm`, or `perceptron` |
| `--k` | 50 | Number of nearest candidate presynaptic neurons |
| `--epochs` | 100 | Max training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch` | 8 | Batch size (number of postsynaptic neurons per batch) |
| `--patience` | 15 | Early stopping patience (epochs without val AUC improvement) |
| `--dt` | 1.0 | Spike train bin size in ms |
| `--hidden` | 64 | Hidden layer dimension |
| `--recording` | 0 | Which recording to use |
| `--session` | interactive | Session directory path |

### Model Architectures

**CNNModel**: 4-layer 1D CNN (kernel=5 → 5ms at 1ms bins, matching synaptic timescale). Input: [2, T] (pre + post spike trains). Pooling reduces T from 60000 → 8. Final: adaptive avg pool → classifier MLP with distance feature.

**LSTMModel**: Conv downsampling (60000→600) then 2-layer bidirectional LSTM. Captures long-range temporal dependencies. Slower than CNN.

**PerceptronModel**: Linear temporal summarizers for pre and post spike trains, then interaction MLP. Simplest model — learned version of the classical perceptron baseline.

All models include `LearnedThreshold` — a learnable parameter replacing the hardcoded activation threshold.

### Data Pipeline

| Component | Description |
|-----------|-------------|
| `NeuronPairDataset` | Each sample = 1 postsynaptic neuron. Returns K pre spike trains, 1 post spike train, K labels, K weights, K distances |
| `compute_neighbor_indices()` | Precomputes K nearest neighbors per neuron by Euclidean distance |
| Train/val split | By neuron (80/20 default). Val neurons completely unseen during training |
| Loss | `BCEWithLogitsLoss` with positive class weighting (default 5.0) |
| Optimizer | Adam with weight decay 1e-4, ReduceLROnPlateau scheduler |

---

## resample_hires.py — Resample Recordings to 0.1ms

Overwrites each `recording*.npz` in-place, replacing the resampled spike train fields with 0.1ms resolution binary spike trains.

**Usage:**
```
python resample_hires.py
```

Processes all sessions in `LIF data/` automatically. No arguments needed.

### Fields Modified (in each recording .npz)

| Field | Before | After |
|-------|--------|-------|
| `resampled_spikes` | [n_neurons, 1200] at 20Hz | [n_neurons, 600000] at 10kHz |
| `resampling_frequency` | 20 | 10000 |
| `resampling_interval_ms` | 50.0 | 0.1 |
| `resampled_time_points` | [1200] | [600000] |
| `resampled_spike_positions` | sparse indices | sparse indices |

All other fields (`spike_times`, `voltage_traces`, etc.) are unchanged.

**File size:** ~125-133 MB per recording (dominated by voltage traces; the sparse binary matrix compresses well).

---

## create_simulation_learned_lif_presentation.py — Generate Project Presentation

Builds the current simulation + learned-LIF PowerPoint deck from local result files and figure assets.

**Usage:**
```
python create_simulation_learned_lif_presentation.py
```

**Output:** `LIF_Simulation_Learned_LIF_Presentation.pptx`

**Requires:** `python-pptx`, `Pillow`, `numpy`, and `torch`, plus the local asset folders and images referenced in the script (`presentation_assets/`, `correlation_outputs/`, `learned_lif_outputs/`, `raster_plot.png`, `output.png`, `network spikes (resampled ata 20Hz).png`).

### Content Covered

- Simulation overview using the saved raster, burst, and voltage figures
- Preprocessing and pipeline architecture for the learned-LIF workflow
- Training strategy and differentiable LIF model summary
- Voltage-augmented variant and recent quantitative results
- Dataset-by-dataset truth vs lagged-correlation vs learned-LIF comparison slides
- Tradeoff and closing slides for presentation delivery

---

## learned_lif_connectivity.py — Differentiable LIF Connectivity

Fits a differentiable LIF model separately for each postsynaptic neuron using only its K nearest spatial neighbors as candidate presynaptic sources. Compared with the perceptron and CNN/LSTM baselines, this script tries to recover connectivity by learning membrane dynamics directly: delayed presynaptic input drives a soft LIF neuron, and the learned synaptic weights become the connectivity estimate.

By default it trains on all recordings in a session concatenated along time, but it keeps recording boundaries so event windows never cross from one recording into the next.

**Usage:**
```
python learned_lif_connectivity.py --k 50 --epochs 40 --batch 128 --max-delay 8
python learned_lif_connectivity.py --subsample 10000
python learned_lif_connectivity.py --single-recording --recording 0 --k 30
```

**Recommended working configuration after tuning:** `K=50`, `epochs=40`, `batch=128`, `max-delay=8`, `l1=0.01`, `val-fraction=0.2`, `candidate-mode=hybrid`, `candidate-spatial-frac=0.8`

**Outputs:**
- `learned_lif_outputs/learned_lif_<session>.png` — training curves, score distributions, PR curve, true/predicted connectivity views, learned membrane parameters
- `learned_lif_outputs/learned_lif_<session>.pt` — checkpoint with `model_state_dict`, `K`, `T`, `dt`, `max_delay`, `n_neurons`, `validation_strategy`, `neighbor_indices`, `connectivity_matrix`, held-out window loss summary, all-neuron connectivity metrics, training loss history, validation loss history, connectivity AUC history
- `learned_lif_outputs/connectivity_<session>.npz` — compressed export with `connectivity_matrix`, chosen threshold, `neighbor_indices`, and `neuron_positions`

### Data Pipeline

1. Load either one recording or all `recordingNNN.npz` files from a session.
2. Convert spike times into a binary spike matrix `[n_neurons, T]` at resolution `dt` ms.
3. Build the ground-truth adjacency from the saved `connections` array.
4. For each postsynaptic neuron, build a candidate presynaptic set. The current default is a hybrid proposal: `40` nearest spatial neighbors plus `10` causal temporal candidates at `K=50`.
5. Build train/validation event sets for the same neurons: if multiple recordings are available, hold out whole recordings; otherwise hold out a fraction of event windows per neuron.

### Event-Window Training Design

This script no longer trains on entire 60 s spike trains directly. Instead it extracts short event windows so the optimizer sees informative examples instead of mostly zero timesteps.

- **Positive windows:** centered on real postsynaptic spikes
- **Negative windows:** sampled at times at least `neg_min_distance` bins away from any postsynaptic spike
- **Window length:** `warmup + pre_context + post_context`
- **Positive alignment:** the real postsynaptic spike sits at index `warmup + pre_context`
- **Warmup region:** the first `warmup` bins are passed through the model but excluded from the loss so the membrane voltage can settle
- **Boundary handling:** windows must fit completely inside a single recording; concatenated sessions use `boundaries` to prevent cross-recording leakage
- **Class balance:** `neg_ratio` controls how many negative windows are sampled per positive window

Validation follows the same-neuron principle required by the model design:

- **Multi-recording sessions:** hold out one or more recordings for the same neurons
- **Single-recording sessions:** split each neuron's event windows into train and validation subsets
- **Why:** every postsynaptic neuron has its own learnable weight row, so holding out whole neurons leaves those rows untrained and makes validation meaningless

This event-window dataset is the main fix for the trivial "always zero" solution that appears when training on full spike sequences with extremely sparse targets.

### Candidate Proposal Stage

The script now supports two candidate modes before fitting the differentiable LIF model:

- **Spatial:** pure K-nearest neighbors in physical space
- **Hybrid:** nearest spatial neighbors plus causal temporal candidates extracted from the training recordings only

The hybrid mode scores a presynaptic neuron higher when it repeatedly spikes a few bins before a postsynaptic spike. In the current tuned setup, hybrid candidates increased true-edge coverage substantially on both benchmark sessions while improving held-out loss and precision-oriented connectivity metrics.

### Model Design: `PerNeuronLIF`

For each postsynaptic neuron `j`, the model learns a private candidate-synapse bank while sharing a small set of global membrane parameters across all neurons.

**Per-neuron parameters:**
- `W[j, :]` — one learnable weight per candidate presynaptic neuron; this is the inferred connectivity
- `delay_logits[j, :, :]` — a learnable discrete delay distribution over `0..max_delay-1` bins for each candidate synapse

**Shared global parameters:**
- `alpha_logit` — membrane leak factor after sigmoid mapping
- `threshold` — firing threshold
- `beta` — spike nonlinearity sharpness
- `reset_strength` — post-spike reset term after softplus mapping

At each step the model builds delayed presynaptic inputs with a softmax over discrete delays, sums them with the learned weights, and updates a soft LIF membrane:

```
I_syn(t) = sum_i W[j, i] * pre_i(t - delay_ji)
V(t) = alpha * V(t-1) + I_syn(t)
spike_prob(t) = sigmoid(beta * (V(t) - threshold))
V(t) = V(t) - reset_strength * spike_prob(t)
```

The final connectivity score for a candidate edge is `abs(W[j, i])`. Larger magnitude means more evidence for a connection.

### Training Procedure

- Uses `EventWindowDataset` batches, where `--batch` means windows per batch, not postsynaptic neurons per batch
- Computes BCE spike-prediction loss only on the non-warmup portion of each window
- Adds L1 sparsity on learned weights to push unneeded candidate edges toward zero
- Upweights positive spike bins with `--pos-weight` because spikes are rare even inside event windows
- Clips gradients to norm 1.0 every step
- Uses Adam plus `ReduceLROnPlateau`
- Uses early stopping with `--patience` based on held-out window loss, while still tracking connectivity AUC over all fitted neurons for monitoring

`PerNeuronLIF.forward()` supports truncated BPTT for long sequences by simulating in `tbptt_len` chunks and detaching voltage at chunk boundaries. In the current event-window training loop, each window is short enough that the full window is used as one chunk.

### Evaluation and Saved Connectivity

Validation is performed on held-out data from the same neurons, not held-out neurons. After each epoch, the script measures spike-prediction loss on held-out windows or held-out recordings, then assembles the full `[post, pre]` weight matrix with `get_connectivity_matrix()` and evaluates connectivity over all fitted neurons.

- Score per candidate edge: `abs(learned_weight)`
- Metrics: ROC AUC, average precision, precision, recall, F1
- Threshold selection: best F1 point from the validation precision-recall curve
- Final reporting: held-out window loss plus connectivity metrics over all fitted neurons

### CLI Reference

**Session and data selection**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--session` | interactive prompt | Session folder to load; if omitted, the script prompts from `LIF data/` |
| `--single-recording` | off | Use one recording instead of concatenating all recordings in the session |
| `--recording` | 0 | Recording index to load when `--single-recording` is enabled |
| `--subsample` | None | Keep only the first `N` ms after loading, mainly for quick tests |
| `--dt` | 1.0 | Bin width in ms for converting spike times to binary spike trains |
| `--val-fraction` | 0.2 | Fraction used for validation: held-out recordings when possible, otherwise held-out event windows |

**Model and optimization**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--k` | 50 | Number of nearest candidate presynaptic neurons per postsynaptic neuron |
| `--max-delay` | 8 | Number of discrete delay bins per candidate synapse |
| `--epochs` | 40 | Maximum training epochs |
| `--lr` | 0.001 | Adam learning rate |
| `--batch` | 128 | Event windows per batch |
| `--patience` | 20 | Early-stopping patience in epochs without held-out window loss improvement |
| `--l1` | 0.01 | L1 sparsity penalty on learned weights |
| `--pos-weight` | 5.0 | Positive-class weight in BCE spike loss |
| `--device` | `cpu` | Compute device: `cpu` or `cuda` |
| `--candidate-mode` | `hybrid` | Candidate proposal mode: `hybrid` or `spatial` |
| `--candidate-spatial-frac` | 0.8 | In hybrid mode, fraction of K reserved for spatial neighbors |
| `--candidate-min-lag` | 1 | In hybrid mode, minimum causal lag in bins for temporal candidates |
| `--candidate-max-lag` | uses `--max-delay` | In hybrid mode, maximum causal lag in bins for temporal candidates |

**Event-window extraction**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pre-context` | 50 | Bins of causal presynaptic history before the event |
| `--post-context` | 10 | Bins after the event to capture spike and reset dynamics |
| `--warmup` | 30 | Bins used for membrane settling before loss starts |
| `--neg-ratio` | 1.0 | Number of negative windows to sample per positive window |
| `--neg-min-dist` | 100 | Minimum distance in bins between a negative-window center and any postsynaptic spike |
