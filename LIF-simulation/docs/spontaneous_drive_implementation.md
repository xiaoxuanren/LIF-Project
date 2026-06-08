# Implementation task: noise-free self-sustaining activity (asynchronous baseline + network bursts)

## Objective
Make the LIF network produce **self-sustaining spontaneous activity with `noise_sigma = 0`**: a continuous *asynchronous-irregular* baseline (spikes between bursts) punctuated by intermittent *network bursts*. Currently, with noise off and no stimulation, the deterministic network relaxes to a subthreshold fixed point (~-61 mV, ~11 mV below the -50 mV threshold) and goes silent / flatlines. The fix is a **persistent, per-neuron, heterogeneous sub-rheobase baseline drive** plus a modest reduction of recurrent excitation. Adaptation (already in the model) terminates each burst.

Do **not** add membrane noise, do **not** lower the threshold globally, and do **not** route the baseline drive through the stimulation/`i_ext` path (see Pitfalls).

## Files in scope
- `lif_simulation/models.py` — `LIFNeuron` (add baseline field), `ExpSynapse` (excitatory scaling helper)
- `lif_simulation/network.py` — `create_clustered_network` / a post-build setup helper
- `lif_simulation/simulation.py` — confirm the baseline is NOT overwritten each step (read-only check)
- `lif_simulation/probes.py` (or a new `lif_simulation/analysis.py`) — state-segmentation helper
- the no-stimulation / spontaneous workflow entry point (e.g. `scripts/run_no_stim_validation.py` or the `workflows.py` spontaneous branch) — wire up the new config

---

## Task 1 — Add a persistent baseline-current field to `LIFNeuron`
In `LIFNeuron.__init__`, add a per-neuron baseline current that is **separate from `i_ext`** and defaults to 0:

```python
self.i_baseline = 0.0   # persistent background drive (nA-equivalent), fixed per neuron, frozen per recording
```

In `LIFNeuron.update`, add `self.i_baseline` into the existing `current_term` (alongside `i_ext`):

```python
# before:
# current_term = self.R_m * (i_exc + i_inh + self.i_h + self.i_ext - self.i_adapt)
current_term = self.R_m * (i_exc + i_inh + self.i_h + self.i_ext + self.i_baseline - self.i_adapt)
```

Also reset it appropriately: in `reset_state()`, **do NOT** zero `i_baseline` (it is structural, not dynamic state). Only zero the dynamic variables (`v`, `i_adapt`, conductances, spike history) as already done. The baseline must persist across `reset_state()` so repeated recordings of the same network keep identical drive.

## Task 2 — Assign heterogeneous baseline drive (excitatory cells only)
Add a setup helper (in `network.py`, or wherever the spontaneous workflow builds the network) that assigns each excitatory neuron a fixed baseline drawn once from a Gaussian, clipped at 0:

```python
def assign_baseline_drive(neurons, mean=0.11, sd=0.05, seed=0, excitatory_only=True):
    """Frozen per-neuron sub-rheobase drive. Rheobase ~= 0.115 for the default
    excitatory params (v_rest=-61.5, v_thresh=-50, R_m=100), so keep `mean` below it.
    The upper tail (drive > rheobase) self-ignites and seeds activity; the rest sit
    poised near threshold and fire when synaptically pushed."""
    rng = np.random.default_rng(seed)
    for n in neurons:
        if excitatory_only and n.is_inhibitory:
            n.i_baseline = 0.0
        else:
            n.i_baseline = max(0.0, float(rng.normal(mean, sd)))
```

Requirements:
- Draw **once**, store on the neuron, never re-draw per step or per recording (this "frozen disorder" replaces the role noise used to play and keeps runs reproducible/deterministic).
- Inhibitory neurons get `i_baseline = 0` by default (they are recruited synaptically).
- `mean` must stay **below rheobase (~0.115)**; the spread (`sd`) is what keeps the baseline asynchronous and supplies the self-igniting tail.

## Task 3 — Scale recurrent excitation
Add a helper (or a `create_clustered_network` parameter `exc_weight_scale=1.0`) that multiplies excitatory synaptic strength after the network is built:

```python
def scale_excitatory_weights(synapses, scale):
    for syn in synapses:
        if not syn.is_inhibitory:
            syn.weight *= scale
            syn.g_increment *= scale   # keep g_increment consistent with weight
```

Use `scale ~= 0.55-0.70` for the inference-friendly regime (modest, non-saturating bursts on a clear baseline). `scale = 1.0` gives larger, more synchronous bursts at the cost of a noisier baseline.

## Task 4 — Force noise off in the spontaneous regime
At network setup for spontaneous runs, set every neuron's `noise_sigma = 0.0`:

```python
for n in neurons:
    n.noise_sigma = 0.0
```

(The class default is non-zero; this regime relies on heterogeneity, not noise, for variability.)

## Task 5 (optional) — Adaptation / burst spacing
Burst spacing is set by the spike-frequency adaptation already in the model. Leave defaults unless culture-realistic inter-burst intervals (seconds) are needed; in that case increase the slow `tau_adaptation` and/or `adaptation_increment`. Expose them as optional setup params; do not change defaults silently.

## Task 6 — State-segmentation helper (for the inference pipeline)
Add a function that splits a recording into **burst** vs **inter-burst** windows from the population activity, so downstream inference can weight inter-burst epochs for pairwise structure and use burst onsets for directionality:

```python
def segment_states(spike_data, n_neurons, duration_ms,
                   bin_ms=5.0, burst_frac_thresh=0.12,
                   merge_gap_ms=30.0, min_burst_ms=5.0):
    """Return (burst_windows, interburst_windows) as lists of (start_ms, end_ms).
    burst = bins where active fraction (fraction of neurons firing in the bin)
    exceeds burst_frac_thresh; adjacent bursts within merge_gap_ms are merged;
    bursts shorter than min_burst_ms are dropped."""
    import numpy as np
    nb = int(np.ceil(duration_ms / bin_ms))
    af = np.zeros(nb)
    for nid, st in spike_data.items():
        if not st: continue
        b = np.floor(np.asarray(st) / bin_ms).astype(int)
        b = b[(b >= 0) & (b < nb)]
        for bb in np.unique(b):
            af[bb] += 1
    af /= n_neurons
    is_b = af > burst_frac_thresh
    raw = []
    i = 0
    while i < nb:
        if is_b[i]:
            j = i
            while j < nb and is_b[j]:
                j += 1
            raw.append((i * bin_ms, j * bin_ms)); i = j
        else:
            i += 1
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] < merge_gap_ms:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    bursts = [(a, b) for a, b in merged if (b - a) >= min_burst_ms]
    inter = []
    prev = 0.0
    for a, b in bursts:
        if a > prev: inter.append((prev, a))
        prev = b
    if prev < duration_ms: inter.append((prev, duration_ms))
    return bursts, inter
```

## Task 7 — Wire up the spontaneous workflow
In the no-stimulation / spontaneous entry point, after `create_clustered_network`:
1. `for n in neurons: n.noise_sigma = 0.0`
2. `assign_baseline_drive(neurons, mean=0.11, sd=0.05, seed=<run_seed>)`
3. `scale_excitatory_weights(synapses, 0.65)`
4. Run `simulate_network(..., stimulation_events=[], ...)` — **no stim events**.
5. Optionally call `segment_states(...)` and persist the windows alongside the spikes/voltage.

---

## Recommended default parameters
| parameter | value | notes |
|---|---|---|
| `noise_sigma` | `0.0` | all neurons |
| baseline `mean` | `0.11` | must stay below rheobase ~0.115 |
| baseline `sd` | `0.05` | controls baseline density + supplies self-igniting tail |
| baseline target | excitatory only | inhibitory driven synaptically |
| `exc_weight_scale` | `0.65` | 0.55-0.70 inference-friendly; 1.0 = bigger/synchronous bursts |
| burst threshold | `0.12` active frac | for `segment_states`, 5 ms bins |

## Pitfalls — do NOT
- **Do NOT** implement the baseline by injecting full-duration stimulation events. `simulation.py` overwrites `neuron.i_ext` every step from the active stimulation set (`neuron.i_ext = active_stims.get(...)`), so anything placed there is clobbered when no event is active. Use the separate `i_baseline` field added in Task 1. Confirm `i_ext` and `i_baseline` are distinct and both summed in `current_term`.
- **Do NOT** add Gaussian membrane noise or lower `v_thresh` globally — both defeat the purpose (reproducible determinism; preserved threshold gap).
- **Do NOT** re-draw the per-neuron baseline each step or each recording. Draw once with a fixed seed; keep it frozen so the data is deterministic and reproducible.
- **Do NOT** zero `i_baseline` in `reset_state()`.

## Acceptance criteria (validation)
Run the spontaneous workflow for >= 3000 ms with the recommended params and a fixed seed, then verify:
1. **No flatline:** spikes occur in the final 1000 ms of the recording.
2. **Live asynchronous baseline:** with `segment_states`, the inter-burst windows contain the majority of spikes (expect ~70-85%), at a population rate of roughly 2-4 Hz/neuron; the per-bin active fraction between bursts is > 0.
3. **Bursts present:** at least a few windows cross the burst threshold over 3 s.
4. **Determinism:** re-running with the same seed reproduces the spike trains exactly (no RNG in the per-step update once `noise_sigma = 0`).
5. **Separation works:** `segment_states` returns both non-empty burst and inter-burst lists.

If criterion 1 fails (still flatlines): `mean` is too low or `exc_weight_scale` too low. If bursts saturate (active fraction > ~0.4, criterion 2 baseline too sparse): lower `exc_weight_scale`. If too bursty overall: lower `mean`.

## Related (separate task, not in this change)
For the voltage-augmented supervision that consumes this data: mask spikes at native dt (0.1 ms), supervise/train at 1 ms via **mean-pooling** the cleaned trace (not striding), and set the coupling-kernel window (`max_delay`) to ~20 ms in real time (rescaled if dt changes) so it spans the full EPSP/IPSP support. Keep that out of this PR.
