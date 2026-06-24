# Connectivity-inference talk figures (v2) — run 20260523_084407

Regenerated from `connectivity_eval_export_20260523_084407.npz` (voltage AUC 0.916) with verified
sign conventions, one consistent edge set, and crisp cells. Every number below was recomputed from
the data, not carried over from older slides.

## Global conventions (apply to every figure)
- **Edge set (single denominator everywhere):** the **candidate edges** — **N = 46,100 (2,683 true: 2,275 exc / 408 inh, 5.6:1)**. Printed on every figure.
- **Sign convention (verified):** excitatory = `true_W_signed > 0`, inhibitory = `true_W_signed < 0`; predicted = `conn_signed_*`. Assertion: exc true edges land predicted-positive (voltage 99.7%, spike 98.9%); inhibitory land predicted-negative (**voltage 79.9%, spike 69.9%**). No flip.
- **Colours:** excitatory = red `#C0392B`, inhibitory = blue `#2471A3`, non-edge = grey `#9AA0A6`; TP = green `#27AE60`, FP = orange `#E67E22`, FN = slate `#5D6D7E`.
- **Visibility:** heatmaps `interpolation='nearest'`; subset matrices use bordered `pcolormesh` (light-grey cell borders); colour normalized to the **90th percentile of |W|** with `TwoSlopeNorm(vcenter=0)`; faint-grey zero so weak edges read.

## ⚠️ Recomputed numbers that differ from earlier slides (these figures use the recomputed values)
1. **Inhibitory edges land predicted-negative only ~80% (voltage) / 70% (spike)** — older slides assumed ~90%. The convention is correct; the rate is just lower.
2. **Inhibitory AUC = voltage 0.80 / spike 0.69** (positives = inh, detector = −signed, all candidates) — **not** the old 0.66/0.46 (an over-pessimistic definition; 0.46 is below chance).
3. **Exc:inh = 5.6:1** (2275/408), not 6:1.

## Headline numbers
| metric | spike-only | voltage | CCG |
|---|---|---|---|
| overall AUC / AP | 0.883 / 0.766 | 0.916 / 0.760 | 0.720 / 0.246 |
| excitatory AUC | 0.979 | 0.988 | 0.802 |
| inhibitory AUC | 0.689 | 0.803 | 0.259 |
| surrogate-FDR threshold → true FDR | 0.193 → 0.112 | 0.146 → **0.64** | (oracle-F1 0.016) |
| oracle threshold → true FDR | 0.446 → 0.005 | 1.481 → 0.033 (0.005 unreachable) | — |
| exc / inh recall @ matched 11% FDR | 0.75 / 0.04 | 0.70 / **0.13** | 0.03 / 0.00 |

**Story:** all methods recover excitation well and inhibition poorly; **voltage best at inhibition** (still only ~13%); **CCG weakest** and sign-blind. The voltage surrogate threshold over-selects badly (true FDR 0.64 vs the oracle's 0.033) — a characterized limitation of the unsupervised cutoff.

---

## 0 · Simulated circuit & activity

### Subnetwork circuit (clusters 0–2, 48 neurons)
![](v2_subnetwork_circuit.png)
*Spatial-layout node-link: neurons excitatory (red) / inhibitory (blue), directed arrows = synapses (205 within the subset). Inhibitory cells wire locally; excitatory project across clusters.*

### Recovered subnetwork circuit — spike-only (detected edges)
![](v2_subnetwork_circuit_recovered_spike.png)
*Same subnetwork, but arrows are the spike-only model's **detected** synapses coloured by predicted sign (124 exc / 15 inh detected; TP 119 / FP 20, thr 0.19); nodes keep their true E/I. Compared with the ground-truth circuit above, the excitatory backbone is recovered while most inhibition is missed.*

### Recovered subnetwork circuit by correctness — spike-only
![](v2_subnetwork_circuit_recovered_correctness.png)
*Same recovered circuit coloured by correctness: **TP recovered** (green, 119), **FP false** (orange, 20), **FN missed** (slate dashed, 86; thr 0.19). The dense green is the recovered excitatory backbone; the dashed slate (missed edges) clusters around the inhibitory cells.*

### Full circuit (461 neurons)
![](v2_fullnetwork_circuit.png)
*Same style, whole network: 461 neurons (366 exc / 95 inh), 2,780 directed synapses; spatially-clustered wiring.*

### Subnetwork raster — 48 neurons, 60 s
![](v2_subnetwork_raster.png)
*Spike raster of the subnetwork over the full 60 s recording (exc red below the divider / inh blue above), bold ticks, population active-fraction below. Inhibitory cells fire mainly in the synchronized stim bursts (dashed).*

### Whole-network raster — 461 neurons, 60 s
![](v2_fullnetwork_raster.png)
*Full 461-neuron raster, 60 s, bold spikes; five stim bursts (dashed) punctuate sparse spontaneous firing — the input the inference is fit to.*

### Subcluster voltage traces — around a stimulus burst
![](v2_subcluster_voltage_traces.png)
*Raw membrane voltage for example neurons spanning subclusters 0–2 (exc red / inh blue) around a stimulus burst (~11.7 s); ▼ = spikes. The subthreshold dynamics + coordinated burst spiking the voltage-augmented model is fit to.*

### Subcluster voltage traces — spontaneous (inter-burst)
![](v2_subcluster_voltage_spontaneous.png)
*The same subcluster neurons in two inter-burst windows (3–5.5 s and 14–16.5 s, each >3 s from any stim burst). Without burst drive the cells show low-rate, irregular spontaneous spiking on noisy subthreshold fluctuations — the asynchronous between-burst regime.*

---

## 1 · Score separation
![](v2_score_separation.png)
*Per-class density of predicted signed weight (voltage), each class area-normalized to 1. Exc median +0.45, inh −0.07, non 0.00; negative half shaded so inhibitory mass reads as negative.*

![](v2_score_separation_counts.png)
*Same distribution **without normalization** (raw counts, log-y). Shows the true class imbalance — 43,417 non-edges vs 2,275 exc vs 408 inh — that area-normalization hides.*

## 2 · Predicted vs true (signed)
![](v2_pred_vs_true_scatter.png)
*Per-candidate predicted vs true signed weight. Excitatory upper-right, inhibitory lower-left (verified); non-edge band shaded. Sign-agreement annotated: voltage 99.7% exc / 79.9% inh, spike 98.9% / 69.9%.*

## 3 · E/I detected edges @ matched 11% true-FDR
![](v2_EI_detected_bars.png)
*Recall by sign at a matched 11% true-FDR (fair cross-method comparison). exc: spike 0.75 / voltage 0.70 / CCG 0.03; inh: spike 0.04 / voltage 0.13 / CCG 0.00. Excitation recovered, inhibition missed; voltage best at inhibition.*

## 4 · Per-class precision–recall
![](v2_per_class_PR.png)
*PR for detecting each class vs all other candidates. Excitatory: all methods strong (AUC 0.98/0.99/0.80). Inhibitory: voltage 0.80 > spike 0.69 ≫ CCG 0.26 (AP 0.077/0.037/0.008) — inhibition is hard for everything.*

## 5 · Cluster-subset connectivity (clusters 0–2, 48 neurons, same ordering)
![](v2_cluster_subset.png)
*Signed weights (90th-pct saturated). Ground truth (signed), spike-only (signed), CCG (magnitude, no sign). Crisp bordered cells.*

![](v2_cluster_subset_detected.png)
*Same subset, **detected edges only** (thresholded), coloured by sign. Truth 160 exc / 45 inh; spike-only detects 124 exc / 15 inh @thr 0.19; CCG 172 edges (no sign, over-detects).*

![](v2_cluster_subset_correctness.png)
*Same subset, detected edges coloured by **correctness**. Spike-only TP 119 / FP 20 / FN 86; CCG TP 60 / FP 112 / FN 145 (CCG's many FP are visible).*

## 6 · Full-network adjacency, all four methods (461 neurons, cluster-blocked)
![](v2_adjacency_all4_full.png)
*True / spike-only / voltage (signed) + CCG (magnitude). Global block structure; cell-level detail is in the 48-node subset above.*

## 7 · Full-size single-method matrices
| | | | |
|---|---|---|---|
| ![](v2_matrix_true.png) | ![](v2_matrix_spike.png) | ![](v2_matrix_voltage.png) | ![](v2_matrix_ccg.png) |
*Each method full-size (461, cluster-blocked): true, spike-only, voltage (signed), CCG (magnitude).*

## 8 · Surrogate-vs-oracle recovery (TP/FP/FN)
![](v2_surr_vs_oracle_spike.png)
*Spike-only: 48-node + full-461 recovery at surrogate (0.193 → FDR 0.112) and oracle (0.446 → 0.005). Well-calibrated.*

![](v2_surr_vs_oracle_voltage.png)
*Voltage: surrogate (0.146 → **FDR 0.64**, 4078 FP) vs oracle (1.481 → 0.033). The surrogate over-selects badly — 0.005 is unreachable on the voltage frontier.*

## 9 · Detected edges by predicted sign (full 461)
![](v2_detected_by_sign.png)
*Detected edges coloured red/blue by predicted sign, spike-only and voltage, at surrogate and oracle thresholds (CCG omitted — no sign).*

---

## 10 · Voltage's sign-recovery advantage & detected-edge operating points

### How voltage helps sign recovery
![](v2_sign_recovery_improvement.png)
*Left: sign-agreement among the 2,683 true edges, spike-only (hatched) vs voltage (solid). Voltage lifts **inhibitory** sign recovery 69.9 → 79.9% (**+10 pp**) and excitatory 98.9 → 99.7%. Right: true edges whose sign voltage **corrects** (spike-wrong → voltage-right) minus the reverse — net **+20 excitatory, +41 inhibitory (+61 total)**, two-thirds inhibitory. This is the mechanism behind voltage's inhibitory AUC/recall gains.*

### Detected edges by correctness — oracle-F1 vs surrogate (circuits)
![](v2_detected_circuits_by_correctness.png)
*Subnetwork (clusters 0–2) detected edges, TP green / FP orange / FN slate, at each model's F1-optimal threshold (oracle-F1) and its unsupervised surrogate-FDR threshold. Spike-only is well-calibrated (surrogate ≈ oracle-F1: subnet TP 119/125, FP 20/26). **Voltage at its surrogate floods with false positives** — subnet FP 30 → 190 — the characterized over-selection (global FDR 0.64).*

### Detected edges by correctness — oracle-F1 vs surrogate (matrices)
![](v2_detected_matrices_by_correctness.png)
*The same four operating points as cluster-blocked 48×48 matrices (post × pre). The voltage-surrogate panel's orange (FP) spread across off-diagonal blocks shows the over-selection the circuit view also makes visible.*

> **Oracle-F1 thresholds** (F1-maximizing on the true labels): spike-only **0.186** (F1 0.750, TP 1779 / FP 284 / FN 904, rec 0.663), voltage **0.280** (F1 0.757, TP 1873 / FP 391 / FN 810, rec 0.698). These are **distinct** from the near-zero-FDR "oracle" of §8 (0.446 / 1.481). At oracle-F1 voltage detects more true edges at higher recall; the gap blows up only at the voltage **surrogate** (global FP 4,111, FDR 0.64), while the spike surrogate sits essentially at its own oracle-F1 (0.193 ≈ 0.186).
