# Voltage-model surrogate-FDR calibration — characterized limitation

Scope: voltage-augmented learned-LIF, session `20260523_084407` (461 neurons, all 20
recordings, K=100, burst-excluded), evaluated at the achievable target **true-FDR = 0.11**
(0.5% is off this model's frontier — unreachable at any cutoff). All cells reuse the **same**
observed model, so the ranking metrics are identical across cells (thresholding cannot move
the ranking): **AUC 0.916 / AP 0.760 / inhibitory-AUPRC 0.076** — confirmed equal in every
run; any deviation would be a bug, and none was observed.

## Null-type × surrogate-epochs (target_fdr = 0.11)

| null            | surrogate_epochs | power (edge-collapse) | calibration (FP-floor match) | cutoff | true_prec | true_recall | true_FDR | FP    |
|-----------------|:----------------:|:---------------------:|:----------------------------:|:------:|:---------:|:-----------:|:--------:|:-----:|
| circular_shift  | 1                | 0.034 (good)          | 0.295 (bad)                  | 0.054  | 0.125     | 0.920       | 0.875    | 17315 |
| circular_shift  | 12               | *not completed*       | *not completed*              |  —     |  —        |  —          |  —       |  —    |
| interval_jitter | 1                | 0.166 (good)          | 0.326 (bad)                  | 0.083  | 0.176     | 0.896       | 0.824    | 11224 |
| interval_jitter | 12               | 0.752 (bad)           | 0.952 (good)                 | 1.012  | 0.920     | 0.047       | 0.080    | 11    |

`power` = median(null score @ true-edge candidates) / median(observed @ true-edge candidates);
small ⇒ true edges collapse under the null. `calibration` = (null non-edge p95)/(observed
non-edge p95); ~1 ⇒ the null's false-positive floor matches the real one. Want both — see below.

**Oracle frontier reference (sim-only, label-aware ceiling) at true-FDR 0.11:** cutoff 0.328,
precision **0.890**, recall **0.611**. (5%→0.951/0.232; 20%→0.800/0.714; 0.5%→unreachable.)

The fourth cell (`circular_shift`, 12 epochs) was launched but its process died mid-fit and was
not re-run (per decision). It is not required for the conclusion: circular shift rigidly
de-correlates each neuron (shifting spikes *and* voltage), so it destroys true edges (low power
at any epoch) **and** the common-input that creates the FP floor (low calibration at any epoch);
more epochs cannot make a structureless null match the floor. The decisive contrast is the
interval_jitter epochs axis (rows 3–4), where calibration and power trade off directly.

## Conclusion (final, do not describe as "fixed")

The model's ranking is good (AUC 0.916, unchanged across all variants) but the unsupervised
cutoff cannot reach the achievable frontier (0.89 precision / 0.61 recall at 11% true-FDR,
oracle, sim-only). `interval_jitter` fixes the null *type* vs `circular_shift` (FP −35–46%,
precision +41–50% at matched target). But power (edge-collapse) and calibration (FP-floor match)
trade off and can't both hold: at `surrogate_epochs=1` calibration is 0.33 (FP explosion); at
matched `surrogate_epochs=12` calibration is 0.95 but the null re-learns true edges so power
blows up (0.17→0.75) and recall collapses to 4.7%. Cause: the null keeps the real voltage to
preserve common input, but the real voltage carries the connectivity signal, so common input and
edges are inseparable in voltage. Confirmed on both the window axis and the epochs axis.

This is a **characterized limitation of the unsupervised threshold**, not a bug. The frontier
itself (the ranking) is the lever to improve — see the model-side sparsity / conductance-inhibition
work, which targets the score distribution rather than the cutoff.

## Recommended voltage configuration

- `--surrogate-null interval_jitter --jitter-bins 10` with `surrogate_epochs` matched to the
  observed model — the best *null type* and the smallest full-collapse window.
- **Explicit caveat:** on the voltage model the unsupervised cutoff does **not** reach the oracle
  frontier; treat the surrogate operating point as approximate and report threshold-free AUC/AP
  alongside it. For a fixed true-FDR on simulated data, the oracle cutoff is the reference.
- Global defaults are unchanged (`circular_shift`, `surrogate_epochs=1`); these recommendations
  apply to voltage runs specifically and are not new defaults.
