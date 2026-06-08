---
name: lif-threshold-burst-comparison
description: 'Use when working on learned-LIF threshold selection, surrogate_fdr vs oracle_f1, label-leak questions, circular-shift surrogate calibration, burst-included vs burst-excluded comparisons, all-recordings runs, event-anchor comparisons, clean output tagging, or side-by-side connectivity metric summaries.'
argument-hint: 'Describe the thresholding, burst comparison, or all-recordings inference task.'
user-invocable: true
---

# Learned-LIF Thresholding And Burst Comparison

## When To Use
- User asks whether `surrogate_fdr` leaks ground-truth labels.
- User asks how a connectivity threshold is selected.
- User wants `oracle_f1` vs `surrogate_fdr` explained clearly.
- User wants an all-recordings run that includes or excludes detected burst windows.
- User wants a direct `inclbursts` vs `exclbursts` comparison.
- User wants clean output tags for oracle-first and surrogate-follow-up workflows.
- User wants threshold, FDR, AUC, AP, F1, precision, recall, or selected-edge comparisons.

## Ground Rules
- Distinguish threshold selection from post-threshold evaluation.
- `oracle_f1` is retrospective and label-aware.
- `surrogate_fdr` threshold selection is label-free: it uses observed learned scores plus surrogate null score sets.
- Labels can still be used afterward for evaluation metrics such as AUC, AP, F1, precision, and recall.
- Always verify what “all recordings” means from the current session contents before naming or summarizing a run.
- Always report whether exclusions mean only saved stimulation-onset bins or also detected network-burst windows.
- If a task finishes a meaningful comparison, ablation, sweep, or validated run, append a compact entry to the root `EXPERIMENT_LOG.md` using the existing template.

## Primary Code Anchors
- `lif_inference/learned_lif_connectivity.py`
- `lif_inference/voltage_augmented_learned_lif_connectivity.py`
- `learned_lif_outputs/`
- `EXPERIMENT_LOG.md`
- `SCRIPTS_SUMMARY.md`

## Threshold Selection Procedure
1. Inspect `select_connectivity_threshold(...)`.
2. Confirm which branch is active:
   - `oracle_f1`: chooses the threshold from labels and scores with a precision-recall/F1 sweep.
   - `surrogate_fdr`: chooses the loosest threshold whose estimated FDR is below the target.
3. Inspect `estimate_surrogate_connectivity_score_sets(...)` to confirm how null score sets are built.
4. State explicitly whether labels participate in threshold selection or only in evaluation.
5. If the user asks about leak, answer in these terms:
   - `oracle_f1` does leak labels into threshold choice because it is intentionally retrospective.
   - `surrogate_fdr` does not leak labels into threshold choice because it uses surrogate null scores instead.

## Surrogate Calibration Procedure
1. Verify the target run configuration from the saved checkpoint or CLI command:
   - session
   - duration / all-recordings count
   - candidate settings
   - anchor mode
   - burst exclusion mode
2. Confirm the null calibration path:
   - circularly shift spikes within recording boundaries
   - fit lightweight surrogate models
   - flatten surrogate candidate-edge scores
   - compute expected null selections above each candidate threshold
   - choose the highest-recall threshold that still satisfies the FDR target
3. Report:
   - target FDR
   - surrogate model count
   - edges per surrogate model
   - selected threshold
   - expected null selections
   - estimated FDR

## All-Recordings Burst Comparison Procedure
1. Verify the current recording count and total duration from the session before launching runs.
2. Use the same hyperparameters for both comparison arms except for burst exclusion.
3. Use clean, duration-accurate tags:
   - `..._inclbursts_allrec`
   - `..._exclbursts_allrec`
4. If the burst-excluded arm is requested, toggle detected-burst exclusion explicitly.
5. After both runs finish, compare both oracle and surrogate outputs side by side.
6. Always report:
   - recording count
   - total duration
   - validation strategy text
   - excluded bin count
   - detected burst window count
   - oracle threshold, AUC, AP, F1, precision, recall, selected edges
   - surrogate threshold, estimated FDR, AUC, AP, F1, precision, recall, selected edges
7. If the run produced a result worth preserving for later sessions, add a concise `EXPERIMENT_LOG.md` entry with goal, parameters, validation run, key result, interpretation, and next step.

## Oracle-First Naming Procedure
1. Keep the base tag free of `fdr0005` if the run is intended to be oracle-first.
2. Remember that figure names append `_oracle_f1` or `_surrogate_fdr` after the base output name.
3. Explain that an `fdr0005` string in an oracle filename usually came from the user-chosen tag, not from oracle thresholding itself.

## Comparison Reporting Standard
- Do not summarize with AUC alone.
- Report AUC, AP, F1, precision, recall, selected edges, threshold, and estimated FDR together.
- Note whether burst inclusion improved ranking AUC while harming thresholded precision/F1.
- Separate these concepts clearly:
  - ranking quality
  - thresholded edge-call quality
  - exclusion policy
  - candidate coverage

## Command Pattern
Use the spike-only learned-LIF CLI with matched settings for both comparison arms, changing only the burst-exclusion flag and output tag.

Example pattern:

```powershell
python -m lif_inference.learned_lif_connectivity \
  --session "LIF data/<session>" \
  --output-tag "cmp<duration>s_spike_postcentered_k100_<incl|excl>bursts_allrec" \
  --k 100 \
  --epochs 6 \
  --batch 128 \
  --max-delay 8 \
  --l1 0.01 \
  --pos-weight 5.0 \
  --dt 1.0 \
  --device cpu \
  --candidate-mode hybrid \
  --candidate-spatial-frac 0.8 \
  --candidate-min-lag 1 \
  --candidate-max-lag 8 \
  --threshold-mode adaptive \
  --connectivity-threshold-mode oracle_f1 \
  --event-anchor-mode post \
  --n-threshold-surrogates 4 \
  --surrogate-epochs 2 \
  --surrogate-patience 1
```

Add `--exclude-detected-bursts` only for the excluded arm.

## Output Expectations
- Give a direct answer on leak vs no leak.
- Cite the actual threshold-selection branch in code.
- Use exact artifact names when summarizing finished runs.
- Mention whether the comparison is all-recordings and what that currently means numerically.
- State whether `EXPERIMENT_LOG.md` was updated.
- Update `SCRIPTS_SUMMARY.md` only if behavior, defaults, outputs, or workflow changed.