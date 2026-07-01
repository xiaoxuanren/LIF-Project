# Inhibition-Detection Improvement Plan

*Working plan for raising inhibitory **edge detection** out of chance (AUC ~0.48) in the
learned-LIF pipeline, borrowing concretely from the four analyzed papers and the related-work
set. Read `00_LIF_pipeline_context.md`, `01_paper_analyses_and_relevance.md`, and
`02_related_work_and_suggested_reading.md` first. This file is meant to live in the repo
(commit it alongside `EXPERIMENT_LOG.md`) so it can be edited across Claude Code sessions.*

---

## 0. Purpose and how this doc is used

This is a sequenced, decision-gated plan, not a wishlist. Each phase states a **hypothesis**,
the **exact change**, **what to run**, a **decision rule** (what result advances vs kills the
idea), and the **risk**. Phases are ordered cheapest-and-most-diagnostic first, so we learn
where the ceiling actually is before spending effort on the expensive model change.

**Iteration loop (the working protocol):**

1. The plan author (Claude, in chat) writes a scoped Claude Code prompt for the next phase
   (prompts are archived in the Appendix of this file).
2. Claude Code implements, runs, and writes a results entry into `EXPERIMENT_LOG.md` using the
   existing template, saves any figures/CSVs under the run's diagnostics directory, commits to
   the working branch, and pushes.
3. The author pulls, reads the new `EXPERIMENT_LOG.md` entry plus artifacts, and either advances
   to the next phase or modifies the plan, then writes the next prompt.

**Working branch:** `feature/inhibition-detection`. Each phase appends one or more
`EXPERIMENT_LOG.md` entries and (where code changes) lands behind an explicit CLI flag so the
existing baselines stay runnable and unchanged.

**Ground-truth numbers to preserve exactly** (from `00_LIF_pipeline_context.md`; do not round or
let them drift): benchmark 461 neurons (366 E / 95 I), 2,780 directed synapses. Learned-LIF edge
AUC 0.883 spike-only, 0.916 voltage-augmented; CCG baseline ~0.742. Inhibitory **sign** recovery
0.70 vs CCG 0.03. Voltage augmentation ~3× inhibitory recall. Standing open problem: inhibitory
**detection** near chance, AUC ~0.48. Repo runs (e.g. session `20260523_084407` voltage AUC
0.904; `20260604_023028` vlam1 AUC 0.846) are different sessions — cite the exact run from
`EXPERIMENT_LOG.md`, never mix the two.

---

## 1. Diagnosis (the hypothesis the whole plan tests)

**Established (read directly from code):** the *generative* model uses conductance-based synapses
with driving force — `i_exc = g_exc·(e_exc − v)`, `i_inh = g_inh·(e_inh − v)`
(`lif_simulation/models.py:145–146`). The *inference* model uses a **current-based** kernel,
`I_syn = Σ_k w_k·(delay-weighted pre_spikes) + bias` fed into `v = alpha·v + I_syn`
(`lif_inference/voltage_augmented_learned_lif_connectivity.py:851, 896`). Detection AUC is
computed on `|effective_W|` (`connectivity_metrics.py:209`; voltage `evaluate_connectivity:1563`),
while sign accuracy is computed separately over true edges (`:1605`). The voltage target is
z-scored per neuron (`preprocess_voltage_recording:196`), which removes absolute Vm level. The
voltage loss is a uniform `smooth_l1` (`:1013`, `:1229`); L1 is sign-agnostic (`:1019`).

**Working hypothesis (mine, to be tested — not established):** inhibitory detection sits at
chance because a current-based kernel is the wrong functional form for a voltage-gated effect.
A real IPSP's amplitude scales with `(v − e_inh)`; a fixed-current kernel must fit the *average*
of that voltage-dependent effect, biasing recovered `|W_inhibitory|` small and noisy, so it fails
a `|W|` threshold even when its **sign** is right (which is why sign survives at 0.70 while
detection is 0.48). Voltage augmentation helps (0.883 → 0.916, ~3× recall) because the Vm channel
carries the IPSP, but z-scoring plus the current-based form cap how much of it is usable. Two
contributing mechanisms compound this: (a) the inhibitory spike-timing gradient is intrinsically
weak (Banerjee 2016 — spikes ride the rising phase of IPSPs, ∂t_out/∂t_in small), and (b) the
detection null pools inhibitory and excitatory `|W|` together, so the large excitatory spread sets
the bar inhibition must clear.

The plan attacks (b) and the loss first (cheap, model-free), then the functional form (the real
bet), and keeps falsification in view at every step: P0 can show the ceiling is upstream
(candidate coverage), in which case no model change helps.

---

## 2. Phased plan

| Phase | Borrow from | What | Touches | Cost | Gate |
|---|---|---|---|---|---|
| **P0** | internal diagnostic | E/I candidate coverage + type-restricted AUC | analysis only | ~free | informs everything |
| **P1a** | STR FDR-power (#1), Banerjee (#2) | post-hoc type-stratified thresholding | analysis only | low | does stratifying help at all? |
| **P1b** | same | wire type-stratified surrogate-FDR into pipeline | `connectivity_metrics.py` + null pool | low | does it survive a real null? |
| **P0b** | internal diagnostic | depression-vs-functional-form attribution (⟨R⟩/rate decomposition of I-only AUC) | analysis only | ~free | **depressing sessions only**; depression vs form |
| **P2** | STR subthreshold (#1), Banerjee (#2) | reshape voltage + L1 loss for inhibition | loss fns (2 flags) | low | recall up without precision collapse? |
| **P3** | Zou/Destexhe (#3), Fiete & Seung (#D10) | conductance-based inference synapse | forward `I_syn` + Dale latent | high | oracle-type prototype must beat current form |
| **P4** | GLM-CCG (#A1), Hafizi (#A2), STR linear (#1) | real baselines + linear ablation | new modules | medium | clears the straw-man bar |
| **P5** | STR STP (#1), internal sim STP | short-term-depression-aware inference synapse (per-pre resource trace R̂) | forward `I_syn` + STP state | high | **depressing sessions**; fixed-STP oracle prototype must beat static-weight form |
| **P6** | internal | sign-agnostic candidate score | `candidate_selection.py` | low | only if P0 shows I-coverage gap |

Run order (general): **P0 → P1a → P0b (depressing sessions) → P2 → (P3 ∥ P5 by P0b outcome)**, with
P4 and P6 slotted by what P0/P1a reveal. On a **non-depressing session** (e.g. `20260523_084407`,
confirmed no STP) P0b is moot, so the path is **P0 → P1a → P2 → P3**. P3/P5 are the scientific bets
(functional form vs short-term depression); everything before is meant to either move the metric
cheaply or tell us which bet is necessary.

---

### P0 — Measure the ceiling before touching the model

**Hypothesis.** Some fraction of the 0.48 is upstream of inference: inhibitory true edges whose
presynaptic neuron is not in the postsynaptic neuron's candidate set are unrecoverable by any
model. Reported candidate coverage (e.g. 96.5% on `20260523_084407`) is aggregated over E and I;
inhibitory coverage may be materially lower, especially under `candidate_mode=hybrid`
(`candidate_selection.py`), whose temporal slots are filled by an excitatory-only coincidence
score (`causal_pair_score:52–69`).

**Change.** None to the model. A post-hoc analysis script that, from a saved
`connectivity_<session>_<tag>.npz` (keys: `connectivity_matrix`, `true_weights`,
`neighbor_indices`; all `[post, pre]`):
- derives ground-truth presynaptic type from the **sign of `true_weights`** (each presynaptic
  neuron's outgoing nonzero weights share a sign under the simulator's Dale-respecting
  generation — assert this and report any violations);
- reports **candidate coverage split by E vs I** (fraction of true E edges, and of true I edges,
  whose presynaptic appears in the postsynaptic candidate list);
- reports **detection AUC/AP restricted to I candidates only** and **E candidates only** (this is
  the honest decomposition of the aggregate AUC and pins down today's inhibitory-only number);
- reports the **distribution of recovered `|W|` for true-I edges vs true-negative I candidates**
  (how far inhibitory signal sits above its own noise floor).

**Decision rule.** If inhibitory candidate coverage is well below excitatory (say >10 points
lower), **P6 (candidate fix) jumps ahead of P3** — the model cannot be the bottleneck for edges it
never sees. If inhibitory coverage is comparable but I-restricted AUC ≈ 0.5 while the I-edge `|W|`
distribution overlaps its null, the diagnosis in §1 holds and we proceed P1→P3.

**Log.** `EXPERIMENT_LOG.md` entry + a coverage/AUC table CSV and the `|W|`-vs-null histogram.

**Risk.** Low. Worst case it confirms what we suspect; either way it's the prerequisite number we
currently don't have.

**Status (DONE, 2026-06-30, `20260523_084407` voltage export).** No coverage gap (I 0.9423 vs E
0.9693, 2.71-point gap); I-only AUC 0.6337 vs E-only 0.9747 (AP 0.1300 vs 0.8917); inhibitory
`|W|`-vs-null overlap 0.72 vs E 0.14. **Magnitude collapse confirmed** and localized: inhibition is
recoverable in sign but not in magnitude. Not coverage, not thresholding. → proceed to the loss/form
levers. (`scripts/inhibition_diagnostics.py`; entry 2026-06-30.)

---

### P1a — Post-hoc type-stratified thresholding (no retrain)

**Hypothesis.** Pooling I and E candidates in one FDR null sets the inhibitory bar too high. A
type-conditioned threshold should let more true-I edges clear at the same global FDR target.

**Change.** None to the model or the surrogate machinery yet. Post-hoc, using ground-truth type
(label it **oracle** explicitly): build a per-type empirical null from the recovered `|W|` of
true-negative candidates of each type as a fast stand-in null, and report how many true-I edges
clear a type-stratified threshold vs the existing pooled threshold, at a matched FDR. This is a
directional check, approximate because true-negative scores leak common-input structure (the FP
diagnostic on `20260604_023028` showed within-cluster enrichment), so undersell it as such.

**Decision rule.** If type-stratification recovers a meaningful number of true-I edges with
tolerable added FP, P1b (real surrogate-based stratification) is worth the plumbing. If it does
nothing, stratification is not the lever and we skip P1b.

**Log.** Entry + a small table: I-edges selected and estimated FDR, pooled vs stratified.

**Risk.** Low. Approximate-null caveat must be stated in the entry.

**Status (DONE, 2026-06-30).** **Skipped P1b — stratification cannot fix within-type separability.**
On `20260523_084407` the inhibitory `|W|` overlaps its own null (overlap 0.72, I-only AUC 0.6337)
and the realized inhibitory FDR floor is 0.50, so no threshold (pooled or oracle type-stratified)
selects true-I edges at any tolerable FDR; at the matched FDR=0.005 both select 0. The bottleneck is
the `|W|` signal, not the bar. (entry 2026-06-30.)

---

### P1b — Type-stratified surrogate-FDR in the pipeline

**Hypothesis.** The principled circular-shift surrogate null, partitioned by presynaptic type,
gives the same benefit as P1a without the true-negative-leakage caveat.

**Change.** `select_connectivity_threshold` already supports `surrogate_fdr` and
`surrogate_fdr_per_neuron` (`connectivity_metrics.py:117, 133`). Add a `surrogate_fdr_per_type`
mode: partition both the observed candidate scores and the surrogate candidate scores by inferred
presynaptic type, run `_select_surrogate_fdr_threshold` within each type, and aggregate. First
pass uses **oracle type** to isolate the effect; a second pass uses **inferred type** (sign of the
unconstrained per-presynaptic mean `W`) to check it survives without ground truth. The surrogate
pool already carries candidate→presynaptic identity via `neighbor_indices`, so type labels attach
cleanly.

**Decision rule.** Advance if inferred-type stratification holds most of the oracle-type gain.
Keep it behind a flag; do not change the default thresholding mode.

**Log.** Entry comparing pooled vs per-type (oracle) vs per-type (inferred), reporting I-recall,
overall AUC (must stay near 0.916 voltage-augmented), and sign recovery (must stay near 0.70).

**Risk.** Inferred type is itself error-prone for low-rate I units — that error propagates into
the null. Report the inferred-vs-oracle type confusion so we know how much is type error.

---

### P0b — Depression-vs-functional-form attribution (depressing sessions only)

**Applicability.** Only for sessions whose simulator used short-term depression
(`DepressingExpSynapse`, `lif_simulation/depressing_synapse.py`; `workflows.py` records `depressing`,
`depression_params={tau_q, delta_q}`, `synapse_depression_model`). **Moot on non-depressing sessions
such as `20260523_084407`** (user-confirmed no STP), where the magnitude collapse cannot be a
depression artifact and the suspects reduce to functional form (P3) and voltage-loss weighting (P2).

**Hypothesis.** When depression is on, the resource `R` is per-presynaptic-neuron, so a presynaptic
unit's time-averaged efficacy ≈ `nominal_weight × ⟨R_j⟩`, which falls steeply with its firing rate.
A static-weight inference model recovers that *effective* efficacy, so high-rate units (including
fast-spiking inhibitory cells) get systematically shrunk — a depression artifact masquerading as an
inference failure.

**Change.** None to the model. Post-hoc, reusing the P0 export + the session spike trains: compute
per-neuron firing rate and `⟨R_j⟩` from the simulator's resource recursion (on spike: release `R`,
`R *= (1−delta_q)`; each ms `R += (1−R)·dt/tau_q`) under the recorded (or, if unrecorded, explicitly
*assumed default*) `tau_q/delta_q`. Then: (a) Pearson/Spearman of recovered `|W|` with `⟨R_j⟩` and
with rate, split E/I (positive `|W|`–⟨R⟩ / negative `|W|`–rate for true-I edges is the depression
signature); (b) **the disambiguator** — decompose the I-only separation AUC by `⟨R⟩`/rate tertile:
if low-⟨R⟩ (high-rate) I collapses while high-⟨R⟩ (low-rate) I separates → depression dominates; if
even low-rate I fails → functional form dominates; (c) signed `weight_corr` of recovered weights vs
`true_weights` and vs `true_weights · ⟨R_j⟩` (a large rise with the effective target quantifies
uncorrected depression). Lands as a flag-guarded mode of `scripts/inhibition_diagnostics.py`
(`--depression-attribution`).

**Decision rule.** Depression ON + low-rate-I separates but high-rate-I collapses (+ strong
`|W|`–⟨R⟩ corr) → **P5** (depression-aware synapse, STP fixed-from-known) is the priority treatment,
candidate to fuse with P3; P2 loss-reshaping alone is likely insufficient. Depression ON but
I-collapse rate-independent → functional form dominates, P2 then P3. Depression OFF/unrecorded → not
implicated; go P2 → P3 (and check the canonical 0.48-benchmark session's `depressing` flag
separately).

**Log.** Entry + the `|W|`-vs-⟨R⟩ scatter (colored by E/I) and the per-tertile I-AUC bar chart.

**Risk.** Low. Approximate where `⟨R⟩` assumes default STP params (unrecorded session); label as a
sensitivity check.

---

### P2 — Reshape the voltage and L1 loss toward inhibition

**Hypothesis.** A uniform voltage `smooth_l1` is dominated by large excitatory excursions, so IPSP
deflections barely register; uniform L1 erases the already-underestimated inhibitory weights.
Concentrating voltage supervision where synapses are identifiable (STR's subthreshold trick) and
where inhibition is visible (hyperpolarized epochs — Banerjee's remedy, loss-side) should raise
inhibitory recall.

**Change.** In `compute_voltage_augmented_event_loss` / `compute_voltage_augmented_continuous_loss`:
- **Hyperpolarization-weighted voltage term [IMPLEMENTED 2026-06-30]:** weight the voltage residual
  by `1 + γ·relu(−vt)` where `vt` is the normalized, per-neuron baseline-subtracted target Vm, so
  below-baseline (IPSP) bins contribute more gradient (implemented as a hyperpolarization-weighted
  mean `Σ w·ℓ / Σ w`). Flag `--voltage-hyperpol-gamma` (default 0.0, exact current behavior).
- **Subthreshold concentration (STR):** optionally restrict/upweight the voltage term on
  non-refractory subthreshold bins (spike neighborhoods are already masked; this further
  emphasizes the clean-kernel regime). Flag `--voltage-subthreshold-only`. *(not yet implemented)*
- **Asymmetric L1 [IMPLEMENTED 2026-06-30, oracle]:** scale (e.g. zero) the L1 penalty on
  inhibitory candidates so sparsity does not preferentially kill the edges we are trying to detect.
  Flag `--l1-inhibitory-scale` (default 1.0); first pass uses **oracle** presynaptic type (sign of
  outgoing `true_weights`), inferred-type is a follow-up. *My inference, not from a paper — strictly
  an ablation.* Both flags default to no-op and leave the surrogate machinery untouched.

**Decision rule.** Advance any sub-change that raises inhibitory recall without dropping overall
AUC below the 0.916 voltage-augmented baseline (or sign below 0.70) by more than a small margin.
Run each sub-change as an isolated ablation against a fixed strong session; do not stack them
before each is individually characterized.

**Log.** One entry per sub-change (Goal/Hypothesis/Files/Parameters/Validation/Key result/
Interpretation/Next step), each reporting the inhibitory-restricted AUC and recall from P0's
decomposition, plus the overall guardrail metrics.

**Risk.** Low-moderate. Over-weighting hyperpolarized bins could trade excitatory precision; the
isolated-ablation discipline catches that.

**Status (DONE, 2026-07-01, both ablations run on restored `20260523_084407`).** Flags implemented +
unit/smoke-tested (2026-06-30), then run against a **matched γ=0/scale-1 baseline** (identical code/
config; reproduced the normal pipeline: overall AUC 0.9085, E-only 0.9544, I-only 0.6823, sign
0.9687 — validating the comparison). **Neither loss-reshaping lever moved I-only separation.**
Ablation B (`--voltage-hyperpol-gamma 2.0`): I-only 0.6823 → 0.6701 (flat/down) while E-only
collapsed 0.9544 → 0.5874 and overall → 0.6121, sign → 0.4413 (γ=2 over-weights IPSP bins ~7× and
abandons excitatory PSP fitting; all guardrails fail). Ablation A (`--l1-inhibitory-scale 0.0`,
oracle): I-only 0.6823 → 0.6877 (+0.005, within RNG noise), no collateral (overall 0.9106, sign
0.9668) — L1 is a negligible fraction of the objective, so removing it changes nothing.
**Verdict → commit to P3**: the cheap model-free levers are ruled out; evidence points to the
current-based functional form as the inhibitory ceiling (§1). Optional cheap pre-step: a
`--voltage-hyperpol-gamma {0.25,0.5,1.0}` sweep to fully close the hyperpol lever. (entries
2026-07-01; compare vs the matched baseline I-only 0.6823, not the P0 export 0.6337.)

**Hypothesis (the core test of §1).** Replacing the current-based kernel with a conductance-based
drive that matches the generative model removes the functional-form bias and lets the voltage
channel constrain inhibition directly, lifting inhibitory detection materially above chance.

**Change.** Replace `I_syn = Σ w·s + bias` with
`I_syn = gE(t)·(E_E − v) + gI(t)·(E_I − v)`, where `gE`/`gI` are nonnegative, delay-filtered,
weighted sums of presynaptic spikes routed by each presynaptic neuron's **type latent** — which
unifies this with Dale: the presynaptic sign logit becomes the gE-vs-gI routing gate (hard or
annealed assignment), not a `tanh` multiplier on a current. Reversal potentials `E_E`, `E_I` are
expressed in the normalized voltage frame (saved `baseline_medians`/`baseline_scales`,
`preprocess_voltage_recording:201–202`, make the physical↔normalized map recoverable) or made
learnable. Training: autograd through the existing surrogate sigmoid should still apply since only
the forward `I_syn` changes; **Fiete & Seung 2006** (gradient learning by perturbing conductances
in recurrent conductance-based spiking nets) is the citation and the fallback if gradients destabilize. *Fiete-lab paper — relevant to the MIT postdoc conversation; read closely.*

**Sequencing (de-risking, learned from the Dale probe).** The `2026-06-07` Dale run dropped AUC
0.846 → 0.682 because `tanh(sign_logit≈0)·softplus(W) ≈ 0` starved the weights. So:
1. **Oracle-type prototype first.** Fix the gE/gI routing from ground-truth type and ask the
   isolated question: *does the conductance form beat the current form when types are correct?*
   This is the clean test of §1, free of type-inference confounds.
2. Only if (1) wins, couple it to inferred/annealed type assignment (Hafizi-style network-level
   type latent, P4 #A2), warm-started from the sign of the unconstrained `W` — never from a
   saturated zero logit.
3. Prototype on one strong session (e.g. `20260523_084407`) before any end-to-end claim.

**Decision rule.** The oracle-type conductance model must beat the current-based model on
inhibitory-restricted detection AUC by a clear margin while holding overall AUC ≥ 0.916
(voltage-augmented) and sign ≥ 0.70. If oracle-type conductance does **not** beat the current form,
§1's hypothesis is wrong — log that as a real negative result and pivot to P1/P2/P4 gains plus a
re-examination of where inhibitory information actually is.

**Log.** Entries for the oracle-type prototype and (if reached) the inferred-type version, with
the full guardrail metric set and the inhibitory decomposition from P0.

**Risk.** High. Biggest change; coupled synaptic/membrane dynamics, voltage-frame consistency, and
optimization stability are all live. The oracle-first gating keeps a clean falsification path and
prevents a repeat of the Dale confound (where a mechanically-correct change still hurt ranking).

---

### P5 — Short-term-depression-aware inference synapse (depressing sessions)

**Applicability / gating.** Reserved for **depressing sessions** and prioritized by **P0b**: only
worth the plumbing if P0b shows the inhibitory collapse tracks `⟨R⟩`/rate (depression artifact)
rather than being rate-independent. Moot where the simulator used no STP (e.g. `20260523_084407`).

**Hypothesis (the depression counterpart of §1's form bet).** A static-weight kernel recovers each
presynaptic unit's *time-averaged* efficacy `nominal_weight × ⟨R_j⟩`, which is rate-suppressed for
high-rate (often inhibitory) units. Giving the inference synapse its own per-presynaptic resource
trace `R̂_j(t)` — so the effective drive is `w_k · R̂_j(t) · spike` — lets the model factor the
nominal weight out of the depression envelope and recover the *structural* weight, lifting inhibitory
magnitude (hence detection) where depression is the cause.

**Change.** Add a per-presynaptic-neuron resource state `R̂_j(t)` to the forward synapse (release on
presynaptic spike, recover toward 1 with `tau_q`; drop by `delta_q` per release), and route it into
`I_syn` as `Σ_k w_k · R̂_{pre(k)}(t) · (delay-weighted spikes)`. **STP params fixed-from-known first**
(the simulator's `tau_q`/`delta_q`, labeled **oracle**) to isolate the question *does undoing
depression recover inhibitory weight?*; only if that wins, make `tau_q`/`delta_q` learnable per type.
The resource recursion mirrors the simulator's `DepressingExpSynapse`. **Candidate to fuse with P3**:
a conductance-based, depression-aware synapse (`gE/gI` driven by `R̂_j`-scaled spikes) is the joint
form if both P0b and §1 implicate their respective mechanisms.

**Decision rule.** The fixed-STP oracle prototype must beat the static-weight form on
inhibitory-restricted detection AUC by a clear margin while holding overall AUC ≥ 0.916 and sign
≥ 0.70. If it does not, depression is not the lever even on a depressing session — fall back to P3.

**Log.** Entries for the fixed-STP oracle prototype and (if reached) the learnable-STP version, with
the P0 inhibitory decomposition and the full guardrail set; report against the matched static-weight
baseline on the same depressing session.

**Risk.** High. Adds coupled synaptic state and another optimization surface; oracle-first gating and
the isolated-ablation discipline keep falsification clean. Type-error in learnable-STP propagates as
in P1b/P3 — report inferred-vs-oracle type confusion if types are inferred.

---

### P4 — Real baselines and a linear ablation (defends the method)

**Hypothesis.** Not an improvement to the inference script, but a reviewer will reject a
comparison that pits the learned model only against raw CCG (`ccg_baseline.py`).

**Change.** Implement **GLM-CCG** (Ren/Ito/Mizuseki, #A1 — *verify author list before citing*) as
the real baseline-to-beat; **STR** (Zhang/Cai 2017, #1) as a linear-subthreshold ablation on the
same sessions to isolate what the learned/nonlinear model buys over the linear-subthreshold form;
optionally **Hafizi** E/I-type + latency (#A2 — *verify*) as the published analog of our Dale/type
latent. Gerhard/Marder STG (#A4) is the citation that linear models are documented to fail on a
real circuit.

**Decision rule.** Standalone deliverable; gate on whether the learned model clears GLM-CCG and
STR on the same benchmark, reported honestly even if the margin is small.

**Log.** Entries per baseline; a "how each baseline handles inhibition" comparison table once
GLM-CCG and STR are in, since inhibition is the differentiating axis.

**Risk.** Medium implementation effort; low scientific risk.

---

### P6 — Sign-agnostic candidate score (conditional on P0)

**Hypothesis.** `causal_pair_score` counts only short-latency pre→post coincidences with positive
weighting — an excitatory detector. Inhibitory pairs produce a coincidence *deficit* and score
~0, so in `hybrid` mode they enter only via the spatial fraction. If P0 shows an inhibitory
coverage gap, this caps recall regardless of the model.

**Change.** Make the temporal candidate score two-sided (a CCG-style deviation flagging both
excess and deficit of short-latency coincidences) so inhibitory pairs also rank into the candidate
set. Flag-guarded; default unchanged.

**Decision rule.** Advance only if P0 shows the coverage gap; success = inhibitory candidate
coverage rises toward excitatory without bloating K.

**Risk.** Low.

---

## 3. Cross-cutting integrity rules (apply to every phase)

- Keep the benchmark numbers exact (§0). Cite the exact run from `EXPERIMENT_LOG.md`; never mix
  sessions.
- Always label **oracle (ground-truth type/sign)** vs **inferred**. Oracle results are diagnostics,
  not headline claims.
- Every code change lands behind an explicit CLI flag with the current behavior as default; the
  existing baselines must stay runnable and reproducible.
- Run changes as **isolated ablations** against a fixed session before stacking. One variable at a
  time, or the comparison is confounded (cf. the `2026-06-07` candidate-lag confound noted in the
  log).
- Do not replace a baseline checkpoint/config with a new variant unless it wins on a clean,
  matched comparison.
- Paper citations marked *verify* in `02_related_work_…` must be checked against the source before
  any thesis/paper use.
- Report guardrail metrics every time: inhibitory-restricted detection AUC and recall (the target),
  overall AUC (≥ 0.916 voltage-augmented), sign recovery (≥ 0.70), and FP composition
  (shared-parent / within-cluster / reverse-of-true enrichment from the existing FP diagnostic).

---

## 4. Open decisions / parking lot

- Voltage frame for P3 reversal potentials: normalized-frame constants vs learnable `E_E`/`E_I` —
  decide after the oracle-type prototype shows whether fixed reversals suffice.
- Whether inferred presynaptic type should come from weight-sign, spike-waveform width (if exposed
  by the simulator), or a Hafizi-style joint latent — decide after P1b's inferred-vs-oracle type
  confusion is measured.
- STR "inference invariance" robustness experiment (one constant recovers strength across regimes):
  test whether our learned weights are regime-robust (`weight_corr` is already ~0.75 on the strong
  session). Paper-experiment, not an inference-script change — park until P3 resolves.

---

## Appendix A — Claude Code prompt log

Prompts are archived here as they are issued, so the plan and its execution history live together
in git. Append new prompts; do not delete old ones.

### Prompt 1 — P0 + P1a (post-hoc, no retrain)

> See the chat message accompanying this commit, or paste the block below into Claude Code from
> the repo root on branch `feature/inhibition-detection`. (The full text is duplicated in the chat
> for convenience; this appendix is the version-controlled copy.)

```
Task: Phase P0 + P1a from 03_inhibition_improvement_plan.md — post-hoc inhibition diagnostics, no model changes, no retraining (unless no saved outputs exist).
You are on branch feature/inhibition-detection. Read 03_inhibition_improvement_plan.md (sections 0–2, P0, P1a) for intent. Do not change any model code, defaults, or the simulator. Everything here is post-hoc analysis plus one log entry. Keep these benchmark numbers exact if you reference them and never round: inhibitory sign recovery 0.70, overall AUC 0.883 spike-only / 0.916 voltage-augmented, inhibitory detection ~0.48, 461 neurons (366 E / 95 I), 2,780 synapses.
Background you need (don't re-derive): the inference model recovers a signed connectivity_matrix[post, pre]. Saved connectivity exports (connectivity_<session>_<tag>.npz) contain at least connectivity_matrix, true_weights, neighbor_indices (all [post, pre] or [post, K]), and may contain threshold, per_neuron_ids, per_neuron_thresholds, neuron_positions. See fp_diagnostic.py and stratified_eval.py for the exact load pattern. Detection AUC in this project is computed on |connectivity_matrix|. Ground-truth presynaptic E/I type = sign of a presynaptic neuron's nonzero outgoing weights in true_weights (negative = inhibitory, positive = excitatory, per the simulator's Dale-respecting generation).
Step 1 — locate inputs. Find saved connectivity_*.npz under LIF-simulation/voltage_augmented_learned_lif_outputs/ and LIF-simulation/learned_lif_outputs/. Prefer a voltage-augmented run on a strong session (e.g. a 20260523_084407 or 20260604_023028 vlam1 export). If none exist, run the smallest viable voltage-augmented fit to produce one: auto-discover a session under LIF-simulation/LIF data/, inspect the voltage CLI via --help, and use a deliberately tiny budget (few epochs, ~4 surrogates, connectivity_threshold_mode=surrogate_fdr, --exclude-detected-bursts). State clearly in the log that this is a throwaway fit for producing a diagnostic input, not a result run.
Step 2 — write scripts/inhibition_diagnostics.py (argparse: --conn <npz>, optional --session <dir>, --out-dir <dir>). It must:

Derive ground-truth presynaptic type from true_weights sign. Assert per-presynaptic sign consistency; report any presynaptic neurons with mixed-sign outgoing edges as a count (do not crash — report).
P0 candidate coverage, split E vs I: fraction of true E edges, and of true I edges, whose presynaptic appears in the postsynaptic neuron's neighbor_indices list. Report both, plus the gap.
P0 type-restricted detection: detection AUC and AP on |connectivity_matrix| computed (a) over all candidates, (b) over excitatory candidates only, (c) over inhibitory candidates only. This is the honest decomposition of the aggregate AUC.
P0 signal-vs-null: for inhibitory candidates, plot/quantify the distribution of |W| for true-I edges vs true-negative I candidates (overlap / AUC between the two); same for excitatory as a reference. Save a histogram PNG.
P1a post-hoc type-stratified thresholding (label OUTPUT as ORACLE type): build a per-type empirical null from the |W| of true-negative candidates of each type as a fast stand-in null, choose a threshold per type at a matched FDR target (use surrogate_fdr=0.005), and report true-I edges selected and estimated FDR under (i) the existing pooled threshold and (ii) the type-stratified threshold. State explicitly in output and log that this null is approximate (true-negative scores leak common-input structure), so it's a directional check only.
Write a CSV summary and the PNG(s) to LIF-simulation/voltage_augmented_learned_lif_outputs/diagnostics_<session>/.

Step 3 — log. Append one EXPERIMENT_LOG.md entry using the existing template (### YYYY-MM-DD - Short Title, then Goal / Hypothesis / Files changed / Parameters changed / Validation run / Key result / Interpretation / Next step), today's date, title like "P0+P1a Inhibition Coverage And Type-Stratified Threshold Diagnostics For <session>". In Key result give the concrete numbers: E vs I candidate coverage and gap; overall / E-only / I-only AUC and AP; I-edge-vs-null overlap; and pooled-vs-stratified I-edges-selected and estimated FDR. In Interpretation, answer the two decision gates from the plan directly: (a) is there an inhibitory candidate-coverage gap >10 points (→ P6 jumps ahead of P3)? (b) does oracle type-stratification recover meaningful true-I edges at tolerable FDR (→ P1b worth doing)? In Next step, recommend which phase to run next based on those answers.
Step 4 — archive the prompt. Append this prompt's full text under "Prompt 1" in Appendix A of 03_inhibition_improvement_plan.md (replace the placeholder line).
Step 5 — commit + push. Stage the new script, the diagnostics artifacts, the EXPERIMENT_LOG.md entry, and the updated plan doc. Commit with a descriptive message and push feature/inhibition-detection.
Definition of done: the diagnostics script runs to completion on a real saved (or freshly produced) connectivity export; the CSV + PNG exist; the EXPERIMENT_LOG.md entry contains the actual numbers and a direct answer to both decision gates; the plan doc's Appendix A holds this prompt; everything is pushed. Do not change model defaults, the surrogate machinery, or the simulator in this phase.
```

**Result (run 2026-06-30, export `connectivity_20260523_084407_voltage_all20_K100.npz`):** coverage E 0.9693 / I 0.9423 (gap 2.71 pts → no I-coverage gap); AUC all 0.9162 / E-only 0.9747 / I-only 0.6337 (AP 0.7602 / 0.8917 / 0.1300); inhibitory signal-vs-null overlap 0.7195 vs excitatory 0.1392; existing pooled threshold selects 70 true-I edges at realized I-FDR 0.6833; inhibitory FDR floor 0.50 makes the FDR=0.005 target unreachable, so pooled and oracle type-stratified both select 0 true-I edges at matched FDR. Gate (a): no (skip P6 jump). Gate (b): no (skip P1b). Recommended next: P2 loss reshaping, with P3 queued. See `EXPERIMENT_LOG.md` 2026-06-30 entry.

### Prompt 2 — P2 loss-reshaping flags + ablations (P0b/P5 plan edits)

> Issued after the user confirmed `20260523_084407` is non-depressing (redirecting from P0b straight
> to P2). Version-controlled copy; full text below.

```
Task: Phase P2 from 03_inhibition_improvement_plan.md — reshape the voltage/L1 loss toward inhibition, on the non-depressing 20260523_084407 session, as isolated flag-guarded ablations. The metric is the P0 inhibitory decomposition.
On branch feature/inhibition-detection. Context: P0/P1a (entry 2026-06-30) localized the inhibition failure to magnitude recovery (I-only AUC 0.6337 vs E-only 0.9747; inhibitory |W|-vs-null overlap 0.72), and ruled out coverage and thresholding. The user has confirmed 20260523_084407 does not use depression, so depression is excluded as a cause on this session; the remaining suspects are synaptic functional form and voltage-loss weighting. P2 tests the loss-weighting suspect cheaply before the P3 conductance rewrite. Keep canonical benchmark numbers (0.883 / 0.916 / 0.70 / 0.48 / 461 / 366 E / 95 I / 2,780) distinct from this run's I-only 0.6337; never round.
Baseline config to match exactly (the 0523 voltage run behind the P0 export): --training-mode event_window --slow-state-mode adaptation_h --connectivity-threshold-mode surrogate_fdr_per_neuron --surrogate-fdr 0.005 --n-threshold-surrogates 4 --surrogate-epochs 2 --surrogate-patience 1 --k 100 --epochs 40 --patience 8 --batch 128 --max-delay 8 --l1 0.01 --pos-weight 5.0 --voltage-lambda 1.0 --mask-pre-ms 0.0 --warmup 100 --exclude-detected-bursts, hybrid candidates (80 spatial + 20 temporal, lag 1–8), session LIF-simulation/LIF data/20260523_084407. Each ablation changes exactly ONE thing; do not stack.
Implement two flags in the voltage script (lif_inference/voltage_augmented_learned_lif_connectivity.py: build_parser, and compute_voltage_augmented_event_loss / compute_voltage_augmented_continuous_loss), each defaulting to current behavior:
--voltage-hyperpol-gamma γ (default 0.0): weight the per-bin voltage residual by 1 + γ·relu(−vt) before the smooth_l1 reduction, applied only on valid voltage bins (the existing mask). The normalized target vt is already per-neuron baseline-subtracted, so relu(−vt) upweights below-baseline (IPSP) bins. This needs no type information — Banerjee's remedy, loss-side.
--l1-inhibitory-scale s (default 1.0): scale the L1 penalty per candidate by s for candidates whose presynaptic neuron is inhibitory, 1.0 otherwise. For this first test use ORACLE presynaptic type (sign of each presynaptic neuron's outgoing nonzero true_weights), passed in and clearly labeled oracle in the run tag and log; an inferred-type version is a follow-up only if this helps. This stops L1's constant-magnitude gradient from pinning small inhibitory weights to zero.
Runs. Ablation B: baseline + --voltage-hyperpol-gamma 2.0 (tag suffix _hyperpolg2). Ablation A: baseline + --l1-inhibitory-scale 0.0 oracle (tag suffix _l1inh0_ORACLE). If compute is tight, run B first alone and push — it's the principled, oracle-free discriminator; A can follow in the next push. (Note for the user: these are full ~multi-hour CPU runs each; reduce --epochs for a faster first read if needed, matching the reduced budget across arms.)
Evaluate. For each ablation's saved connectivity export, re-run python scripts/inhibition_diagnostics.py --conn <export> to get the inhibitory decomposition, and report vs the 0523 baseline: I-only AUC (target: up from 0.6337), I-only AP (from 0.1300), inhibitory |W|-vs-null overlap (down from 0.72), with guardrails overall AUC ≥ ~0.916 and sign_accuracy ≥ ~0.97 (must not regress materially; the canonical inhibitory sign floor is 0.70). E-only AUC should stay ~0.975.
Log + plan update. One EXPERIMENT_LOG.md entry per ablation (template, today's date, titles like "P2 Hyperpol-Weighted Voltage Ablation (γ=2) For 20260523_084407" / "P2 Oracle Asymmetric-L1 (inh=0) Ablation For 20260523_084407"), each reporting the metrics above and stating in Interpretation whether the loss change lifted I-only separation (→ inhibition was present-but-underweighted, keep/extend the change) or did not (→ evidence the current-based form is the ceiling, commit to P3). Then apply these edits to 03_inhibition_improvement_plan.md (these carry the P0b-prompt edits that were never run, since the user redirected before P0b): add a Status line to P0 ("no coverage gap; I-only AUC 0.6337 vs E-only 0.9747; magnitude collapse confirmed") and P1a ("skipped P1b — stratification cannot fix within-type separability"); add P0b — depression-vs-functional-form attribution as a section, marked applicable only to depressing sessions (moot on non-depressing 0523); add P5 — short-term-depression-aware inference synapse after P3 (per-presynaptic resource trace R̂_j(t), effective drive w_k·R̂_j(t)·spike, STP params fixed-from-known/oracle first, candidate to fuse with P3, reserved for depressing sessions); update the run-order line to: general P0 → P1a → P0b(depressing) → P2 → (P3 ∥ P5), and note non-depressing 0523 runs P0 → P1a → P2 → P3. Append this prompt under "Prompt 2" in Appendix A.
Commit + push. Stage the flag implementation, ablation exports/artifacts, diagnostic re-runs, log entries, and plan edits; commit; push feature/inhibition-detection.
Definition of done: at least Ablation B runs end-to-end and its inhibitory decomposition is reported against the 0523 baseline with guardrails; the log states whether loss-reshaping moved I-only separation; the plan reflects the P0/P1a status and the P0b/P5 additions; everything pushed. No default/model/simulator changes outside the two new flags; oracle usage explicitly labeled.
```

**Result (2026-06-30).** Both flags implemented in
`lif_inference/voltage_augmented_learned_lif_connectivity.py` (parser + both loss fns + run-pipeline
plumbing + saved provenance), defaulting to exact no-ops and leaving the surrogate machinery
untouched. Verified: unit tests (defaults reproduce the prior loss exactly; γ-weighting and L1
scaling match closed form) and end-to-end CLI smoke runs on a present voltage session incl. the
surrogate-FDR path. **Ablations A/B on `20260523_084407` were NOT run: that session's raw spike+
voltage data is absent on this machine** (gitignored; only the connectivity export is present), and
P2 ablations require training on the recordings. Plan edits (P0/P1a Status, P0b, P5, run order)
applied. Pending a decision on data provenance / session choice for the actual runs. See
`EXPERIMENT_LOG.md` 2026-06-30 P2 entry.
