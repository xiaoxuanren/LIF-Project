# Connectivity-eval export — per-edge signed scores + signed ground truth

A small, self-contained snapshot for writing **sign-conditional thresholding**,
**inhibitory-AUPRC**, and **predicted-weight-distribution** analyses that plug
straight into the inference pipeline's `evaluate_connectivity`. It deliberately
ships only the per-edge scores + ground truth — **not** the multi-GB session
(spikes/voltage). Everything here aligns index-for-index with what
`evaluate_connectivity` already returns.

## Provenance

- **Session:** `20260523_084407` — 461 neurons (358 excitatory sources, 95 inhibitory
  sources; strict **Dale**, every source neuron has a single sign), all **20 recordings**
  (60 s each), active stimulus-driven regime.
- **Candidate edges:** K=100 hybrid (spatial_frac 0.8), burst-excluded, `dt=1.0`,
  `max_delay=8`. **46,100** directed candidate edges `(post, pre)`. Within the candidate
  set: **2,275 true excitatory + 408 true inhibitory = 2,683** true edges (the full
  ground truth has 2,780 edges; 97 fall outside the candidate set → recall ceiling).
- **Predicted scores included:**
  - `conn_signed_voltage` — voltage-augmented learned-LIF (AUC 0.916, AP 0.760).
  - `conn_signed_spikeonly` — spike-only learned-LIF (AUC ≈ 0.883).

## The `evaluate_connectivity` contract (what these arrays mirror)

`lif_inference/connectivity_metrics.py`:

```python
results, scores, labels, conn_matrix = evaluate_connectivity(
    model, neighbor_indices, true_binary, neuron_ids=None,
    connectivity_threshold_mode='oracle_f1',     # or 'surrogate_fdr' / 'surrogate_fdr_per_neuron'
    surrogate_score_sets=None, surrogate_fdr=0.005)
```

- `conn_matrix` — **signed** `[post, pre]` weight matrix = `model.get_connectivity_matrix(neighbor_indices)`.
  Sign IS the predicted polarity: `> 0` → predicted excitatory, `< 0` → predicted inhibitory.
- `scores` — flattened **`abs(conn_matrix[post, pre])`** over candidate edges, in this exact order:

  ```python
  for neuron_id in neuron_ids:            # neuron_ids defaults to arange(n_neurons)
      for pre in neighbor_indices[neuron_id]:
          ...                              # one flattened entry per (post=neuron_id, pre)
  ```

  **`evaluate_connectivity` takes the absolute value — it discards sign for AUC/AP/threshold.**
  To do anything sign-aware you must use `conn_matrix` (signed), flattened in the SAME order.
- `labels` — `true_binary[post, pre]` at those edges (1 = real edge).
- `results` keys: `auc, ap, threshold, connectivity_threshold_mode, estimated_fdr,
  expected_null_selected, selected_edges, per_score_thresholds (optional), f1, precision,
  recall, tp, fp, fn, tn, mean_connected_weight, n_positive, n_total`.

## Files

### `connectivity_eval_export_20260523_084407.npz`

Matrices (all `[post, pre]`, shape `461×461`) and flattened per-edge arrays (length 46,100,
**in `evaluate_connectivity` flatten order** so they align index-for-index with `scores`/`labels`):

| key | shape | meaning |
|-----|-------|---------|
| `neuron_ids` | (461,) | `arange(461)` — the order edges are flattened in |
| `pre_is_inh` | (461,) bool | source neuron is inhibitory (Dale, neuron-level) |
| `true_W_signed` | (461,461) f32 | ground-truth **signed** weights (+exc / −inh / 0) |
| `true_binary` | (461,461) i32 | ground-truth binary edges (= `evaluate_connectivity`'s `true_binary`) |
| `neighbor_indices` | (461,) object | candidate `pre` ids per `post` |
| `conn_signed_voltage` | (461,461) f32 | voltage model **signed** scores (= its `conn_matrix`) |
| `conn_signed_spikeonly` | (461,461) f32 | spike-only model signed scores |
| `flat_post` | (46100,) | `post` id per candidate edge (= `score_neuron_ids`) |
| `flat_pre` | (46100,) | `pre` id per candidate edge |
| `flat_label` | (46100,) | `true_binary` at edge (= `evaluate_connectivity`'s `labels`) |
| `flat_true_sign` | (46100,) i8 | +1 true-exc edge, −1 true-inh edge, 0 non-edge |
| `flat_pre_is_inh` | (46100,) bool | source neuron inhibitory |
| `flat_signed_voltage` | (46100,) | voltage signed score (`abs` of this == `scores`) |
| `flat_signed_spikeonly` | (46100,) | spike-only signed score |

### `ccg_score_matrix_20260523_084407.npy`

Full **461×461 `[post, pre]`** CCG-baseline score matrix, same neuron ordering / index
convention as the npz above. Raw **excitatory-coincidence excess** (positive = excess,
negative = below-baseline deficit; non-candidate pairs are 0). CCG assigns **no E/I sign**
(the pipeline detects on `|score|`), so this is the only CCG file — values are kept signed
(excess), not abs'd. Load: `np.load("ccg_score_matrix_20260523_084407.npy")` and plot next to
`conn_signed_voltage` / `true_W_signed`. Ordering verified: `|CCG|` vs `true_binary` AUC ≈ 0.72;
median `|CCG|` at true-excitatory edges (0.012) ≫ non-edges (0.0025) and ≈ 0 at inhibitory edges
(CCG is excitatory-only). Min/max ≈ −0.617 / 0.328, 33,555 nonzero.

### `per_edge_scores_voltage_all20.csv`

Same flattened table, human-readable: `post, pre, label, true_sign, pre_is_inh,
signed_score_voltage, abs_score_voltage, signed_score_spikeonly`.

### `../export_eval_data.py`

The reproducer (numpy-only) that built this from the saved `connectivity_*.npz` outputs.

## What the drop-in script should implement

Three functions that consume `evaluate_connectivity`'s output plus the signed conn /
signed `W`, so they can be added to `connectivity_metrics.py` and called right after
`evaluate_connectivity(...)`:

```python
def sign_conditional_threshold(conn_matrix, neighbor_indices, true_binary,
                               neuron_ids=None, mode='surrogate_fdr', surrogate_fdr=0.005):
    """Threshold excitatory candidates (conn>0) and inhibitory candidates (conn<0)
    SEPARATELY — inhibitory weights are smaller, so one global |w| cutoff under-detects
    them. Return per-sign thresholds + a combined predicted-edge mask, in the same
    flatten order as evaluate_connectivity."""

def inhibitory_auprc(conn_matrix, neighbor_indices, true_W_signed, neuron_ids=None):
    """AUPRC for detecting INHIBITORY edges specifically: detector = -signed_score
    (more negative = more confidently inhibitory), positives = (true_W_signed < 0).
    Report alongside the overall (sign-blind) AP so inhibitory recovery is visible."""

def weight_distribution(conn_matrix, neighbor_indices, true_W_signed, neuron_ids=None):
    """Distributions of predicted SIGNED scores split by true class
    (true-exc / true-inh / non-edge), and predicted-vs-true signed weight correlation
    restricted to true edges. Surfaces E/I weight separation that abs() hides."""
```

All three flatten with the canonical loop above (or just use the `flat_*` arrays here,
which are already in that order). `abs(flat_signed_voltage) == scores` from
`evaluate_connectivity`, so results line up edge-for-edge.
