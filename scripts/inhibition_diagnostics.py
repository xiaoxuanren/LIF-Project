"""P0 + P1a inhibition diagnostics for the learned-LIF connectivity pipeline.

Post-hoc analysis only. Reads a saved ``connectivity_<session>_<tag>.npz`` and,
without re-running the model, answers:

P0  - candidate coverage split by presynaptic E vs I type (and the E-I gap);
    - detection AUC / AP on ``|connectivity_matrix|`` over (a) all candidates,
      (b) excitatory candidates only, (c) inhibitory candidates only -- the
      honest decomposition of the aggregate AUC;
    - the |W| distribution of true-I edges vs true-negative I candidates (the
      inhibitory signal-vs-null overlap), with the same for E as a reference.

P1a - ORACLE type-stratified thresholding (no retrain): build a per-type
      empirical null from the |W| of true-negative candidates of each type as a
      fast stand-in null, pick a threshold per type at a matched FDR target
      (default surrogate_fdr=0.005), and report true-I edges selected and the
      realized I-FDR under (i) the existing pooled threshold saved in the export
      and (ii) the type-stratified threshold. A matched pooled-at-FDR threshold
      is also reported so pooled-vs-stratified is apples-to-apples.

CAVEAT (state everywhere): the per-type null is built from true-negative
candidate scores, which leak common-input structure (shared-parent / within-
cluster enrichment -- see fp_diagnostic.py). So the FDR figures here are a
DIRECTIONAL check, not a calibrated surrogate-FDR. The principled circular-shift
per-type null is P1b.

Ground-truth presynaptic type = sign of a presynaptic neuron's nonzero outgoing
weights in ``true_weights`` (negative = inhibitory, positive = excitatory, per
the simulator's Dale-respecting generation).

Pure NumPy + scikit-learn + Matplotlib. No torch, no GPU, no model re-run.

Usage:
    python scripts/inhibition_diagnostics.py \
        --conn "LIF-simulation/voltage_augmented_learned_lif_outputs/connectivity_<session>_<tag>.npz" \
        [--session "LIF-simulation/LIF data/<session>"] \
        [--out-dir "LIF-simulation/voltage_augmented_learned_lif_outputs/diagnostics_<session>"] \
        [--fdr 0.005]
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------------------------------------------------------
# loading (mirrors fp_diagnostic.py / stratified_eval.py load pattern)
# ----------------------------------------------------------------------------
def _scalar(d, key, default=None):
    if key not in d.files:
        return default
    a = np.asarray(d[key])
    return a.item() if a.ndim == 0 else a


def load_connectivity(conn_path):
    d = np.load(conn_path, allow_pickle=True)
    C = np.asarray(d["connectivity_matrix"], dtype=float)   # [post, pre] signed
    Wtrue = np.asarray(d["true_weights"], dtype=float)      # [post, pre] signed
    NB = np.asarray(d["neighbor_indices"])                  # [post, K] candidate pre
    global_thr = _scalar(d, "threshold", None)
    if global_thr is not None:
        global_thr = float(np.asarray(global_thr).ravel()[0])
    thr_mode = _scalar(d, "connectivity_threshold_mode", None)
    if thr_mode is not None:
        thr_mode = str(np.asarray(thr_mode).ravel()[0])
    # per-neuron thresholds (surrogate-FDR mode); empty here for oracle_f1 exports
    thr_by_post = None
    if "per_neuron_ids" in d.files and "per_neuron_thresholds" in d.files:
        ids = np.asarray(d["per_neuron_ids"]).ravel()
        thr = np.asarray(d["per_neuron_thresholds"], dtype=float).ravel()
        if ids.size and ids.size == thr.size:
            thr_by_post = {int(i): float(t) for i, t in zip(ids, thr)}
    return C, Wtrue, NB, global_thr, thr_mode, thr_by_post


def session_from_path(conn_path, override=None):
    if override:
        return os.path.basename(os.path.normpath(override))
    base = os.path.basename(conn_path)
    m = re.search(r"(\d{8}_\d{6})", base)
    return m.group(1) if m else os.path.splitext(base)[0]


# ----------------------------------------------------------------------------
# ground-truth presynaptic type from sign of outgoing true weights
# ----------------------------------------------------------------------------
def presynaptic_type(Wtrue):
    """Return (pretype, n_mixed). pretype[p] in {+1 (E), -1 (I), 0 (no outgoing)}.

    A presynaptic neuron p's outgoing edges are column Wtrue[:, p]. Under the
    simulator's Dale-respecting generation all nonzero outgoing weights share a
    sign; mixed-sign presynaptics are reported (not crashed on).
    """
    npre = Wtrue.shape[1]
    pretype = np.zeros(npre, dtype=int)
    n_mixed = 0
    mixed_ids = []
    for p in range(npre):
        col = Wtrue[:, p]
        nz = col[col != 0.0]
        if nz.size == 0:
            continue
        n_pos = int((nz > 0).sum())
        n_neg = int((nz < 0).sum())
        if n_pos > 0 and n_neg > 0:
            n_mixed += 1
            mixed_ids.append(p)
        # type from the sign of the summed outgoing drive (robust to a stray
        # mixed entry); for clean Dale data this equals every edge's sign.
        pretype[p] = 1 if nz.sum() > 0 else -1
    return pretype, n_mixed, mixed_ids


# ----------------------------------------------------------------------------
# candidate flattening
# ----------------------------------------------------------------------------
def build_candidates(C, Wtrue, NB, pretype):
    """Flatten candidate (post, pre) pairs.

    Returns dict of arrays over candidates:
        post, pre, score=|C|, is_true (bool), pre_is_E, pre_is_I.
    """
    n = C.shape[0]
    posts, pres = [], []
    for j in range(n):
        for p in np.asarray(NB[j]).ravel():
            p = int(p)
            if p == j:
                continue
            posts.append(j)
            pres.append(p)
    posts = np.asarray(posts, dtype=int)
    pres = np.asarray(pres, dtype=int)
    score = np.abs(C[posts, pres])
    true_mag = np.abs(Wtrue[posts, pres])          # |true weight| per candidate
    is_true = true_mag > 0
    pre_is_E = pretype[pres] == 1
    pre_is_I = pretype[pres] == -1
    return dict(post=posts, pre=pres, score=score, true_mag=true_mag, is_true=is_true,
                pre_is_E=pre_is_E, pre_is_I=pre_is_I)


def magnitude_corr(recovered_abs, true_abs):
    """Within-type Pearson correlation of recovered |W| vs true |W| over true edges.

    The honest magnitude-recovery metric under oracle sign routing: unlike a pooled
    signed weight correlation (which is inflated because oracle routing fixes every
    edge's sign, so the E>0 / I<0 split alone drives the pooled Pearson), this
    correlates only magnitudes and is computed separately within each type.
    """
    if recovered_abs.size < 2:
        return float("nan")
    if np.std(recovered_abs) < 1e-12 or np.std(true_abs) < 1e-12:
        return float("nan")
    return float(np.corrcoef(recovered_abs, true_abs)[0, 1])


# ----------------------------------------------------------------------------
# coverage split by type
# ----------------------------------------------------------------------------
def coverage_by_type(Wtrue, NB, pretype):
    """Fraction of true E (resp. I) edges whose pre is in post's candidate set."""
    n = Wtrue.shape[0]
    cand_sets = [set(int(x) for x in np.asarray(NB[j]).ravel()) for j in range(n)]
    cov = {1: [0, 0], -1: [0, 0]}  # type -> [covered, total]
    for j in range(n):
        pres = np.where(np.abs(Wtrue[j]) > 0)[0]
        for p in pres:
            p = int(p)
            if p == j:
                continue
            t = pretype[p]
            if t == 0:
                continue
            cov[t][1] += 1
            if p in cand_sets[j]:
                cov[t][0] += 1
    e_cov = cov[1][0] / cov[1][1] if cov[1][1] else float("nan")
    i_cov = cov[-1][0] / cov[-1][1] if cov[-1][1] else float("nan")
    return e_cov, i_cov, cov[1], cov[-1]


# ----------------------------------------------------------------------------
# detection AUC / AP over a candidate subset
# ----------------------------------------------------------------------------
def auc_ap(scores, labels):
    labels = np.asarray(labels, bool)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan"), float("nan")
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))


def overlap_coefficient(a, b, bins=60):
    """Overlapping coefficient (area shared by two normalized histograms)."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    bw = edges[1] - edges[0]
    return float(np.minimum(ha, hb).sum() * bw)


# ----------------------------------------------------------------------------
# realized-FDR (BH-style) threshold selection on a candidate pool
# ----------------------------------------------------------------------------
def fdr_threshold(scores, labels, q):
    """Pick the deepest score threshold whose realized FDR over the pool is <= q.

    Realized FDR here uses ORACLE labels: among candidates with score >= t,
    FDR = (#false) / (#selected). Because the per-type "null" is built from the
    true-negative candidate scores themselves, this realized FDR IS the
    empirical-null FDR estimate the plan asks for (#null>=t / #selected>=t).
    We sort candidates by score descending and take the largest prefix k with
    FP_k / k <= q (the standard BH-style rule), then threshold = score at that
    prefix boundary. See module docstring caveat (the null leaks structure, so
    it has a non-zero FDR floor -- targets below the floor are unreachable).

    Returns (threshold, n_selected, n_true_selected, realized_fdr, fdr_floor).
    fdr_floor is the minimum realized FDR achievable anywhere in the ranking;
    if fdr_floor > q the target is unreachable and an empty selection is returned.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    if scores.size == 0:
        return float("inf"), 0, 0, float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    lab_sorted = labels[order]
    fp_cum = np.cumsum(~lab_sorted)
    k = np.arange(1, scores.size + 1)
    fdr_cum = fp_cum / k
    fdr_floor = float(fdr_cum.min())
    ok = np.where(fdr_cum <= q)[0]
    if ok.size == 0:
        # target below the FDR floor -> empty selection (unreachable on this null)
        return float("inf"), 0, 0, 0.0, fdr_floor
    kmax = ok.max()  # 0-based index of deepest acceptable prefix
    thr = s_sorted[kmax]
    return (float(thr), int(kmax + 1), int(lab_sorted[:kmax + 1].sum()),
            float(fdr_cum[kmax]), fdr_floor)


def fdr_power_sweep(scores, labels, targets):
    """For each FDR target, the deepest threshold and true-edges selected."""
    out = []
    for q in targets:
        thr, nsel, ntrue, fdr, floor = fdr_threshold(scores, labels, q)
        out.append(dict(target=q, threshold=thr, n_selected=nsel,
                        n_true_selected=ntrue, realized_fdr=fdr, floor=floor))
    return out


def apply_threshold(scores, labels, thr):
    """Apply a fixed threshold; return (n_selected, n_true_selected, realized_fdr)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    sel = scores >= thr
    n_sel = int(sel.sum())
    n_true = int(labels[sel].sum())
    fdr = (n_sel - n_true) / n_sel if n_sel else float("nan")
    return n_sel, n_true, fdr


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------
def make_histogram(cand, out_png, session, auc_I, ovl_I, auc_E, ovl_E):
    sc = cand["score"]
    true = cand["is_true"]
    Iidx = cand["pre_is_I"]
    Eidx = cand["pre_is_E"]
    I_true = sc[Iidx & true]
    I_null = sc[Iidx & ~true]
    E_true = sc[Eidx & true]
    E_null = sc[Eidx & ~true]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, (s_true, s_null, tag, auc, ovl, col) in zip(
        ax,
        [(I_true, I_null, "INHIBITORY", auc_I, ovl_I, "#c1121f"),
         (E_true, E_null, "EXCITATORY", auc_E, ovl_E, "#2a9d8f")],
    ):
        allv = np.concatenate([s_true, s_null]) if (s_true.size + s_null.size) else np.array([0.0, 1.0])
        bins = np.linspace(float(allv.min()), float(allv.max()), 60)
        a.hist(s_null, bins=bins, density=True, alpha=0.55, color="#999",
               label=f"true-negative cand (n={s_null.size})")
        a.hist(s_true, bins=bins, density=True, alpha=0.55, color=col,
               label=f"true edges (n={s_true.size})")
        a.set_xlabel("recovered |W|  (|connectivity_matrix|)")
        a.set_ylabel("density")
        a.set_title(f"{tag} candidates\nsignal-vs-null AUC={auc:.3f}  overlap={ovl:.3f}")
        a.legend(fontsize=9)
    fig.suptitle(f"P0 signal-vs-null: recovered |W|, true edges vs true-negative candidates "
                 f"[{session}]", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_fdr_power_fig(I_sweep, pooled_sweep, n_true_I, strat_floor, pooled_floor,
                       ex_true, ex_fdr, global_thr, out_png, session):
    """Inhibitory yield (true-I edges selected) vs realized I-FDR, pooled vs stratified."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    s_fdr = [r["realized_fdr"] for r in I_sweep if r["n_selected"]]
    s_sel = [r["n_true_selected"] for r in I_sweep if r["n_selected"]]
    p_fdr = [r["realized_fdr"] for r in pooled_sweep if r["n_true_selected"] is not None
             and not np.isnan(r["realized_fdr"])]
    p_sel = [r["n_true_selected"] for r in pooled_sweep if r["n_true_selected"] is not None
             and not np.isnan(r["realized_fdr"])]
    if s_sel:
        ax.plot(s_fdr, s_sel, "o-", color="#c1121f", label="type-stratified (oracle I-null)")
    if p_sel:
        ax.plot(p_fdr, p_sel, "s--", color="#555", label="pooled threshold, restricted to I")
    if global_thr is not None and ex_true is not None and not np.isnan(ex_fdr):
        ax.plot([ex_fdr], [ex_true], "*", color="#2a9d8f", markersize=18,
                label=f"existing pooled thr={global_thr:.3f}")
    ax.axvline(strat_floor, color="#c1121f", ls=":", alpha=0.6,
               label=f"I-only FDR floor={strat_floor:.3f}")
    ax.axvline(0.005, color="k", ls=":", alpha=0.4, label="FDR target 0.005")
    ax.set_xlabel("realized inhibitory FDR (false-I / selected-I)")
    ax.set_ylabel(f"true-I edges selected  (of {n_true_I} recoverable)")
    ax.set_title(f"P1a inhibitory yield vs FDR: stratification cannot beat the\n"
                 f"inhibitory FDR floor of {strat_floor:.2f}  [{session}]")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conn", required=True, help="connectivity_<session>_<tag>.npz")
    ap.add_argument("--session", default=None, help="session dir (only used to name outputs)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <conn dir>/diagnostics_<session>)")
    ap.add_argument("--fdr", type=float, default=0.005,
                    help="matched FDR target for P1a thresholds (default 0.005)")
    args = ap.parse_args()

    session = session_from_path(args.conn, args.session)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.conn),
                                           f"diagnostics_{session}")
    os.makedirs(out_dir, exist_ok=True)

    C, Wtrue, NB, global_thr, thr_mode, thr_by_post = load_connectivity(args.conn)
    n = C.shape[0]
    K = NB.shape[1] if NB.ndim > 1 else None
    print(f"[load] {os.path.basename(args.conn)}")
    print(f"       n_neurons={n}  K={K}  threshold={global_thr}  mode={thr_mode}  "
          f"per_neuron_thr={'yes' if thr_by_post else 'no'}")

    # ---- ground-truth presynaptic type ----
    pretype, n_mixed, mixed_ids = presynaptic_type(Wtrue)
    nE = int((pretype == 1).sum())
    nI = int((pretype == -1).sum())
    n_true_total = int((np.abs(Wtrue) > 0).sum())
    print(f"[type] presynaptic E={nE}  I={nI}  (typed by outgoing-weight sign)  "
          f"mixed-sign presynaptics={n_mixed}")
    if n_mixed:
        print(f"       mixed-sign pre ids (reported, not fatal): {mixed_ids[:20]}"
              f"{' ...' if len(mixed_ids) > 20 else ''}")

    # ---- P0 coverage split ----
    e_cov, i_cov, e_cnt, i_cnt = coverage_by_type(Wtrue, NB, pretype)
    gap_points = (e_cov - i_cov) * 100.0
    print(f"[P0 coverage] E={e_cov:.4f} ({e_cnt[0]}/{e_cnt[1]})  "
          f"I={i_cov:.4f} ({i_cnt[0]}/{i_cnt[1]})  gap={gap_points:.2f} points")

    # ---- P0 type-restricted detection ----
    cand = build_candidates(C, Wtrue, NB, pretype)
    auc_all, ap_all = auc_ap(cand["score"], cand["is_true"])
    Em = cand["pre_is_E"]; Im = cand["pre_is_I"]
    auc_E, ap_E = auc_ap(cand["score"][Em], cand["is_true"][Em])
    auc_I, ap_I = auc_ap(cand["score"][Im], cand["is_true"][Im])
    print(f"[P0 detection] AUC all={auc_all:.4f} E={auc_E:.4f} I={auc_I:.4f}")
    print(f"               AP  all={ap_all:.4f} E={ap_E:.4f} I={ap_I:.4f}")

    # ---- P0 signal-vs-null overlap ----
    I_true = cand["score"][Im & cand["is_true"]]
    I_null = cand["score"][Im & ~cand["is_true"]]
    E_true = cand["score"][Em & cand["is_true"]]
    E_null = cand["score"][Em & ~cand["is_true"]]
    ovl_I = overlap_coefficient(I_true, I_null)
    ovl_E = overlap_coefficient(E_true, E_null)
    # signal-vs-null AUC is the same quantity as the type-restricted detection AUC
    sig_auc_I = auc_I
    sig_auc_E = auc_E
    print(f"[P0 signal/null] I overlap={ovl_I:.4f} (AUC={sig_auc_I:.4f})  "
          f"E overlap={ovl_E:.4f} (AUC={sig_auc_E:.4f})")

    # ---- within-type magnitude recovery (honest under oracle sign routing) ----
    tm = cand["true_mag"]
    magcorr_E = magnitude_corr(cand["score"][Em & cand["is_true"]], tm[Em & cand["is_true"]])
    magcorr_I = magnitude_corr(cand["score"][Im & cand["is_true"]], tm[Im & cand["is_true"]])
    print(f"[P0 magnitude]  within-type |W| Pearson: E={magcorr_E:.4f}  I={magcorr_I:.4f} "
          f"(over true edges; NOT the pooled signed weight_corr, which oracle routing inflates)")

    png_path = os.path.join(out_dir, f"inhibition_signal_vs_null_{session}.png")
    make_histogram(cand, png_path, session, sig_auc_I, ovl_I, sig_auc_E, ovl_E)
    print(f"[fig] {png_path}")

    # ---- P1a oracle type-stratified thresholding ----
    q = args.fdr
    sc = cand["score"]; tr = cand["is_true"]
    I_sc = sc[Im]; I_tr = tr[Im]
    n_true_I = int(I_tr.sum())

    # (i) existing pooled threshold from the export, applied to I candidates
    ex_sel, ex_true, ex_fdr = (apply_threshold(I_sc, I_tr, global_thr)
                               if global_thr is not None else (None, None, None))
    # (i') matched pooled threshold at FDR=q (pooled null over ALL candidates),
    #      then applied to I candidates -- apples-to-apples reference
    pooled_thr, pooled_nsel_all, pooled_ntrue_all, pooled_fdr_all, pooled_floor = \
        fdr_threshold(sc, tr, q)
    p_sel, p_true, p_fdr = apply_threshold(I_sc, I_tr, pooled_thr)
    # (ii) type-stratified threshold: I-only null at FDR=q
    strat_thr, strat_nsel, strat_true, strat_fdr, strat_floor = fdr_threshold(I_sc, I_tr, q)

    print(f"[P1a] FDR target q={q}   pooled FDR-floor={pooled_floor:.4f}  "
          f"I-only FDR-floor={strat_floor:.4f}")
    print(f"      (target q below an FDR-floor is UNREACHABLE on this approximate null)")
    if global_thr is not None:
        print(f"      (i)  existing pooled thr={global_thr:.5f} ({thr_mode}) -> "
              f"true-I selected={ex_true}/{n_true_I}  realized I-FDR={ex_fdr:.4f}")
    print(f"      (i') pooled@FDR{q} thr={pooled_thr:.5f} -> "
          f"true-I selected={p_true}/{n_true_I}  realized I-FDR={p_fdr:.4f}")
    print(f"      (ii) stratified-I@FDR{q} thr={strat_thr:.5f} -> "
          f"true-I selected={strat_true}/{n_true_I}  realized I-FDR={strat_fdr:.4f}")

    # FDR-power sweep so the pooled-vs-stratified tradeoff is visible even where
    # the headline q=0.005 target is below the FDR floor.
    sweep_targets = [0.005, 0.01, 0.05, 0.10, 0.20, 0.50]
    I_sweep = fdr_power_sweep(I_sc, I_tr, sweep_targets)        # stratified-I
    print(f"[P1a sweep] stratified-I true-edges selected by FDR target:")
    for r in I_sweep:
        reach = "" if r["n_selected"] else "  (UNREACHABLE)"
        print(f"      FDR<={r['target']:<5}: true-I={r['n_true_selected']:3d}/{n_true_I}  "
              f"realized I-FDR={r['realized_fdr']:.4f}  thr={r['threshold']:.4f}{reach}")
    # pooled threshold at each target, then restricted to I candidates
    pooled_sweep = []
    for q_s in sweep_targets:
        p_thr, _, _, _, _ = fdr_threshold(sc, tr, q_s)
        ns, nt, fd = apply_threshold(I_sc, I_tr, p_thr)
        pooled_sweep.append(dict(target=q_s, threshold=p_thr, n_true_selected=nt,
                                 realized_fdr=fd))

    # ---- FDR-power figure (pooled-vs-stratified inhibitory yield) ----
    png2_path = os.path.join(out_dir, f"inhibition_fdr_power_{session}.png")
    make_fdr_power_fig(I_sweep, pooled_sweep, n_true_I, strat_floor, pooled_floor,
                       ex_true, ex_fdr, global_thr, png2_path, session)
    print(f"[fig] {png2_path}")

    # ---- CSV summary ----
    csv_path = os.path.join(out_dir, f"inhibition_diagnostics_{session}.csv")
    rows = [
        ("session", session),
        ("conn_file", os.path.basename(args.conn)),
        ("n_neurons", n),
        ("K", K),
        ("threshold_mode", thr_mode),
        ("existing_pooled_threshold", global_thr),
        ("fdr_target", q),
        ("n_true_edges_total", n_true_total),
        ("presyn_E", nE),
        ("presyn_I", nI),
        ("presyn_mixed_sign", n_mixed),
        # P0 coverage
        ("coverage_E", e_cov),
        ("coverage_E_covered", e_cnt[0]),
        ("coverage_E_total", e_cnt[1]),
        ("coverage_I", i_cov),
        ("coverage_I_covered", i_cnt[0]),
        ("coverage_I_total", i_cnt[1]),
        ("coverage_gap_points", gap_points),
        # P0 detection decomposition
        ("auc_all", auc_all),
        ("ap_all", ap_all),
        ("auc_E", auc_E),
        ("ap_E", ap_E),
        ("auc_I", auc_I),
        ("ap_I", ap_I),
        # P0 signal vs null
        ("I_signal_vs_null_auc", sig_auc_I),
        ("I_signal_vs_null_overlap", ovl_I),
        ("E_signal_vs_null_auc", sig_auc_E),
        ("E_signal_vs_null_overlap", ovl_E),
        # within-type magnitude recovery (honest under oracle sign routing)
        ("magnitude_corr_E_withintype", magcorr_E),
        ("magnitude_corr_I_withintype", magcorr_I),
        ("n_candidates_total", int(sc.size)),
        ("n_candidates_I", int(I_sc.size)),
        ("n_true_I_in_candidates", n_true_I),
        # P1a thresholding (I edges)
        ("pooled_fdr_floor", pooled_floor),
        ("I_only_fdr_floor", strat_floor),
        ("existing_pooled_I_selected", ex_true),
        ("existing_pooled_I_fdr", ex_fdr),
        ("pooled_fdr_threshold", pooled_thr),
        ("pooled_fdr_I_selected", p_true),
        ("pooled_fdr_I_fdr", p_fdr),
        ("stratified_I_threshold", strat_thr),
        ("stratified_I_selected", strat_true),
        ("stratified_I_fdr", strat_fdr),
    ]
    # FDR-power sweep rows (stratified-I and pooled-restricted-to-I)
    for r in I_sweep:
        rows.append((f"sweep_stratifiedI_fdr{r['target']}_Iselected", r["n_true_selected"]))
        rows.append((f"sweep_stratifiedI_fdr{r['target']}_realizedfdr", r["realized_fdr"]))
    for r in pooled_sweep:
        rows.append((f"sweep_pooledI_fdr{r['target']}_Iselected", r["n_true_selected"]))
        rows.append((f"sweep_pooledI_fdr{r['target']}_realizedfdr", r["realized_fdr"]))
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("metric,value\n")
        for k, v in rows:
            fh.write(f"{k},{v}\n")
    print(f"[csv] {csv_path}")

    print("\nNOTE: P1a per-type null is built from true-negative candidate scores, which leak\n"
          "common-input structure (shared-parent / within-cluster enrichment). FDR figures are a\n"
          "DIRECTIONAL check only; the calibrated circular-shift per-type null is P1b.")


if __name__ == "__main__":
    main()
