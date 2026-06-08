"""False-positive diagnostic for learned-LIF connectivity inference.

Reads a saved ``connectivity_<session>_<tag>.npz`` (and the session's
``network_*.npz`` for cluster labels) and characterises *why* false-positive
edges occur: are they enriched for within-cluster pairs, short distances, or
shared common input (a third neuron projecting to both)? Shared common input is
the signature of the common-input confound that per-neuron-independent fitting
cannot resolve.

Pure NumPy + Matplotlib; operates on already-saved outputs. No torch, no model
re-run, no GPU.

Usage:
    python fp_diagnostic.py --conn "path/to/connectivity_<session>_<tag>.npz" \
        --session "LIF data/20260604_023028" \
        --out fp_diagnosis.png
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_array(x):
    a = np.asarray(x)
    return a.item() if a.ndim == 0 else a


def load_connectivity(conn_path):
    d = np.load(conn_path, allow_pickle=True)
    C = np.asarray(d["connectivity_matrix"], dtype=float)        # [post, pre] signed
    Wtrue = np.asarray(d["true_weights"], dtype=float)           # [post, pre] signed
    NB = np.asarray(d["neighbor_indices"])                       # [post, K] candidate pre
    positions = np.asarray(d["neuron_positions"], dtype=float) if "neuron_positions" in d.files else None
    global_thr = float(_as_array(d["threshold"])) if "threshold" in d.files else None
    # Optional per-neuron thresholds (surrogate-FDR mode)
    thr_by_post = None
    if "per_neuron_ids" in d.files and "per_neuron_thresholds" in d.files:
        ids = np.asarray(d["per_neuron_ids"]).ravel()
        thr = np.asarray(d["per_neuron_thresholds"], dtype=float).ravel()
        if ids.size and ids.size == thr.size:
            thr_by_post = {int(i): float(t) for i, t in zip(ids, thr)}
    return C, Wtrue, NB, positions, global_thr, thr_by_post


def load_clusters(session_dir):
    if not session_dir:
        return None
    nets = glob.glob(os.path.join(session_dir, "network_*.npz"))
    if not nets:
        print(f"  (no network_*.npz in {session_dir}; skipping cluster analysis)")
        return None
    net = np.load(nets[0], allow_pickle=True)
    return np.asarray(net["cluster_assignments"]).ravel()


def threshold_for(post, global_thr, thr_by_post):
    if thr_by_post is not None and post in thr_by_post:
        return thr_by_post[post]
    return global_thr if global_thr is not None else 0.5


def build_pairs(C, Wtrue, NB, global_thr, thr_by_post):
    """Flatten candidate (post, pre) pairs with predicted/true labels."""
    n = C.shape[0]
    parents = (np.abs(Wtrue) > 0).astype(np.int16)  # parents[post, pre] = 1 if true pre->post
    posts, pres, scores, preds, trues = [], [], [], [], []
    for j in range(n):
        thr = threshold_for(j, global_thr, thr_by_post)
        for p in np.asarray(NB[j]).ravel():
            p = int(p)
            if p == j:
                continue
            s = abs(float(C[j, p]))
            posts.append(j); pres.append(p)
            scores.append(s)
            preds.append(s >= thr)
            trues.append(abs(Wtrue[j, p]) > 0)
    posts = np.array(posts); pres = np.array(pres)
    scores = np.array(scores); preds = np.array(preds, bool); trues = np.array(trues, bool)
    return posts, pres, scores, preds, trues, parents


def shared_parents_for_pairs(parents, posts, pres, chunk=4000):
    """Count common presynaptic neurons projecting to BOTH post and pre, per pair."""
    out = np.zeros(len(posts), dtype=np.int32)
    for s in range(0, len(posts), chunk):
        e = min(s + chunk, len(posts))
        out[s:e] = (parents[posts[s:e], :] * parents[pres[s:e], :]).sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", required=True, help="connectivity_<session>_<tag>.npz")
    ap.add_argument("--session", default=None, help="session dir (for cluster labels)")
    ap.add_argument("--out", default="fp_diagnosis.png")
    args = ap.parse_args()

    C, Wtrue, NB, positions, global_thr, thr_by_post = load_connectivity(args.conn)
    clusters = load_clusters(args.session)
    print(f"n_neurons={C.shape[0]}  K={NB.shape[1] if NB.ndim > 1 else '?'}  "
          f"global_thr={global_thr}  per_neuron_thr={'yes' if thr_by_post else 'no'}")

    posts, pres, scores, preds, trues, parents = build_pairs(C, Wtrue, NB, global_thr, thr_by_post)

    TP = preds & trues
    FP = preds & ~trues
    FN = ~preds & trues
    TN = ~preds & ~trues
    print(f"candidate pairs={len(posts)}  TP={TP.sum()}  FP={FP.sum()}  FN={FN.sum()}  TN={TN.sum()}")

    # Features
    shared = shared_parents_for_pairs(parents, posts, pres)
    if positions is not None:
        dist = np.linalg.norm(positions[posts] - positions[pres], axis=1)
    else:
        dist = np.full(len(posts), np.nan)
    same_cluster = (clusters[posts] == clusters[pres]) if clusters is not None else None
    # direction error: FP pair (pre->post not true) but reverse (post->pre) IS true
    reverse_true = np.abs(Wtrue[pres, posts]) > 0

    def frac(mask_sub, base_mask):
        b = base_mask.sum()
        return float(mask_sub[base_mask].mean()) if b else float("nan")

    print("\n=== Why are non-edges falsely called? (FP vs TN among true non-edges) ===")
    print(f"{'feature':28s} {'FP':>10s} {'TN':>10s} {'enrich(FP/TN)':>14s}")
    # shared input
    fp_share = shared[FP].mean() if FP.sum() else float("nan")
    tn_share = shared[TN].mean() if TN.sum() else float("nan")
    print(f"{'mean shared parents':28s} {fp_share:10.3f} {tn_share:10.3f} {fp_share / tn_share if tn_share else float('nan'):14.2f}")
    fp_share1 = frac(shared >= 1, FP); tn_share1 = frac(shared >= 1, TN)
    print(f"{'frac with >=1 shared parent':28s} {fp_share1:10.3f} {tn_share1:10.3f} {fp_share1 / tn_share1 if tn_share1 else float('nan'):14.2f}")
    if same_cluster is not None:
        fp_sc = frac(same_cluster, FP); tn_sc = frac(same_cluster, TN)
        print(f"{'frac within-cluster':28s} {fp_sc:10.3f} {tn_sc:10.3f} {fp_sc / tn_sc if tn_sc else float('nan'):14.2f}")
    if np.isfinite(dist).any():
        print(f"{'mean distance':28s} {dist[FP].mean():10.3f} {dist[TN].mean():10.3f} {'(lower=closer)':>14s}")
    print(f"\nFP that are reverse-of-true (direction errors): "
          f"{reverse_true[FP].mean() if FP.sum() else float('nan'):.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    maxsh = int(min(shared.max(), 8)) if shared.size else 1
    bins = np.arange(0, maxsh + 2) - 0.5
    ax[0, 0].hist(shared[TN], bins=bins, density=True, alpha=0.6, label=f"TN (n={TN.sum()})", color="#999")
    ax[0, 0].hist(shared[FP], bins=bins, density=True, alpha=0.6, label=f"FP (n={FP.sum()})", color="#c1121f")
    ax[0, 0].set_xlabel("# shared presynaptic parents (common input)")
    ax[0, 0].set_ylabel("density"); ax[0, 0].set_title("Shared common input: FP vs TN"); ax[0, 0].legend()

    if np.isfinite(dist).any():
        ax[0, 1].hist(dist[TN], bins=30, density=True, alpha=0.6, label="TN", color="#999")
        ax[0, 1].hist(dist[FP], bins=30, density=True, alpha=0.6, label="FP", color="#c1121f")
        ax[0, 1].set_xlabel("pairwise distance"); ax[0, 1].set_ylabel("density")
        ax[0, 1].set_title("Distance: FP vs TN"); ax[0, 1].legend()
    else:
        ax[0, 1].axis("off"); ax[0, 1].text(0.5, 0.5, "no positions", ha="center")

    cats = ["TP", "FP", "TN"]
    masks = [TP, FP, TN]
    if same_cluster is not None:
        vals = [frac(same_cluster, m) for m in masks]
        ax[1, 0].bar(cats, vals, color=["#2a9d8f", "#c1121f", "#999"])
        ax[1, 0].set_ylabel("fraction within-cluster"); ax[1, 0].set_title("Within-cluster fraction by category")
        ax[1, 0].set_ylim(0, 1)
    else:
        ax[1, 0].axis("off"); ax[1, 0].text(0.5, 0.5, "no cluster labels", ha="center")

    # text summary
    ax[1, 1].axis("off")
    lines = [
        "FALSE-POSITIVE DIAGNOSIS", "=" * 34,
        f"TP={TP.sum()}  FP={FP.sum()}  FN={FN.sum()}  TN={TN.sum()}",
        f"precision={TP.sum()/(TP.sum()+FP.sum()):.3f}  recall={TP.sum()/(TP.sum()+FN.sum()):.3f}",
        "",
        f"FP mean shared-parents : {fp_share:.3f}  (TN {tn_share:.3f})",
        f"FP frac >=1 shared      : {fp_share1:.3f}  (TN {tn_share1:.3f})",
    ]
    if same_cluster is not None:
        lines.append(f"FP frac within-cluster  : {frac(same_cluster, FP):.3f}  (TN {frac(same_cluster, TN):.3f})")
    lines += [
        f"FP reverse-of-true      : {reverse_true[FP].mean() if FP.sum() else float('nan'):.3f}",
        "",
        "Read: if FP >> TN on shared-parents / within-cluster,",
        "the false positives are common-input confounds ->",
        "Dale tying / graph-level structure is the fix.",
    ]
    ax[1, 1].text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)

    fig.suptitle(f"FP diagnostic: {os.path.basename(args.conn)}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
