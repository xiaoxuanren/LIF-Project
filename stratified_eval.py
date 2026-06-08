"""Stratified comparison of spike-only vs voltage-augmented connectivity recovery.

Breaks detection performance out by true synaptic |weight| tertile and
presynaptic firing-rate tertile, to locate *where* voltage supervision earns its
keep. Hypothesis: voltage helps most on weak and/or low-rate synapses (whose
EPSPs rarely trigger a postsynaptic spike but are still visible in V), even if
aggregate AUC/AP looks flat.

Reads two saved ``connectivity_*.npz`` files (the lambda=0 spike-only run and the
lambda=1 voltage run, same session) plus the session recordings for rates.
Pure NumPy + Matplotlib; no torch, no model re-run.

Usage:
    python stratified_eval.py \
        --spikeonly "path/connectivity_<session>_..SPIKEONLY..npz" \
        --voltage   "path/connectivity_<session>_..voltage..npz" \
        --session   "LIF data/20260604_023028" \
        --out stratified_voltage_vs_spikeonly.png
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _scalar(d, key, default=None):
    if key not in d.files:
        return default
    a = np.asarray(d[key])
    return a.item() if a.ndim == 0 else a


def load_run(path):
    d = np.load(path, allow_pickle=True)
    C = np.asarray(d["connectivity_matrix"], dtype=float)
    NB = np.asarray(d["neighbor_indices"])
    Wtrue = np.asarray(d["true_weights"], dtype=float)
    global_thr = float(_scalar(d, "threshold", 0.5))
    thr_by_post = None
    if "per_neuron_ids" in d.files and "per_neuron_thresholds" in d.files:
        ids = np.asarray(d["per_neuron_ids"]).ravel()
        thr = np.asarray(d["per_neuron_thresholds"], dtype=float).ravel()
        if ids.size and ids.size == thr.size:
            thr_by_post = {int(i): float(t) for i, t in zip(ids, thr)}
    return C, NB, Wtrue, global_thr, thr_by_post


def presynaptic_rates(session_dir, n_neurons):
    recs = sorted(glob.glob(os.path.join(session_dir, "recording[0-9][0-9][0-9].npz")))
    if not recs:
        raise FileNotFoundError(f"no recordings in {session_dir}")
    counts = np.zeros(n_neurons, dtype=float)
    total_ms = 0.0
    for r in recs:
        data = np.load(r, allow_pickle=True)
        st = data["spike_times"]
        total_ms += float(data["duration"])
        for i in range(n_neurons):
            try:
                counts[i] += len(np.asarray(st[i]).ravel())
            except (TypeError, IndexError):
                pass
    return counts / (total_ms / 1000.0)


def tertile_labels(values):
    """Return integer tertile index (0,1,2) per value, robust to ties."""
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    lab = np.zeros(len(values), dtype=int)
    lab[values > q1] = 1
    lab[values > q2] = 2
    return lab, (q1, q2)


def predicted(C, NB, global_thr, thr_by_post, j, p):
    thr = thr_by_post[j] if (thr_by_post is not None and j in thr_by_post) else global_thr
    return abs(C[j, p]) >= thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spikeonly", required=True)
    ap.add_argument("--voltage", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default="stratified_voltage_vs_spikeonly.png")
    args = ap.parse_args()

    Cs, NBs, Wt, thr_s, thrpn_s = load_run(args.spikeonly)
    Cv, NBv, Wtv, thr_v, thrpn_v = load_run(args.voltage)
    assert Wt.shape == Wtv.shape, "runs are from different networks"
    n = Cv.shape[0]
    rates = presynaptic_rates(args.session, n)

    # candidate true edges (recoverable): true and in candidate set (use voltage NB as reference)
    edges = []  # (post j, pre p, |w|, prerate)
    n_true_total = int((np.abs(Wt) > 0).sum())
    for j in range(n):
        cand = set(int(x) for x in np.asarray(NBv[j]).ravel())
        for p in np.where(np.abs(Wt[j]) > 0)[0]:
            p = int(p)
            if p in cand and p != j:
                edges.append((j, p, abs(Wt[j, p]), rates[p]))
    edges = np.array(edges, dtype=float)
    js = edges[:, 0].astype(int); ps = edges[:, 1].astype(int)
    w = edges[:, 2]; pr = edges[:, 3]
    print(f"true edges total={n_true_total}, recoverable (in candidate set)={len(edges)} "
          f"-> {len(edges)/n_true_total:.1%}; the rest are pruned by the candidate screen")

    pred_s = np.array([predicted(Cs, NBs, thr_s, thrpn_s, j, p) for j, p in zip(js, ps)])
    pred_v = np.array([predicted(Cv, NBv, thr_v, thrpn_v, j, p) for j, p in zip(js, ps)])
    score_s = np.array([abs(Cs[j, p]) for j, p in zip(js, ps)])
    score_v = np.array([abs(Cv[j, p]) for j, p in zip(js, ps)])

    wlab, (wq1, wq2) = tertile_labels(w)
    rlab, (rq1, rq2) = tertile_labels(pr)

    print(f"\nweight tertile cuts: |w| <= {wq1:.3f} (low), <= {wq2:.3f} (mid), > (high)")
    print(f"pre-rate tertile cuts: <= {rq1:.2f} Hz (low), <= {rq2:.2f} Hz (mid), > (high)")

    def recall(mask, pred):
        return float(pred[mask].mean()) if mask.sum() else float("nan")

    # 3x3 recall grid (rows=weight tertile, cols=rate tertile)
    grid_s = np.full((3, 3), np.nan); grid_v = np.full((3, 3), np.nan); grid_n = np.zeros((3, 3), int)
    for wi in range(3):
        for ri in range(3):
            m = (wlab == wi) & (rlab == ri)
            grid_n[wi, ri] = m.sum()
            grid_s[wi, ri] = recall(m, pred_s)
            grid_v[wi, ri] = recall(m, pred_v)
    delta = grid_v - grid_s

    names = ["low", "mid", "high"]
    print("\n=== recall by (weight tertile x pre-rate tertile) ===")
    print("rows = true |weight| tertile, cols = presynaptic rate tertile")
    for tag, g in [("SPIKE-ONLY", grid_s), ("VOLTAGE", grid_v), ("DELTA (V - S)", delta)]:
        print(f"\n{tag}")
        print(f"{'':6s}" + "".join(f"{'rate ' + names[r]:>12s}" for r in range(3)))
        for wi in range(3):
            print(f"w {names[wi]:4s}" + "".join(f"{g[wi, ri]:12.3f}" for ri in range(3)))

    # marginals
    print("\n=== marginal recall ===")
    for label, lab in [("weight", wlab), ("pre-rate", rlab)]:
        print(f"by {label} tertile:   spike-only / voltage / delta")
        for t in range(3):
            m = lab == t
            print(f"  {names[t]:4s} (n={m.sum():4d}):  {recall(m, pred_s):.3f} / {recall(m, pred_v):.3f} / {recall(m, pred_v) - recall(m, pred_s):+.3f}")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    im = ax[0, 0].imshow(delta, cmap="RdBu_r", vmin=-max(0.05, np.nanmax(np.abs(delta))),
                         vmax=max(0.05, np.nanmax(np.abs(delta))), origin="lower")
    ax[0, 0].set_xticks(range(3)); ax[0, 0].set_xticklabels([f"rate {x}" for x in names])
    ax[0, 0].set_yticks(range(3)); ax[0, 0].set_yticklabels([f"|w| {x}" for x in names])
    for wi in range(3):
        for ri in range(3):
            ax[0, 0].text(ri, wi, f"{delta[wi, ri]:+.2f}\n(n={grid_n[wi, ri]})", ha="center", va="center", fontsize=9)
    ax[0, 0].set_title("Recall gain from voltage (V - spike-only)")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046)

    x = np.arange(3); wbar = 0.35
    ms = [recall(wlab == t, pred_s) for t in range(3)]
    mv = [recall(wlab == t, pred_v) for t in range(3)]
    ax[0, 1].bar(x - wbar / 2, ms, wbar, label="spike-only", color="#999")
    ax[0, 1].bar(x + wbar / 2, mv, wbar, label="voltage", color="#2a9d8f")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(names); ax[0, 1].set_xlabel("true |weight| tertile")
    ax[0, 1].set_ylabel("recall"); ax[0, 1].set_title("Recall by weight tertile"); ax[0, 1].legend(); ax[0, 1].set_ylim(0, 1)

    rs = [recall(rlab == t, pred_s) for t in range(3)]
    rv = [recall(rlab == t, pred_v) for t in range(3)]
    ax[1, 0].bar(x - wbar / 2, rs, wbar, label="spike-only", color="#999")
    ax[1, 0].bar(x + wbar / 2, rv, wbar, label="voltage", color="#2a9d8f")
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(names); ax[1, 0].set_xlabel("presynaptic rate tertile")
    ax[1, 0].set_ylabel("recall"); ax[1, 0].set_title("Recall by pre-rate tertile"); ax[1, 0].legend(); ax[1, 0].set_ylim(0, 1)

    # mean predicted score for true edges, by weight tertile (does voltage lift weak-synapse scores?)
    sms = [score_s[wlab == t].mean() for t in range(3)]
    smv = [score_v[wlab == t].mean() for t in range(3)]
    ax[1, 1].bar(x - wbar / 2, sms, wbar, label="spike-only", color="#999")
    ax[1, 1].bar(x + wbar / 2, smv, wbar, label="voltage", color="#2a9d8f")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(names); ax[1, 1].set_xlabel("true |weight| tertile")
    ax[1, 1].set_ylabel("mean predicted |score| (true edges)")
    ax[1, 1].set_title("Score on true edges by weight tertile"); ax[1, 1].legend()

    fig.suptitle("Where does voltage help? spike-only vs voltage-augmented", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
