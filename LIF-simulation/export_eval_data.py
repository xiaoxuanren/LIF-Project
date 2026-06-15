"""Export per-edge SIGNED scores + signed ground truth, aligned to evaluate_connectivity order."""
import csv, glob, os
import numpy as np

SESSION = r"D:/HAI Lab/2026/LIF model/05 May 2026/0526 2026/LIF-simulation/LIF data/20260523_084407"
VOLT = "voltage_augmented_learned_lif_outputs/connectivity_20260523_084407_voltage_all20_K100.npz"
SPK = "learned_lif_outputs/connectivity_20260523_084407_may0523_all20_K100.npz"
OUTDIR = "eval_export"
os.makedirs(OUTDIR, exist_ok=True)

net = np.load(glob.glob(SESSION + "/network_*.npz")[0], allow_pickle=True)
conns = net["connections"]; n = len(net["neuron_positions"])

# build_ground_truth (inlined; identical to lif_inference.shared_data.build_ground_truth)
W = np.zeros((n, n), np.float32); B = np.zeros((n, n), np.int32)
for c in conns:
    pre, post = int(c[0]), int(c[1]); W[post, pre] = float(c[2]); B[post, pre] = 1
pre_is_inh = (W < 0).any(axis=0)

zv = np.load(VOLT, allow_pickle=True); zs = np.load(SPK, allow_pickle=True)
conn_volt = zv["connectivity_matrix"].astype(np.float32)
conn_spike = zs["connectivity_matrix"].astype(np.float32)
ni = [np.asarray(a, int) for a in zs["neighbor_indices"]]   # candidates (same for both all-20 runs)
neuron_ids = np.arange(n)

# flatten in EXACT evaluate_connectivity order: for neuron_id in neuron_ids: for pre in neighbor_indices[neuron_id]
post_flat = np.concatenate([np.full(len(ni[j]), j, int) for j in neuron_ids])
pre_flat = np.concatenate([ni[j] for j in neuron_ids])
flat_label = B[post_flat, pre_flat].astype(np.int32)              # == evaluate_connectivity 'labels'
flat_true_sign = np.sign(W[post_flat, pre_flat]).astype(np.int8)  # +1 exc edge, -1 inh edge, 0 non-edge
flat_pre_is_inh = pre_is_inh[pre_flat]
flat_signed_volt = conn_volt[post_flat, pre_flat]
flat_signed_spike = conn_spike[post_flat, pre_flat]

np.savez_compressed(
    os.path.join(OUTDIR, "connectivity_eval_export_20260523_084407.npz"),
    neuron_ids=neuron_ids, pre_is_inh=pre_is_inh, true_W_signed=W, true_binary=B,
    neighbor_indices=np.array(ni, dtype=object),
    conn_signed_voltage=conn_volt, conn_signed_spikeonly=conn_spike,
    flat_post=post_flat, flat_pre=pre_flat, flat_label=flat_label, flat_true_sign=flat_true_sign,
    flat_pre_is_inh=flat_pre_is_inh, flat_signed_voltage=flat_signed_volt, flat_signed_spikeonly=flat_signed_spike)

with open(os.path.join(OUTDIR, "per_edge_scores_voltage_all20.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["post", "pre", "label", "true_sign", "pre_is_inh",
                "signed_score_voltage", "abs_score_voltage", "signed_score_spikeonly"])
    for i in range(len(post_flat)):
        w.writerow([int(post_flat[i]), int(pre_flat[i]), int(flat_label[i]), int(flat_true_sign[i]),
                    int(flat_pre_is_inh[i]), f"{flat_signed_volt[i]:.6g}",
                    f"{abs(flat_signed_volt[i]):.6g}", f"{flat_signed_spike[i]:.6g}"])

print("neurons=%d (exc_src=%d inh_src=%d) | candidate edges=%d | true exc=%d inh=%d" % (
    n, int((~pre_is_inh & (W > 0).any(axis=0)).sum()), int(pre_is_inh.sum()), len(post_flat),
    int((flat_true_sign == 1).sum()), int((flat_true_sign == -1).sum())))
print("wrote", OUTDIR + "/connectivity_eval_export_20260523_084407.npz + per_edge_scores_voltage_all20.csv")
