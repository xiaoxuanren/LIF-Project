"""Standalone validation of the driver-free spontaneous-bursting LIF setup.

This script proves three things about the current spontaneous-mode operating
point and writes every raster plus a full parameter/results table to
``validation_outputs/``:

  A. "The setting works" - the driver-free, noise-only network produces discrete,
     network-wide spontaneous bursts on a sparse (non-silent, non-runaway)
     baseline.
  B. "No stimulation is used" - the run carries no stimulation events, every
     neuron has ``i_baseline == 0`` and ``i_ext == 0`` at all times, a noise-off
     copy of the identical network is completely silent, and two different noise
     seeds ignite bursts at different times (stochastic, not scheduled).
  C. "Changing adaptation reproduces a 4-AP-like state" - on the *same* wiring,
     lowering the spike-triggered adaptation increment (a point-neuron proxy for
     4-AP's K+/AHP block) raises excitability and firing.

The network is built exactly the way the spontaneous branch of
``sequential_simulation_individual_saves`` builds it: ``create_clustered_network``
followed by ``assign_baseline_drive`` (drivers OFF), ``scale_adaptation_dynamics``,
and ``scale_excitatory_weights``, then ``simulate_network`` with an empty
stimulation list.

Reproducibility model
---------------------
A single fixed build seed (``BUILD_SEED``) is applied via ``np.random.seed`` before
*every* ``create_clustered_network`` call, so the wiring and the resting initial
voltages are byte-identical across all runs. The only thing varied between runs is
(1) the membrane-noise RNG seed set immediately before ``simulate_network`` and
(2) the adaptation ``increment_scale``. This is what lets Validation B vary noise
while holding wiring fixed, and Validation C vary adaptation while holding wiring
fixed.

Run:  python -m scripts.validate_spontaneous_bursts
(from the LIF-simulation directory, or run the file directly).
"""

import contextlib
import csv
import io
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lif_simulation.analysis import segment_states
from lif_simulation.network import (
    assign_baseline_drive,
    create_clustered_network,
    scale_adaptation_dynamics,
    scale_excitatory_weights,
)
from lif_simulation.simulation import simulate_network

# --------------------------------------------------------------------------- #
# Frozen validated configuration (the exact spontaneous-mode operating point).
# --------------------------------------------------------------------------- #
BUILD_SEED = 7  # fixed across ALL builds -> identical wiring + identical rested v.

NETWORK_KWARGS = dict(
    num_clusters=15,
    neurons_per_cluster_range=(12, 18),
    inhibitory_probability=0.2,
    within_cluster_prob=0.5,
    between_cluster_prob=0.15,
    max_connection_distance=8.0,
    space_size=15,
    hub_fraction=0.10,
    hub_between_prob=0.40,
    hub_weight_scale=1.5,
    hub_reciprocal_factor=2.0,
    use_h_current=True,
    background_noise_sigma=1.4,
    depressing=True,
    delta_q=0.8,
    tau_q=1500,
)

# Spontaneous-mode transforms (applied exactly as the workflow applies them).
BASELINE_MEAN = 0.0          # NO DRIVERS -> every i_baseline == 0
BASELINE_SD = 0.0
BASELINE_SEED = 0
ADAPT_TAU_SCALE = 3.0        # baseline adaptation recovery
ADAPT_INCREMENT_SCALE = 1.0  # baseline adaptation strength (the "4-AP knob")
EXC_WEIGHT_SCALE = 4.0       # recurrent gain

DT = 0.1
BURN_IN_MS = 1000.0          # drop the t~0 startup transient before scoring
BIN_MS = 10.0                # burst-detection bin
AF_THRESH = 0.12             # active-fraction burst threshold

# Run durations (ms).
DUR_VALIDATED = 30000.0      # Validation A (extend to 60000 for a final figure)
DUR_NOISE_OFF = 3000.0       # Validation B.2
DUR_SEED_CTRL = 3000.0       # Validation B.3
DUR_ADAPT = 6000.0           # Validation C sweep

# Noise (membrane-RNG) seeds. Build seed is always BUILD_SEED; only these vary.
NOISE_SEED_A = 100           # validated run + seed-control run "A"
NOISE_SEED_B = 200           # seed-control run "B" (must ignite differently)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "validation_outputs"


# --------------------------------------------------------------------------- #
# Build / run helpers.
# --------------------------------------------------------------------------- #
def build_validated_network(increment_scale=ADAPT_INCREMENT_SCALE, noise_sigma=None):
    """Build the validated spontaneous-mode network exactly as the workflow does.

    Args:
        increment_scale: Adaptation-increment multiplier (the 4-AP knob). 1.0 is
            the baseline validated value; lowering it is the 4-AP direction.
        noise_sigma: Optional override for ``background_noise_sigma`` (used by the
            noise-off control). ``None`` keeps the validated value.

    Returns:
        Tuple ``(neurons, synapses, connections, positions, cluster_info)``. The
        global NumPy RNG is seeded with ``BUILD_SEED`` first, so wiring and rested
        initial voltages are identical across every call with the same arguments.
    """
    kwargs = dict(NETWORK_KWARGS)
    if noise_sigma is not None:
        kwargs["background_noise_sigma"] = noise_sigma

    np.random.seed(BUILD_SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        neurons, synapses, connections, positions, cluster_info = create_clustered_network(**kwargs)
        # Drivers OFF: mean=sd=0 -> every i_baseline is exactly 0.0.
        assign_baseline_drive(neurons, mean=BASELINE_MEAN, sd=BASELINE_SD, seed=BASELINE_SEED)
        scale_adaptation_dynamics(neurons, tau_scale=ADAPT_TAU_SCALE, increment_scale=increment_scale)
        scale_excitatory_weights(synapses, EXC_WEIGHT_SCALE, connections)
    return neurons, synapses, connections, positions, cluster_info


def run_spontaneous(neurons, synapses, duration_ms, noise_seed):
    """Run one driver-free, stimulation-free simulation.

    Args:
        neurons: Neuron list from :func:`build_validated_network`.
        synapses: Synapse list from :func:`build_validated_network`.
        duration_ms: Simulated duration in milliseconds.
        noise_seed: Seed applied to the global RNG immediately before simulating,
            which fixes the membrane-noise realization. The rested initial voltages
            were already frozen at build time, so the t~0 startup event is identical
            regardless of this seed.

    Returns:
        The ``spike_data`` dict mapping neuron id -> list of spike times (ms).
        ``stimulation_events`` is the empty list ``[]`` in every run.
    """
    np.random.seed(noise_seed)
    stimulation_events = []  # NO stimulation, ever.
    with contextlib.redirect_stdout(io.StringIO()):
        spike_data, _ = simulate_network(
            neurons,
            synapses,
            stimulation_events,
            dt=DT,
            duration=duration_ms,
            record_voltage=False,  # spikes only -> fast
        )
    return spike_data


# --------------------------------------------------------------------------- #
# Diagnostics.
# --------------------------------------------------------------------------- #
def _count_in_windows(times, windows):
    """Count how many spike times fall inside a set of non-overlapping windows.

    Args:
        times: 1-D array of spike times (ms).
        windows: Sorted, non-overlapping ``(start_ms, end_ms)`` pairs.

    Returns:
        Integer count of ``times`` that lie inside any window (``start <= t < end``).
    """
    if len(times) == 0 or not windows:
        return 0
    starts = np.array([w[0] for w in windows])
    ends = np.array([w[1] for w in windows])
    idx = np.searchsorted(starts, times, side="right") - 1
    valid = idx >= 0
    inside = np.zeros(len(times), dtype=bool)
    inside[valid] = times[valid] < ends[idx[valid]]
    return int(np.sum(inside))


def compute_diagnostics(spike_data, n_neurons, duration_ms):
    """Compute the required per-run spontaneous-bursting diagnostics.

    All burst statistics and the runaway check are computed on the POST-BURN-IN
    window (the first ``BURN_IN_MS`` are dropped) because the t~0 event is a
    startup transient from identical rested initial conditions, not a real burst.

    Args:
        spike_data: Mapping neuron id -> spike times (ms) over the full run.
        n_neurons: Number of neurons in the network.
        duration_ms: Full simulated duration (ms).

    Returns:
        Dict with mean firing rate, network-burst frequency, steady-state peak
        active fraction, percent of spikes in bursts, inter-burst per-neuron rate,
        a runaway flag, the per-2 s window rates, and the total spike count.
    """
    eff_dur = duration_ms - BURN_IN_MS
    all_times_full = np.concatenate(
        [np.asarray(s, dtype=float) for s in spike_data.values()]
    ) if any(len(s) for s in spike_data.values()) else np.array([])
    n_spikes_full = int(all_times_full.size)

    # Post-burn-in spikes, shifted so the window starts at 0.
    pb_lists = []
    for spikes in spike_data.values():
        arr = np.asarray(spikes, dtype=float)
        arr = arr[arr >= BURN_IN_MS] - BURN_IN_MS
        pb_lists.append(arr)
    pb_all = np.concatenate(pb_lists) if pb_lists else np.array([])
    total_pb = int(pb_all.size)

    mean_rate = total_pb / n_neurons / (eff_dur / 1000.0)

    # Active-fraction per bin (post burn-in) -> steady-state peak AF.
    n_bins = int(np.ceil(eff_dur / BIN_MS))
    active_count = np.zeros(n_bins)
    for arr in pb_lists:
        if arr.size == 0:
            continue
        idx = np.floor(arr / BIN_MS).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        active_count[np.unique(idx)] += 1
    af = active_count / n_neurons
    peak_af = float(af.max()) if n_bins > 0 else 0.0

    # Burst / inter-burst segmentation on the post-burn-in window.
    shifted = {nid: arr for nid, arr in zip(spike_data.keys(), pb_lists)}
    burst_windows, interburst_windows = segment_states(
        shifted, n_neurons, eff_dur,
        bin_ms=BIN_MS, burst_frac_thresh=AF_THRESH, merge_gap_ms=30.0, min_burst_ms=5.0,
    )
    n_bursts = len(burst_windows)
    burst_freq = n_bursts / (eff_dur / 1000.0)

    spikes_in_burst = _count_in_windows(pb_all, burst_windows)
    pct_in_burst = 100.0 * spikes_in_burst / total_pb if total_pb > 0 else 0.0

    ib_dur_ms = sum(e - s for s, e in interburst_windows)
    spikes_in_ib = _count_in_windows(pb_all, interburst_windows)
    ib_rate = (spikes_in_ib / n_neurons / (ib_dur_ms / 1000.0)) if ib_dur_ms > 0 else 0.0

    # Runaway check: per-2 s window per-neuron rate, first-third vs last-third.
    win = 2000.0
    nwin = int(np.floor(eff_dur / win))
    win_rates = []
    for w in range(nwin):
        ws = BURN_IN_MS + w * win
        we = ws + win
        c = int(np.sum((all_times_full >= ws) & (all_times_full < we)))
        win_rates.append(c / n_neurons / (win / 1000.0))
    if nwin >= 2:
        k = max(1, nwin // 3)
        first = float(np.mean(win_rates[:k]))
        last = float(np.mean(win_rates[-k:]))
        runaway = bool(last > 1.5 * max(first, 1e-6) and (last - first) > 0.5)
    else:
        first = last = win_rates[0] if win_rates else 0.0
        runaway = False

    return {
        "mean_rate_Hz": mean_rate,
        "burst_freq_Hz": burst_freq,
        "n_bursts": n_bursts,
        "peak_AF_steady": peak_af,
        "pct_in_burst": pct_in_burst,
        "IB_rate_Hz": ib_rate,
        "runaway_flag": runaway,
        "win_rates": win_rates,
        "win_first": first,
        "win_last": last,
        "n_spikes": n_spikes_full,
        "burst_windows": burst_windows,
    }


def compute_connectivity(connections, cluster_info, n_neurons):
    """Measure the realized connectivity of the built graph.

    Args:
        connections: Object array of ``[pre, post, weight, type]`` rows.
        cluster_info: Cluster metadata (needs ``cluster_assignments`` and
            ``hub_neuron_ids``).
        n_neurons: Total neuron count.

    Returns:
        Dict with realized within-cluster prob, realized (non-hub) between-cluster
        prob, hub share of inter-cluster edges, and overall density. Between-cluster
        edges are dominated by hub projections, so ``realized_between`` reports the
        base (non-hub-origin) probability, which is the small distance-attenuated
        value; ``hub_share_inter`` captures the hub-dominated remainder.
    """
    ca = np.asarray(cluster_info["cluster_assignments"])
    hubs = set(int(h) for h in cluster_info.get("hub_neuron_ids", []))
    pre = connections[:, 0].astype(int)
    post = connections[:, 1].astype(int)
    same = ca[pre] == ca[post]

    within_edges = int(np.sum(same))
    between_edges = int(np.sum(~same))
    pre_is_hub = np.array([p in hubs for p in pre])
    between_hub = int(np.sum((~same) & pre_is_hub))
    between_base = between_edges - between_hub

    sizes = np.bincount(ca)
    within_possible = int(np.sum(sizes * (sizes - 1)))
    total_possible = n_neurons * (n_neurons - 1)
    between_possible = total_possible - within_possible

    return {
        "realized_within": within_edges / within_possible if within_possible else 0.0,
        "realized_between": between_base / between_possible if between_possible else 0.0,
        "realized_between_all": between_edges / between_possible if between_possible else 0.0,
        "hub_share_inter": between_hub / between_edges if between_edges else 0.0,
        "overall_density": len(connections) / total_possible if total_possible else 0.0,
        "within_edges": within_edges,
        "between_edges": between_edges,
        "between_hub_edges": between_hub,
    }


def connection_hash(connections):
    """Return a stable hash of the wiring (pre, post, type, rounded weight).

    Used to prove Validation C keeps identical ground-truth wiring across
    adaptation levels.

    Args:
        connections: Object array of ``[pre, post, weight, type]`` rows.

    Returns:
        An integer hash that is identical for identical wiring.
    """
    rows = []
    for c in connections:
        rows.append((int(c[0]), int(c[1]), str(c[3]), round(float(c[2]), 6)))
    return hash(tuple(sorted(rows)))


def burst_onsets(spike_data, n_neurons, duration_ms):
    """Return network-burst onset times over the FULL run (no burn-in drop).

    Kept full-window so the identical t~0 startup event is visible in the seed
    control alongside the seed-dependent later onsets.

    Args:
        spike_data: Mapping neuron id -> spike times (ms).
        n_neurons: Neuron count.
        duration_ms: Full simulated duration (ms).

    Returns:
        Sorted list of burst-window start times (ms).
    """
    sd = {nid: np.asarray(s, dtype=float) for nid, s in spike_data.items()}
    windows, _ = segment_states(
        sd, n_neurons, duration_ms,
        bin_ms=BIN_MS, burst_frac_thresh=AF_THRESH, merge_gap_ms=30.0, min_burst_ms=5.0,
    )
    return [round(w[0], 1) for w in windows]


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def _cluster_order(cluster_info, n_neurons):
    """Return neuron ids sorted by cluster and a neuron-id -> row-rank map."""
    ca = np.asarray(cluster_info["cluster_assignments"])
    order = np.argsort(ca, kind="stable")
    rank = {int(nid): i for i, nid in enumerate(order)}
    return order, rank, ca


def _scatter_raster(ax, spike_data, neurons, rank, dot=1.6):
    """Draw an exc=blue / inh=red raster (sorted rows) onto ``ax`` in seconds."""
    ex_x, ex_y, in_x, in_y = [], [], [], []
    for nid, spikes in spike_data.items():
        if len(spikes) == 0:
            continue
        y = rank[nid]
        if neurons[nid].is_inhibitory:
            in_x.extend(spikes)
            in_y.extend([y] * len(spikes))
        else:
            ex_x.extend(spikes)
            ex_y.extend([y] * len(spikes))
    if ex_x:
        ax.scatter(np.array(ex_x) / 1000.0, ex_y, s=dot, c="tab:blue",
                   marker=".", linewidths=0, label="excitatory")
    if in_x:
        ax.scatter(np.array(in_x) / 1000.0, in_y, s=dot, c="tab:red",
                   marker=".", linewidths=0, label="inhibitory")


def _active_fraction(spike_data, n_neurons, duration_ms):
    """Return (time_axis_s, active_fraction_per_bin) over the full run."""
    n_bins = int(np.ceil(duration_ms / BIN_MS))
    af = np.zeros(n_bins)
    for spikes in spike_data.values():
        if len(spikes) == 0:
            continue
        idx = np.floor(np.asarray(spikes, dtype=float) / BIN_MS).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        af[np.unique(idx)] += 1
    af /= n_neurons
    return np.arange(n_bins) * BIN_MS / 1000.0, af


def plot_raster_with_af(spike_data, neurons, cluster_info, duration_ms, title, out_path):
    """Save a cluster-sorted raster with a population active-fraction panel below.

    Args:
        spike_data: Mapping neuron id -> spike times (ms).
        neurons: Neuron list (for exc/inh coloring).
        cluster_info: Cluster metadata (for row ordering).
        duration_ms: Full simulated duration (ms).
        title: Figure title.
        out_path: PNG output path.

    Returns:
        None. Writes a PNG to ``out_path``.
    """
    n = len(neurons)
    _, rank, _ = _cluster_order(cluster_info, n)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    _scatter_raster(ax1, spike_data, neurons, rank)
    burn_s = BURN_IN_MS / 1000.0
    ax1.axvspan(0, burn_s, color="gray", alpha=0.12)
    ax1.axvline(burn_s, color="k", ls="--", lw=1)
    ax1.text(burn_s, n * 1.005, " 1 s burn-in (startup transient, excluded)",
             fontsize=9, va="bottom", ha="left", color="dimgray")
    ax1.set_ylim(-1, n)
    ax1.set_ylabel("neuron (sorted by cluster)")
    ax1.set_title(title)
    ax1.legend(loc="upper right", markerscale=6, framealpha=0.9)

    tb, af = _active_fraction(spike_data, n, duration_ms)
    ax2.fill_between(tb, af, color="0.4", lw=0)
    ax2.axhline(AF_THRESH, color="tab:orange", ls="--", lw=1.2,
                label=f"burst threshold (AF = {AF_THRESH})")
    ax2.axvspan(0, burn_s, color="gray", alpha=0.12)
    ax2.axvline(burn_s, color="k", ls="--", lw=1)
    ax2.set_ylabel("active fraction")
    ax2.set_xlabel("time (s)")
    ax2.set_xlim(0, duration_ms / 1000.0)
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seed_control(run_a, run_b, neurons, cluster_info, duration_ms,
                      onsets_a, onsets_b, out_path):
    """Save a stacked two-seed raster showing stochastic (seed-dependent) ignition."""
    n = len(neurons)
    _, rank, _ = _cluster_order(cluster_info, n)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.12})
    for ax, run, onsets, lab, seed in (
        (axes[0], run_a, onsets_a, "A", NOISE_SEED_A),
        (axes[1], run_b, onsets_b, "B", NOISE_SEED_B),
    ):
        _scatter_raster(ax, run, neurons, rank)
        for t in onsets:
            ax.axvline(t / 1000.0, color="tab:green", lw=0.6, alpha=0.5)
        ax.set_ylim(-1, n)
        ax.set_ylabel("neuron (by cluster)")
        first_late = next((t for t in onsets if t > BURN_IN_MS), None)
        first_late_s = f"{first_late/1000.0:.2f}s" if first_late is not None else "none"
        ax.set_title(
            f"Seed {lab} (noise seed={seed}) - identical wiring; "
            f"first post-burn-in burst onset: {first_late_s}",
            fontsize=10,
        )
    axes[0].legend(loc="upper right", markerscale=6, framealpha=0.9)
    axes[1].set_xlabel("time (s)")
    axes[1].set_xlim(0, duration_ms / 1000.0)
    fig.suptitle("Validation B.3 - Seed control: same wiring, different noise -> "
                 "different ignition times (green = detected burst onsets)", y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_adaptation_sweep(runs, neurons, cluster_info, duration_ms, out_path):
    """Save a stacked raster comparing adaptation levels on identical wiring.

    Args:
        runs: List of ``(label, increment_scale, spike_data, mean_rate)`` tuples,
            ordered from baseline to most hyperexcitable.
        neurons: Neuron list (for exc/inh coloring).
        cluster_info: Cluster metadata (row ordering).
        duration_ms: Full simulated duration (ms).
        out_path: PNG output path.

    Returns:
        None. Writes a PNG to ``out_path``.
    """
    n = len(neurons)
    _, rank, _ = _cluster_order(cluster_info, n)
    k = len(runs)
    fig, axes = plt.subplots(k, 1, sharex=True, figsize=(14, 3.0 * k),
                             gridspec_kw={"hspace": 0.12})
    if k == 1:
        axes = [axes]
    for ax, (label, incr, run, rate) in zip(axes, runs):
        _scatter_raster(ax, run, neurons, rank)
        burn_s = BURN_IN_MS / 1000.0
        ax.axvspan(0, burn_s, color="gray", alpha=0.12)
        ax.axvline(burn_s, color="k", ls="--", lw=0.8)
        ax.set_ylim(-1, n)
        ax.set_ylabel("neuron")
        tag = "baseline" if abs(incr - 1.0) < 1e-9 else (
            "4-AP-like (reduced K+/AHP)" if incr < 1.0 else "more suppressed")
        ax.set_title(
            f"increment_scale = {incr}  [{tag}]   "
            f"mean rate = {rate:.2f} Hz (post burn-in)",
            fontsize=10,
        )
    axes[0].legend(loc="upper right", markerscale=6, framealpha=0.9)
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(0, duration_ms / 1000.0)
    fig.suptitle(
        "Validation C - adaptation increment_scale is the 4-AP knob: "
        "SAME synaptic wiring, different excitability", y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Table output.
# --------------------------------------------------------------------------- #
TABLE_COLUMNS = [
    "run_label", "network_source", "num_clusters", "N",
    "within_prob_base", "between_prob_base", "realized_within", "realized_between",
    "hub_fraction", "hub_between_prob", "hub_weight_scale", "hub_reciprocal_factor",
    "noise_sigma", "depressing", "delta_q", "tau_q",
    "baseline_mean", "baseline_sd", "adapt_increment_scale", "adapt_tau_scale",
    "exc_weight_scale", "duration_ms", "seed",
    "mean_rate_Hz", "burst_freq_Hz", "peak_AF_steady", "pct_in_burst",
    "IB_rate_Hz", "runaway_flag", "n_spikes",
]


def make_row(label, n, conn_stats, diag, *, noise_sigma, increment_scale,
             duration_ms, seed):
    """Assemble one results-table row dict."""
    return {
        "run_label": label,
        "network_source": "clustered",
        "num_clusters": NETWORK_KWARGS["num_clusters"],
        "N": n,
        "within_prob_base": NETWORK_KWARGS["within_cluster_prob"],
        "between_prob_base": NETWORK_KWARGS["between_cluster_prob"],
        "realized_within": round(conn_stats["realized_within"], 4),
        "realized_between": round(conn_stats["realized_between"], 6),
        "hub_fraction": NETWORK_KWARGS["hub_fraction"],
        "hub_between_prob": NETWORK_KWARGS["hub_between_prob"],
        "hub_weight_scale": NETWORK_KWARGS["hub_weight_scale"],
        "hub_reciprocal_factor": NETWORK_KWARGS["hub_reciprocal_factor"],
        "noise_sigma": noise_sigma,
        "depressing": NETWORK_KWARGS["depressing"],
        "delta_q": NETWORK_KWARGS["delta_q"],
        "tau_q": NETWORK_KWARGS["tau_q"],
        "baseline_mean": BASELINE_MEAN,
        "baseline_sd": BASELINE_SD,
        "adapt_increment_scale": increment_scale,
        "adapt_tau_scale": ADAPT_TAU_SCALE,
        "exc_weight_scale": EXC_WEIGHT_SCALE,
        "duration_ms": int(duration_ms),
        "seed": seed,
        "mean_rate_Hz": round(diag["mean_rate_Hz"], 4),
        "burst_freq_Hz": round(diag["burst_freq_Hz"], 4),
        "peak_AF_steady": round(diag["peak_AF_steady"], 4),
        "pct_in_burst": round(diag["pct_in_burst"], 2),
        "IB_rate_Hz": round(diag["IB_rate_Hz"], 4),
        "runaway_flag": diag["runaway_flag"],
        "n_spikes": diag["n_spikes"],
    }


def write_table(rows, out_dir):
    """Write the results table as both CSV and Markdown."""
    csv_path = out_dir / "validation_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md_path = out_dir / "validation_table.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Spontaneous-bursting validation - parameter + results table\n\n")
        fh.write("One row per run. Build seed is fixed at "
                 f"`{BUILD_SEED}` for every run (identical wiring); the `seed` "
                 "column is the membrane-noise seed that distinguishes runs.\n\n")
        fh.write("| " + " | ".join(TABLE_COLUMNS) + " |\n")
        fh.write("| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[c]) for c in TABLE_COLUMNS) + " |\n")
    return csv_path, md_path


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    """Run all three validations, save every raster, and write the results table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def emit(msg=""):
        print(msg)
        log_lines.append(msg)

    emit("=" * 78)
    emit("SPONTANEOUS-BURSTING VALIDATION")
    emit("=" * 78)
    emit(f"Build seed (fixed for all runs): {BUILD_SEED}")
    emit(f"Config: {NETWORK_KWARGS}")
    emit(f"Transforms: baseline mean={BASELINE_MEAN} sd={BASELINE_SD} (drivers OFF), "
         f"adapt tau_scale={ADAPT_TAU_SCALE} increment_scale={ADAPT_INCREMENT_SCALE}, "
         f"exc_weight_scale={EXC_WEIGHT_SCALE}")
    emit(f"Burn-in dropped before scoring: {BURN_IN_MS:.0f} ms")
    emit("")

    rows = []

    # ----------------------------------------------------------------- #
    # VALIDATION A - the setting works (spontaneous network bursts).
    # ----------------------------------------------------------------- #
    emit("-" * 78)
    emit("VALIDATION A - spontaneous network bursts")
    emit("-" * 78)
    neurons, synapses, connections, positions, cluster_info = build_validated_network()
    n = len(neurons)
    n_exc = sum(1 for x in neurons if not x.is_inhibitory)
    n_inh = n - n_exc
    conn_stats = compute_connectivity(connections, cluster_info, n)
    emit(f"Built network: N={n} ({n_exc} exc / {n_inh} inh), "
         f"{len(connections)} connections")
    emit(f"Realized connectivity: within={conn_stats['realized_within']:.4f}  "
         f"between_base={conn_stats['realized_between']:.6f}  "
         f"(between_all={conn_stats['realized_between_all']:.5f})")
    emit(f"  hub share of inter-cluster edges = {conn_stats['hub_share_inter']*100:.1f}%  "
         f"({conn_stats['between_hub_edges']}/{conn_stats['between_edges']}); "
         f"overall density = {conn_stats['overall_density']*100:.2f}%")

    emit(f"Running validated config for {DUR_VALIDATED:.0f} ms "
         f"(noise seed={NOISE_SEED_A}) ...")
    run_validated = run_spontaneous(neurons, synapses, DUR_VALIDATED, NOISE_SEED_A)
    diag_validated = compute_diagnostics(run_validated, n, DUR_VALIDATED)
    emit(f"  mean_rate = {diag_validated['mean_rate_Hz']:.3f} Hz")
    emit(f"  network-burst freq = {diag_validated['burst_freq_Hz']:.3f} Hz "
         f"({diag_validated['n_bursts']} bursts)")
    emit(f"  steady-state peak AF = {diag_validated['peak_AF_steady']:.3f}")
    emit(f"  % spikes in bursts = {diag_validated['pct_in_burst']:.1f}%")
    emit(f"  inter-burst per-neuron rate = {diag_validated['IB_rate_Hz']:.3f} Hz")
    emit(f"  per-2s window rates (Hz) = "
         f"{[round(r,2) for r in diag_validated['win_rates']]}")
    emit(f"  runaway flag = {diag_validated['runaway_flag']}")

    plot_raster_with_af(
        run_validated, neurons, cluster_info, DUR_VALIDATED,
        f"Validation A - spontaneous bursts (N={n}, drivers OFF, no stimulation, "
        f"{DUR_VALIDATED/1000:.0f} s)",
        OUTPUT_DIR / "validation_A_validated_raster.png",
    )
    emit(f"  saved: validation_A_validated_raster.png")

    passA_bursts = diag_validated["peak_AF_steady"] > 0.4
    passA_baseline = diag_validated["IB_rate_Hz"] > 0.0
    passA_stable = not diag_validated["runaway_flag"]
    passA = passA_bursts and passA_baseline and passA_stable
    emit("")
    emit(f"  [A] discrete bursts (peak AF > 0.4): {passA_bursts} "
         f"(peak AF = {diag_validated['peak_AF_steady']:.3f})")
    emit(f"  [A] sparse non-silent inter-burst baseline (IB rate > 0): "
         f"{passA_baseline} (IB rate = {diag_validated['IB_rate_Hz']:.3f} Hz)")
    emit(f"  [A] stable / not runaway: {passA_stable}")
    emit(f"  VALIDATION A: {'PASS' if passA else 'FAIL'}")
    emit("")

    rows.append(make_row(
        "validated_config (A)", n, conn_stats, diag_validated,
        noise_sigma=NETWORK_KWARGS["background_noise_sigma"],
        increment_scale=ADAPT_INCREMENT_SCALE, duration_ms=DUR_VALIDATED,
        seed=NOISE_SEED_A,
    ))

    # ----------------------------------------------------------------- #
    # VALIDATION B - no stimulation is used (prove spontaneity).
    # ----------------------------------------------------------------- #
    emit("-" * 78)
    emit("VALIDATION B - no stimulation is used")
    emit("-" * 78)

    # B.1 assertions on the validated run.
    stimulation_events = []  # this is exactly what was passed to simulate_network.
    assert stimulation_events == [], "stimulation_events must be empty"
    baselines = [x.i_baseline for x in neurons]
    assert all(b == 0.0 for b in baselines), "every i_baseline must be 0.0"
    i_ext_now = [x.i_ext for x in neurons]
    assert all(v == 0.0 for v in i_ext_now), "every i_ext must be 0.0"
    emit("B.1 assertions on the validated run:")
    emit(f"     stimulation_events == []            -> {stimulation_events == []} "
         f"(len={len(stimulation_events)})")
    emit(f"     all i_baseline == 0.0               -> True "
         f"(max |i_baseline| = {max(abs(b) for b in baselines):.1e})")
    emit(f"     all i_ext == 0.0 (post-run)         -> True "
         f"(max |i_ext| = {max(abs(v) for v in i_ext_now):.1e})")
    emit("     i_ext is provably 0 at ALL timesteps: with stimulation_events==[], "
         "simulate_network sets neuron.i_ext = active_stims.get(id,(0.0,0.0))[0] = 0.0 "
         "every step (active_stims never populated).")
    emit("")

    # B.2 noise-off control: same wiring, noise OFF, drivers off, no stim -> silent.
    emit("B.2 noise-off control (background_noise_sigma=0.0, drivers off, no stim):")
    neurons_off, syn_off, conn_off, _, ci_off = build_validated_network(noise_sigma=0.0)
    assert connection_hash(conn_off) == connection_hash(connections), \
        "noise-off wiring must match the validated wiring"
    run_off = run_spontaneous(neurons_off, syn_off, DUR_NOISE_OFF, NOISE_SEED_A)
    total_off = sum(len(s) for s in run_off.values())
    assert total_off == 0, f"noise-off network must be silent, got {total_off} spikes"
    emit(f"     total spikes over {DUR_NOISE_OFF:.0f} ms = {total_off}  -> SILENT")
    emit("     -> no hidden drive: with noise off and no stimulus the network cannot "
         "fire, so the bursts in A are noise-ignited, not stimulus-driven.")
    diag_off = compute_diagnostics(run_off, len(neurons_off), DUR_NOISE_OFF)
    conn_stats_off = compute_connectivity(conn_off, ci_off, len(neurons_off))
    plot_raster_with_af(
        run_off, neurons_off, ci_off, DUR_NOISE_OFF,
        "Validation B.2 - noise-off control (drivers off, no stim): silent network "
        "(0 spikes)",
        OUTPUT_DIR / "validation_B_noise_off_raster.png",
    )
    emit(f"     saved: validation_B_noise_off_raster.png (empty raster)")
    emit("")

    # B.3 seed control: same wiring, two noise seeds -> different ignition.
    emit("B.3 seed control (identical wiring, two noise seeds):")
    neurons_sa, syn_sa, conn_sa, _, ci_sa = build_validated_network()
    run_sa = run_spontaneous(neurons_sa, syn_sa, DUR_SEED_CTRL, NOISE_SEED_A)
    neurons_sb, syn_sb, conn_sb, _, ci_sb = build_validated_network()
    assert connection_hash(conn_sb) == connection_hash(conn_sa), \
        "seed-control wiring must be identical"
    run_sb = run_spontaneous(neurons_sb, syn_sb, DUR_SEED_CTRL, NOISE_SEED_B)
    onsets_a = burst_onsets(run_sa, len(neurons_sa), DUR_SEED_CTRL)
    onsets_b = burst_onsets(run_sb, len(neurons_sb), DUR_SEED_CTRL)
    emit(f"     seed A (noise={NOISE_SEED_A}) burst onsets (ms) = {onsets_a}")
    emit(f"     seed B (noise={NOISE_SEED_B}) burst onsets (ms) = {onsets_b}")
    late_a = [t for t in onsets_a if t > BURN_IN_MS]
    late_b = [t for t in onsets_b if t > BURN_IN_MS]
    onsets_differ = late_a != late_b
    emit(f"     post-burn-in onsets differ across seeds -> {onsets_differ} "
         "(a scheduled/periodic stimulus could not do this)")
    emit("     (the t~0 startup event is identical across seeds - set by initial "
         "conditions, not noise)")
    diag_sb = compute_diagnostics(run_sb, len(neurons_sb), DUR_SEED_CTRL)
    conn_stats_sb = compute_connectivity(conn_sb, ci_sb, len(neurons_sb))
    plot_seed_control(run_sa, run_sb, neurons_sa, ci_sa, DUR_SEED_CTRL,
                      onsets_a, onsets_b,
                      OUTPUT_DIR / "validation_B_seed_control_raster.png")
    emit(f"     saved: validation_B_seed_control_raster.png")
    emit("")

    spontaneity_confirmed = (
        stimulation_events == []
        and all(b == 0.0 for b in baselines)
        and total_off == 0
        and onsets_differ
    )
    emit(f"SPONTANEITY: {'CONFIRMED' if spontaneity_confirmed else 'NOT CONFIRMED'}")
    emit("  evidence: (1) stimulation_events==[] and all i_baseline/i_ext==0; "
         "(2) noise-off copy of identical wiring is completely silent; "
         "(3) two noise seeds ignite bursts at different times.")
    emit("")

    rows.append(make_row(
        "noise_off_control (B.2)", len(neurons_off), conn_stats_off, diag_off,
        noise_sigma=0.0, increment_scale=ADAPT_INCREMENT_SCALE,
        duration_ms=DUR_NOISE_OFF, seed=NOISE_SEED_A,
    ))
    rows.append(make_row(
        "seed_B (B.3)", len(neurons_sb), conn_stats_sb, diag_sb,
        noise_sigma=NETWORK_KWARGS["background_noise_sigma"],
        increment_scale=ADAPT_INCREMENT_SCALE, duration_ms=DUR_SEED_CTRL,
        seed=NOISE_SEED_B,
    ))

    # ----------------------------------------------------------------- #
    # VALIDATION C - changing adaptation reproduces a 4-AP-like state.
    # ----------------------------------------------------------------- #
    emit("-" * 78)
    emit("VALIDATION C - adaptation is the 4-AP knob (same wiring, more excitability)")
    emit("-" * 78)
    emit("4-AP blocks K+ channels -> reduces spike-triggered K+/AHP (adaptation) -> "
         "more excitability. In this point-neuron model that maps onto LOWERING "
         "scale_adaptation_dynamics increment_scale (a K+-block proxy, not a literal "
         "I_A block).")

    adapt_levels = [1.0, 0.5, 0.25, 2.0]  # 2.0 = opposite (more suppressed) direction
    sweep_runs = []
    ref_hash = None
    sweep_rate = {}
    for incr in adapt_levels:
        neurons_c, syn_c, conn_c, _, ci_c = build_validated_network(increment_scale=incr)
        h = connection_hash(conn_c)
        if ref_hash is None:
            ref_hash = h
        same_wiring = (h == ref_hash)
        assert same_wiring, f"wiring changed at increment_scale={incr}"
        run_c = run_spontaneous(neurons_c, syn_c, DUR_ADAPT, NOISE_SEED_A)
        diag_c = compute_diagnostics(run_c, len(neurons_c), DUR_ADAPT)
        conn_stats_c = compute_connectivity(conn_c, ci_c, len(neurons_c))
        sweep_rate[incr] = diag_c["mean_rate_Hz"]
        emit(f"  increment_scale={incr:<5} wiring_hash_match={same_wiring}  "
             f"mean_rate={diag_c['mean_rate_Hz']:.3f} Hz  "
             f"peak_AF={diag_c['peak_AF_steady']:.3f}  "
             f"burst_freq={diag_c['burst_freq_Hz']:.3f} Hz  "
             f"%in_burst={diag_c['pct_in_burst']:.1f}")
        if incr in (0.5, 0.25, 2.0):
            label = f"4AP_incr_{incr} (C)" if incr < 1.0 else f"suppressed_incr_{incr} (C)"
            rows.append(make_row(
                label, len(neurons_c), conn_stats_c, diag_c,
                noise_sigma=NETWORK_KWARGS["background_noise_sigma"],
                increment_scale=incr, duration_ms=DUR_ADAPT, seed=NOISE_SEED_A,
            ))
        tag_run = (f"incr={incr}", incr, run_c, diag_c["mean_rate_Hz"])
        # keep baseline + the two 4-AP directions for the comparison raster
        if incr in (1.0, 0.5, 0.25):
            sweep_runs.append(tag_run)

    # Order the comparison raster from baseline -> most hyperexcitable.
    sweep_runs_sorted = sorted(sweep_runs, key=lambda t: -t[1])  # 1.0, 0.5, 0.25
    plot_adaptation_sweep(
        [(lbl, incr, run, rate) for (lbl, incr, run, rate) in sweep_runs_sorted],
        neurons, cluster_info, DUR_ADAPT,
        OUTPUT_DIR / "validation_C_adaptation_sweep_raster.png",
    )
    emit(f"  saved: validation_C_adaptation_sweep_raster.png")

    base_rate = sweep_rate[1.0]
    emit("")
    emit("  rate shift (same wiring, lowering increment_scale = the 4-AP direction):")
    for incr in [1.0, 0.5, 0.25]:
        delta = sweep_rate[incr] - base_rate
        emit(f"     increment_scale={incr:<5} -> mean_rate={sweep_rate[incr]:.3f} Hz "
             f"({'+' if delta >= 0 else ''}{delta:.3f} Hz vs baseline)")
    emit(f"     increment_scale=2.0   -> mean_rate={sweep_rate[2.0]:.3f} Hz "
         f"(opposite / more-suppressed direction)")
    passC = sweep_rate[0.25] > sweep_rate[0.5] > sweep_rate[1.0]
    emit(f"  monotone increase as adaptation is reduced (0.25 > 0.5 > 1.0): {passC}")
    emit(f"  VALIDATION C: {'PASS' if passC else 'FAIL'}  "
         "-- adaptation increment_scale is the 4-AP knob: same synaptic wiring, "
         "different excitability.")
    emit("")

    # ----------------------------------------------------------------- #
    # Table + summary.
    # ----------------------------------------------------------------- #
    csv_path, md_path = write_table(rows, OUTPUT_DIR)
    emit("=" * 78)
    emit("OUTPUTS")
    emit("=" * 78)
    emit(f"  table: {csv_path.name}, {md_path.name}")
    for name in sorted(p.name for p in OUTPUT_DIR.glob("*.png")):
        emit(f"  raster: {name}")

    emit("")
    emit("OVERALL: "
         f"A={'PASS' if passA else 'FAIL'}  "
         f"B={'CONFIRMED' if spontaneity_confirmed else 'NOT CONFIRMED'}  "
         f"C={'PASS' if passC else 'FAIL'}")

    with open(OUTPUT_DIR / "validation_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"\nSaved summary log to {OUTPUT_DIR / 'validation_summary.txt'}")


if __name__ == "__main__":
    main()
