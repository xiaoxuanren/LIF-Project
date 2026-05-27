import random

import numpy as np

from .models import LIFNeuron, NetworkWeightParameters
from .network import create_clustered_network
from .simulation import simulate_network


def run_h_current_step_probe(use_h_current=True, step_current_nA=-0.12, duration_ms=1200.0, dt=0.1):
    """Run a single-neuron hyperpolarizing current-step probe.

    Args:
        use_h_current: Whether the probe neuron should include the h-current.
        step_current_nA: Amplitude of the injected hyperpolarizing current in nA.
        duration_ms: Total probe duration in milliseconds.
        dt: Simulation time step in milliseconds.

    Returns:
        A dictionary containing the probe time axis, voltage trace, h-current
        state variables, and step timing metadata.
    """
    neuron = LIFNeuron(0, is_inhibitory=False, use_h_current=True)
    neuron.noise_sigma = 0.0
    neuron.g_exc = 0.0
    neuron.g_inh = 0.0
    neuron.set_h_current_enabled(use_h_current)

    step_start_ms = 200.0
    step_end_ms = 900.0
    times = np.arange(0.0, duration_ms, dt)
    voltage = np.zeros_like(times)
    h_gate = np.zeros_like(times)
    i_h = np.zeros_like(times)
    i_ext = np.zeros_like(times)

    for idx, time_ms in enumerate(times):
        neuron.i_ext = step_current_nA if step_start_ms <= time_ms < step_end_ms else 0.0
        neuron.update(time_ms, dt)
        voltage[idx] = neuron.v
        h_gate[idx] = neuron.h_gate
        i_h[idx] = neuron.i_h
        i_ext[idx] = neuron.i_ext

    return {
        "times": times,
        "voltage": voltage,
        "h_gate": h_gate,
        "i_h": i_h,
        "i_ext": i_ext,
        "step_start_ms": step_start_ms,
        "step_end_ms": step_end_ms,
        "v_rest": neuron.v_rest,
    }


def summarize_h_current_step_probe(sag_with_h, sag_without_h):
    """Measure sag and rebound metrics from paired h-current probe outputs.

    Args:
        sag_with_h: Probe output dictionary generated with the h-current enabled.
        sag_without_h: Probe output dictionary generated with the h-current disabled.

    Returns:
        A dictionary summarizing step amplitude, sag depth, and rebound amplitude
        for the two probe conditions.
    """
    during_step_mask = (
        (sag_with_h["times"] >= sag_with_h["step_start_ms"])
        & (sag_with_h["times"] < sag_with_h["step_end_ms"])
    )
    late_step_mask = (
        (sag_with_h["times"] >= sag_with_h["step_end_ms"] - 100.0)
        & (sag_with_h["times"] < sag_with_h["step_end_ms"])
    )
    rebound_mask = (
        (sag_with_h["times"] >= sag_with_h["step_end_ms"])
        & (sag_with_h["times"] < sag_with_h["step_end_ms"] + 150.0)
    )

    sag_depth_with_h = float(
        np.mean(sag_with_h["voltage"][late_step_mask]) - np.min(sag_with_h["voltage"][during_step_mask])
    )
    sag_depth_without_h = float(
        np.mean(sag_without_h["voltage"][late_step_mask]) - np.min(sag_without_h["voltage"][during_step_mask])
    )
    rebound_with_h = float(np.max(sag_with_h["voltage"][rebound_mask]) - sag_with_h["v_rest"])
    rebound_without_h = float(np.max(sag_without_h["voltage"][rebound_mask]) - sag_without_h["v_rest"])

    return {
        "step_current_nA": float(np.min(sag_with_h["i_ext"])),
        "sag_depth_with_h_mV": sag_depth_with_h,
        "sag_depth_without_h_mV": sag_depth_without_h,
        "rebound_with_h_mV": rebound_with_h,
        "rebound_without_h_mV": rebound_without_h,
    }


def run_no_stimulation_validation(
    seed=42,
    use_h_current=True,
    test_duration_ms=10000,
    requested_voltage_sample_rate=0.25,
    num_clusters=20,
    neurons_per_cluster_range=(12, 18),
    inhibitory_probability=0.2,
    within_cluster_prob=0.3,
    between_cluster_prob=0.15,
    max_connection_distance=6.0,
    space_size=15,
    hub_fraction=0.1,
    hub_between_prob=0.3,
    hub_weight_scale=1.5,
    hub_reciprocal_factor=2.0,
    dt=0.1,
    voltage_window_ms=2000,
    raster_window_ms=60000,
    max_connections_plot=600,
):
    """Simulate spontaneous activity and score whether the network stays near critical.

    Args:
        seed: Random seed applied to NumPy and Python's `random` module.
        use_h_current: Whether the generated network should include the h-current.
        test_duration_ms: Duration of the validation recording in milliseconds.
        requested_voltage_sample_rate: Legacy requested voltage sampling interval.
        num_clusters: Number of clusters used when generating the network.
        neurons_per_cluster_range: Inclusive range used to sample cluster sizes.
        inhibitory_probability: Probability that a neuron is inhibitory.
        within_cluster_prob: Same-cluster connection probability.
        between_cluster_prob: Between-cluster connection probability.
        max_connection_distance: Maximum spatial connection distance.
        space_size: Side length of the square spatial embedding used for the network.
        hub_fraction: Fraction of neurons per cluster designated as hubs.
        hub_between_prob: Base inter-cluster connection probability for hub projections.
        hub_weight_scale: Multiplicative weight boost applied to hub-originating edges.
        hub_reciprocal_factor: Probability boost used for hub-to-hub projections.
        dt: Simulation time step in milliseconds.
        voltage_window_ms: Window length in milliseconds used for quick-look voltage plots.
        raster_window_ms: Window length in milliseconds used for quick-look raster plots.
        max_connections_plot: Maximum number of sampled connections shown in layout plots.

    Returns:
        A dictionary containing the generated network, recording data, and validation metrics.
    """
    np.random.seed(seed)
    random.seed(seed)

    raster_window_ms = min(raster_window_ms, test_duration_ms)
    voltage_window_ms = min(voltage_window_ms, test_duration_ms)
    stimulation_events = []
    weight_params = NetworkWeightParameters()

    neurons, synapses, connections, neuron_positions, cluster_info = create_clustered_network(
        num_clusters=num_clusters,
        neurons_per_cluster_range=neurons_per_cluster_range,
        inhibitory_probability=inhibitory_probability,
        within_cluster_prob=within_cluster_prob,
        between_cluster_prob=between_cluster_prob,
        max_connection_distance=max_connection_distance,
        weight_params=weight_params,
        space_size=space_size,
        hub_fraction=hub_fraction,
        hub_between_prob=hub_between_prob,
        hub_weight_scale=hub_weight_scale,
        hub_reciprocal_factor=hub_reciprocal_factor,
        use_h_current=use_h_current,
    )

    spike_data, voltage_data = simulate_network(
        neurons,
        synapses,
        stimulation_events=stimulation_events,
        dt=dt,
        duration=test_duration_ms,
        record_voltage=True,
        voltage_sample_rate=requested_voltage_sample_rate,
        save_raw_voltage=True,
    )

    saved_voltage_step_ms = float(voltage_data["sample_rate"])
    voltage_trace_mode = "raw_full_dt" if bool(voltage_data.get("save_raw_voltage", False)) else "legacy_sampled"
    duration_s = test_duration_ms / 1000.0
    n_neurons = len(neurons)
    cluster_assignments = cluster_info["cluster_assignments"]
    firing_rates = np.array([len(spike_data[nid]) / duration_s for nid in range(n_neurons)])
    total_spikes = int(sum(len(spikes) for spikes in spike_data.values()))
    active_neurons = int(sum(len(spikes) > 0 for spikes in spike_data.values()))

    bin_width_ms = 100.0
    n_bins = int(np.ceil(test_duration_ms / bin_width_ms))
    binned_activity = np.zeros((n_neurons, n_bins), dtype=int)
    for neuron_id, spikes in spike_data.items():
        if len(spikes) == 0:
            continue
        bin_idx = np.floor(np.asarray(spikes) / bin_width_ms).astype(int)
        bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]
        if len(bin_idx) > 0:
            binned_activity[neuron_id, np.unique(bin_idx)] = 1

    active_fraction = binned_activity.mean(axis=0)
    max_active_fraction = float(active_fraction.max())
    bins_ge_10 = int(np.sum(active_fraction >= 0.10))
    bins_ge_25 = int(np.sum(active_fraction >= 0.25))
    bins_ge_50 = int(np.sum(active_fraction >= 0.50))

    if bins_ge_25 > 0 or max_active_fraction >= 0.25:
        verdict = "FAIL_auto_bursting"
    elif firing_rates.mean() < 0.05:
        verdict = "FAIL_too_quiet"
    elif 0.1 <= firing_rates.mean() <= 0.5 and max_active_fraction < 0.10:
        verdict = "PASS_near_critical"
    else:
        verdict = "BORDERLINE"

    active_ids = [nid for nid, spikes in spike_data.items() if len(spikes) > 0]
    if active_ids:
        ranked_active_ids = sorted(active_ids, key=lambda nid: len(spike_data[nid]), reverse=True)
        example_neuron_ids = ranked_active_ids[: min(6, len(ranked_active_ids))]
    else:
        example_neuron_ids = list(range(min(6, n_neurons)))

    metrics = {
        "use_h_current": bool(use_h_current),
        "external_stimulation_events": len(stimulation_events),
        "test_duration_s": duration_s,
        "requested_voltage_sample_rate_ms": float(requested_voltage_sample_rate),
        "saved_voltage_step_ms": saved_voltage_step_ms,
        "voltage_trace_mode": voltage_trace_mode,
        "raster_window_s": raster_window_ms / 1000.0,
        "n_neurons": n_neurons,
        "total_spikes": total_spikes,
        "active_neurons": active_neurons,
        "mean_rate_hz": float(firing_rates.mean()),
        "median_rate_hz": float(np.median(firing_rates)),
        "max_rate_hz": float(firing_rates.max()),
        "max_active_fraction_100ms": max_active_fraction,
        "bins_ge_10pct": bins_ge_10,
        "bins_ge_25pct": bins_ge_25,
        "bins_ge_50pct": bins_ge_50,
        "verdict": verdict,
    }

    return {
        "seed": seed,
        "weight_params": weight_params,
        "neurons": neurons,
        "synapses": synapses,
        "connections": connections,
        "neuron_positions": neuron_positions,
        "cluster_info": cluster_info,
        "spike_data": spike_data,
        "voltage_data": voltage_data,
        "cluster_assignments": cluster_assignments,
        "example_neuron_ids": example_neuron_ids,
        "raster_window_ms": raster_window_ms,
        "voltage_window_ms": voltage_window_ms,
        "max_connections_plot": max_connections_plot,
        "metrics": metrics,
    }