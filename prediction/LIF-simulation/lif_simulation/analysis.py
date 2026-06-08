import numpy as np


def organize_spike_data_by_cluster(spike_data, cluster_info):
    """Group per-neuron spike trains into cluster-ordered lists.

    Args:
        spike_data: Mapping from neuron id to a sequence of spike times.
        cluster_info: Cluster metadata containing ``cluster_neuron_groups``.

    Returns:
        A list whose entries correspond to clusters and contain the spike-train
        lists for neurons in that cluster.
    """
    num_clusters = len(cluster_info["cluster_neuron_groups"])
    cluster_spike_data = [[] for _ in range(num_clusters)]

    for cluster_id in range(num_clusters):
        neuron_ids = cluster_info["cluster_neuron_groups"][cluster_id]
        for neuron_id in neuron_ids:
            cluster_spike_data[cluster_id].append(spike_data[neuron_id])

    return cluster_spike_data


def resample_data(spike_data, cluster_assignments, target_freq=10, duration=60000):
    """Bin spike times onto a fixed grid for saved raster-style outputs.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        cluster_assignments: Unused legacy cluster-assignment array kept for API
            compatibility with existing callers.
        target_freq: Desired resampling frequency in hertz.
        duration: Recording duration in milliseconds.

    Returns:
        A tuple ``(resampled_spikes, resampled_time_points, resampled_positions)``
        containing the binary spike matrix, the bin-center time axis, and the
        nonzero spike coordinates used by plotting helpers.
    """
    if target_freq <= 0:
        raise ValueError("target_freq must be positive.")

    interval_ms = 1000.0 / target_freq
    n_points = int(np.ceil(duration / interval_ms))
    n_neurons = len(spike_data)

    resampled_spikes = np.zeros((n_neurons, n_points), dtype=int)
    resampled_time_points = np.arange(n_points) * interval_ms

    for neuron_id, spikes in spike_data.items():
        if len(spikes) == 0:
            continue
        spike_times = np.asarray(spikes, dtype=float)
        bin_indices = np.floor(spike_times / interval_ms).astype(int)
        valid_bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < n_points)]
        if valid_bin_indices.size > 0:
            resampled_spikes[neuron_id, np.unique(valid_bin_indices)] = 1

    resampled_spike_positions = np.argwhere(resampled_spikes)
    return resampled_spikes, resampled_time_points, resampled_spike_positions


def report_network_statistics(spike_data, neurons, connections, duration):
    """Print a quick descriptive summary of one saved recording.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        neurons: Sequence of neuron objects used in the simulation.
        connections: Connection table describing synapses in the generated network.
        duration: Recording duration in milliseconds.

    Returns:
        None. The function prints summary statistics for neurons, connections, and
        firing rates.
    """
    print("\n" + "=" * 50)
    print("NETWORK STATISTICS")
    print("=" * 50)

    n_exc = sum(1 for neuron in neurons if not neuron.is_inhibitory)
    n_inh = sum(1 for neuron in neurons if neuron.is_inhibitory)
    print(f"\nNeurons: {len(neurons)} total ({n_exc} exc, {n_inh} inh)")

    if len(connections) > 0:
        n_exc_conn = np.sum(connections[:, 3] == "exc")
        n_inh_conn = np.sum(connections[:, 3] == "inh")
        print(f"Connections: {len(connections)} total ({n_exc_conn} exc, {n_inh_conn} inh)")

        exc_weights = connections[connections[:, 3] == "exc", 2].astype(float)
        inh_weights = connections[connections[:, 3] == "inh", 2].astype(float)
        if len(exc_weights) > 0:
            print(f"\nExc weights: mean={np.mean(exc_weights):.3f}, std={np.std(exc_weights):.3f}")
        if len(inh_weights) > 0:
            print(f"Inh weights: mean={np.mean(inh_weights):.3f}, std={np.std(inh_weights):.3f}")

    firing_rates = np.array([
        len(spikes) / (duration / 1000.0)
        for spikes in spike_data.values()
    ])
    print("\nFiring rates (Hz):")
    print(f"  Mean: {np.mean(firing_rates):.2f}")
    print(f"  Std: {np.std(firing_rates):.2f}")
    print(f"  Min: {np.min(firing_rates):.2f}")
    print(f"  Max: {np.max(firing_rates):.2f}")

    total_spikes = sum(len(spikes) for spikes in spike_data.values())
    print(f"\nTotal spikes: {total_spikes}")
    print(f"Active neurons: {sum(1 for spikes in spike_data.values() if len(spikes) > 0)}")
    print("=" * 50 + "\n")


def validate_hub_structure(connections, cluster_info, n_neurons):
    """Compare hub and non-hub degree structure to sanity-check hub generation.

    Args:
        connections: Connection table produced during network construction.
        cluster_info: Cluster metadata containing hub ids and assignments.
        n_neurons: Total number of neurons in the network.

    Returns:
        A dictionary of degree-based hub diagnostics, or ``None`` if the provided
        cluster metadata does not include hub neurons.
    """
    hub_neuron_ids = cluster_info.get("hub_neuron_ids", [])
    hub_set = set(hub_neuron_ids)
    if not hub_neuron_ids:
        print("No hub neurons found in cluster_info.")
        return None

    out_degree = np.zeros(n_neurons, dtype=int)
    in_degree = np.zeros(n_neurons, dtype=int)
    for conn in connections:
        pre_id = int(conn[0])
        post_id = int(conn[1])
        out_degree[pre_id] += 1
        in_degree[post_id] += 1

    total_degree = out_degree + in_degree
    hub_degrees = total_degree[hub_neuron_ids]
    non_hub_ids = [idx for idx in range(n_neurons) if idx not in hub_set]
    non_hub_degrees = total_degree[non_hub_ids]
    hub_out = out_degree[hub_neuron_ids]
    non_hub_out = out_degree[non_hub_ids]
    hub_in = in_degree[hub_neuron_ids]
    non_hub_in = in_degree[non_hub_ids]

    print("=" * 50)
    print("HUB NEURON VALIDATION")
    print("=" * 50)
    print(f"Total neurons: {n_neurons}")
    print(f"Hub neurons: {len(hub_neuron_ids)} ({100 * len(hub_neuron_ids) / n_neurons:.1f}%)")
    print(f"Hub fraction setting: {cluster_info.get('hub_fraction', 'N/A')}")
    print()

    print(f"Hub mean total degree:     {np.mean(hub_degrees):.1f} +/- {np.std(hub_degrees):.1f}")
    print(f"Non-hub mean total degree: {np.mean(non_hub_degrees):.1f} +/- {np.std(non_hub_degrees):.1f}")
    ratio = np.mean(hub_degrees) / np.mean(non_hub_degrees) if np.mean(non_hub_degrees) > 0 else float("inf")
    print(f"Degree ratio (hub/non-hub): {ratio:.2f}x")
    print()
    print(f"Hub mean out-degree: {np.mean(hub_out):.1f} | Non-hub: {np.mean(non_hub_out):.1f}")
    print(f"Hub mean in-degree:  {np.mean(hub_in):.1f} | Non-hub: {np.mean(non_hub_in):.1f}")
    print()

    cluster_assignments = cluster_info["cluster_assignments"]
    hub_between = 0
    non_hub_between = 0
    hub_total = 0
    non_hub_total = 0
    for conn in connections:
        pre_id = int(conn[0])
        post_id = int(conn[1])
        is_between = cluster_assignments[pre_id] != cluster_assignments[post_id]
        if pre_id in hub_set:
            hub_total += 1
            if is_between:
                hub_between += 1
        else:
            non_hub_total += 1
            if is_between:
                non_hub_between += 1

    hub_between_frac = hub_between / hub_total if hub_total > 0 else 0.0
    non_hub_between_frac = non_hub_between / non_hub_total if non_hub_total > 0 else 0.0
    print(f"Hub inter-cluster fraction:     {hub_between_frac:.1%} ({hub_between}/{hub_total})")
    print(
        f"Non-hub inter-cluster fraction: {non_hub_between_frac:.1%} "
        f"({non_hub_between}/{non_hub_total})"
    )
    print()

    checks = [
        ("Degree ratio >= 2x", ratio >= 2.0),
        ("Hub between-cluster fraction > non-hub", hub_between_frac > non_hub_between_frac),
        ("Hub count > 0", len(hub_neuron_ids) > 0),
    ]
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    print("=" * 50)
    return {
        "hub_degrees": hub_degrees,
        "non_hub_degrees": non_hub_degrees,
        "hub_out_degree": hub_out,
        "non_hub_out_degree": non_hub_out,
        "degree_ratio": ratio,
        "hub_between_fraction": hub_between_frac,
        "non_hub_between_fraction": non_hub_between_frac,
    }


def analyze_spike_trains(
    spike_data_dict,
    cluster_assignments,
    duration_ms,
    connections=None,
    burst_onset_times=None,
    expected_mode=None,
):
    """Compute descriptive spike-train metrics and print a recording quality verdict.

    Args:
        spike_data_dict: Mapping from neuron id to spike times in milliseconds.
        cluster_assignments: Cluster index for each neuron.
        duration_ms: Recording duration in milliseconds.
        connections: Optional connection table for connectivity statistics.
        burst_onset_times: Optional stimulation onset times for burst-aligned checks.
        expected_mode: Expected operating mode, typically spontaneous or stimulus-driven.

    Returns:
        A dictionary of aggregate spike-train metrics used by plotting and reporting.
    """
    n_neurons = len(spike_data_dict)
    duration_s = duration_ms / 1000.0
    stimulus_enabled = burst_onset_times is not None and len(burst_onset_times) > 0

    if expected_mode is None:
        expected_mode = "stimulus_driven" if stimulus_enabled else "spontaneous"

    print("=" * 70)
    print("SPIKE TRAIN ANALYSIS")
    print("=" * 70)
    print(f"Expected mode: {expected_mode}")
    print(
        f"Stimulus onsets available: "
        f"{len(burst_onset_times) if burst_onset_times is not None else 0}"
    )

    print("\n" + "-" * 50)
    print("1. FIRING RATE ANALYSIS")
    print("-" * 50)
    firing_rates = []
    spike_counts = []
    for spikes in spike_data_dict.values():
        n_spikes = len(spikes)
        spike_counts.append(n_spikes)
        firing_rates.append(n_spikes / duration_s)

    firing_rates = np.array(firing_rates)
    spike_counts = np.array(spike_counts)
    active_fraction = float(np.mean(firing_rates > 0)) if n_neurons > 0 else 0.0

    print(f"Mean firing rate: {np.mean(firing_rates):.3f} +/- {np.std(firing_rates):.3f} Hz")
    print(f"Median firing rate: {np.median(firing_rates):.3f} Hz")
    print(f"Range: {np.min(firing_rates):.3f} - {np.max(firing_rates):.3f} Hz")
    print(f"Active neurons: {np.sum(firing_rates > 0)} / {n_neurons} ({100 * active_fraction:.1f}%)")
    print(f"Silent neurons: {np.sum(firing_rates == 0)}")

    print("\n" + "-" * 50)
    print("2. INTER-SPIKE INTERVAL (ISI) ANALYSIS")
    print("-" * 50)
    all_isis = []
    cv_isis = []
    for spikes in spike_data_dict.values():
        if len(spikes) > 1:
            isis = np.diff(spikes)
            all_isis.extend(isis)
            if len(isis) > 1 and np.mean(isis) > 0:
                cv_isis.append(float(np.std(isis) / np.mean(isis)))

    all_isis = np.array(all_isis)
    refractory_violations = int(np.sum(all_isis < 2.0)) if len(all_isis) > 0 else 0
    if len(all_isis) > 0:
        print(f"Mean ISI: {np.mean(all_isis):.2f} ms")
        print(f"Median ISI: {np.median(all_isis):.2f} ms")
        print(f"Min ISI: {np.min(all_isis):.2f} ms")
        print(f"ISI < 2 ms: {refractory_violations} ({100 * refractory_violations / len(all_isis):.3f}%)")
        if cv_isis:
            print(f"Mean CV of ISI: {np.mean(cv_isis):.2f} (about 1.0 is Poisson-like)")
    else:
        print("Not enough spikes for ISI analysis")

    print("\n" + "-" * 50)
    print("3. TEMPORAL DYNAMICS")
    print("-" * 50)
    bin_size = 100
    n_bins = max(1, int(np.ceil(duration_ms / bin_size)))
    population_spike_counts = np.zeros(n_bins)
    active_neurons_per_bin = np.zeros(n_bins)
    for spikes in spike_data_dict.values():
        if len(spikes) == 0:
            continue

        spike_bins = np.floor(np.asarray(spikes, dtype=float) / bin_size).astype(int)
        valid_spike_bins = spike_bins[(spike_bins >= 0) & (spike_bins < n_bins)]
        if valid_spike_bins.size == 0:
            continue

        np.add.at(population_spike_counts, valid_spike_bins, 1)
        active_neurons_per_bin[np.unique(valid_spike_bins)] += 1

    population_rate = population_spike_counts / (bin_size / 1000.0) / n_neurons
    active_fraction_per_bin = active_neurons_per_bin / n_neurons
    mean_population_rate = float(np.mean(population_rate))
    early_window = max(1, int(n_bins * 0.2))
    late_window = max(1, int(n_bins * 0.2))
    early_rate = float(np.mean(population_rate[:early_window]))
    late_rate = float(np.mean(population_rate[-late_window:]))
    rate_delta = late_rate - early_rate

    sync_threshold = max(
        float(np.mean(active_fraction_per_bin) + 4 * np.std(active_fraction_per_bin)),
        0.15,
    )
    synchronized_bins = np.where(active_fraction_per_bin >= sync_threshold)[0]

    print(
        f"Population rate (100 ms bins): mean={mean_population_rate:.3f} Hz, "
        f"std={np.std(population_rate):.3f} Hz"
    )
    print(f"Max population rate: {np.max(population_rate):.3f} Hz at t={np.argmax(population_rate) * bin_size} ms")
    print(f"Early activity: {early_rate:.3f} Hz")
    print(f"Late activity: {late_rate:.3f} Hz")
    print(f"Early/late delta: {rate_delta:.3f} Hz")
    print(
        f"Potential synchronized bins: {len(synchronized_bins)} "
        f"(threshold: {100 * sync_threshold:.1f}% active neurons per 100 ms bin)"
    )

    print("\n" + "-" * 50)
    print("4. CLUSTER-LEVEL ANALYSIS")
    print("-" * 50)
    unique_clusters = np.unique(cluster_assignments)
    cluster_rates = []
    for cid in unique_clusters:
        neuron_ids = np.where(cluster_assignments == cid)[0]
        rates = [firing_rates[nid] for nid in neuron_ids]
        cluster_rates.append(np.mean(rates) if len(rates) > 0 else 0.0)

    cluster_rates = np.array(cluster_rates)
    print(f"Number of clusters: {len(unique_clusters)}")
    print(f"Mean cluster firing rate: {np.mean(cluster_rates):.3f} +/- {np.std(cluster_rates):.3f} Hz")
    print(f"Cluster rate range: {np.min(cluster_rates):.3f} - {np.max(cluster_rates):.3f} Hz")
    print(f"Active clusters (>0.1 Hz mean): {np.sum(cluster_rates > 0.1)}/{len(unique_clusters)}")

    if connections is not None and len(connections) > 0:
        print("\n" + "-" * 50)
        print("5. CONNECTIVITY STATISTICS")
        print("-" * 50)
        n_exc = np.sum(connections[:, 3] == "exc")
        n_inh = np.sum(connections[:, 3] == "inh")
        print(f"Total connections: {len(connections)}")
        print(f"  Excitatory: {n_exc} ({100 * n_exc / len(connections):.1f}%)")
        print(f"  Inhibitory: {n_inh} ({100 * n_inh / len(connections):.1f}%)")
        print(f"Connection density: {100 * len(connections) / (n_neurons * (n_neurons - 1)):.2f}%")
        exc_weights = np.abs(connections[connections[:, 3] == "exc", 2].astype(float))
        inh_weights = np.abs(connections[connections[:, 3] == "inh", 2].astype(float))
        if len(exc_weights) > 0 and len(inh_weights) > 0:
            ei_ratio = np.sum(exc_weights) / np.sum(inh_weights)
            print(f"E/I weight ratio: {ei_ratio:.2f}")

    print("\n" + "=" * 70)
    print("MODEL QUALITY VERDICT")
    print("=" * 70)
    issues = []
    good_points = []
    mean_rate = float(np.mean(firing_rates)) if len(firing_rates) > 0 else 0.0
    stability_tolerance = max(0.25, 0.5 * max(mean_rate, 0.05))

    if expected_mode == "spontaneous":
        if 0.05 <= mean_rate <= 2.0:
            good_points.append("Mean firing rate is within a sparse spontaneous range.")
        elif mean_rate < 0.05:
            issues.append("Mean firing rate is very low; the network may be too silent.")
        else:
            issues.append("Mean firing rate is high for spontaneous activity.")

        if active_fraction >= 0.30:
            good_points.append("A substantial fraction of neurons are participating.")
        else:
            issues.append("Too few neurons are active during the recording.")

        if len(synchronized_bins) == 0:
            good_points.append("No obvious population-wide synchronized bursts were detected.")
        else:
            issues.append(
                f"Detected {len(synchronized_bins)} potentially synchronized 100 ms bins; "
                "this suggests auto-bursting."
            )

        if abs(rate_delta) <= stability_tolerance:
            good_points.append("Early and late activity levels are reasonably stable.")
        else:
            issues.append("Activity level drifts substantially across the recording.")
    else:
        if len(synchronized_bins) > 0:
            good_points.append("Synchronized population events were detected.")
        else:
            issues.append("No synchronized population events were detected.")

        if stimulus_enabled:
            good_points.append("Stimulus onset metadata is present for alignment checks.")
        else:
            issues.append("Expected stimulus-driven mode, but no stimulus onsets were saved.")

    if refractory_violations == 0:
        good_points.append("No refractory period violations detected.")
    else:
        issues.append(f"Detected {refractory_violations} ISIs below 2 ms.")

    if cv_isis and np.mean(cv_isis) >= 0.5:
        good_points.append("Spike trains retain variability rather than becoming overly regular.")

    print("\nPOSITIVE ASPECTS:")
    if good_points:
        for point in good_points:
            print(f"  - {point}")
    else:
        print("  - None")

    print("\nISSUES TO ADDRESS:")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  - No major issues detected.")

    print("\n" + "=" * 70)
    return {
        "expected_mode": expected_mode,
        "stimulus_enabled": stimulus_enabled,
        "burst_onset_times": burst_onset_times,
        "firing_rates": firing_rates,
        "spike_counts": spike_counts,
        "all_isis": all_isis,
        "cv_isis": cv_isis,
        "population_rate": population_rate,
        "cluster_rates": cluster_rates,
        "active_fraction_per_bin": active_fraction_per_bin,
        "synchronized_bins": synchronized_bins,
        "sync_threshold": sync_threshold,
    }


def compute_hub_firing_rate_groups(spike_data_dict, hub_ids, duration_ms):
    """Split firing rates into hub and non-hub groups for comparison plots.

    Args:
        spike_data_dict: Mapping from neuron id to spike times in milliseconds.
        hub_ids: Iterable of neuron ids designated as hubs.
        duration_ms: Recording duration in milliseconds.

    Returns:
        A dictionary containing the per-neuron rate arrays and summary means for
        hub and non-hub populations.
    """
    hub_set = set(hub_ids)
    duration_s = duration_ms / 1000.0
    hub_rates = []
    non_hub_rates = []

    for neuron_id, spikes in spike_data_dict.items():
        rate = len(spikes) / duration_s
        if neuron_id in hub_set:
            hub_rates.append(rate)
        else:
            non_hub_rates.append(rate)

    hub_rates = np.array(hub_rates, dtype=float)
    non_hub_rates = np.array(non_hub_rates, dtype=float)
    hub_mean = float(np.mean(hub_rates)) if hub_rates.size > 0 else 0.0
    non_hub_mean = float(np.mean(non_hub_rates)) if non_hub_rates.size > 0 else 0.0
    rate_ratio = hub_mean / non_hub_mean if non_hub_mean > 0 else float("inf")
    return {
        "hub_rates": hub_rates,
        "non_hub_rates": non_hub_rates,
        "hub_mean_rate": hub_mean,
        "non_hub_mean_rate": non_hub_mean,
        "rate_ratio": rate_ratio,
    }