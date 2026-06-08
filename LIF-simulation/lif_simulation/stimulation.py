import numpy as np


def create_initial_stimulation(
    neurons,
    cluster_info,
    stim_amplitude_range=(0.8, 1.2),
    stim_duration_range=(50, 100),
    fraction_to_stimulate=0.3,
):
    """Create a one-time stimulation event set applied at the start of a run.

    Args:
        neurons: Sequence of neurons in the simulated network.
        cluster_info: Unused cluster metadata kept for API compatibility.
        stim_amplitude_range: Inclusive amplitude range in nA used when sampling
            stimulus strengths.
        stim_duration_range: Inclusive duration range in ms used when sampling
            stimulus durations.
        fraction_to_stimulate: Fraction of neurons to stimulate at time 0.

    Returns:
        A list of ``(time_ms, neuron_id, amplitude_nA, duration_ms)`` tuples.
    """
    stimulation_events = []
    all_neuron_ids = list(range(len(neurons)))
    num_to_stim = max(1, int(len(neurons) * fraction_to_stimulate))
    stim_neurons = np.random.choice(all_neuron_ids, num_to_stim, replace=False)

    for neuron_id in stim_neurons:
        amplitude = np.random.uniform(*stim_amplitude_range)
        duration = np.random.uniform(*stim_duration_range)
        stimulation_events.append((0.0, neuron_id, amplitude, duration))

    print(f"Created initial stimulation: {len(stimulation_events)} neurons at t=0")
    print(
        f"  Amplitude range: {stim_amplitude_range[0]:.2f} - "
        f"{stim_amplitude_range[1]:.2f} nA"
    )
    print(
        f"  Duration range: {stim_duration_range[0]:.0f} - "
        f"{stim_duration_range[1]:.0f} ms"
    )
    return stimulation_events


def create_periodic_cluster_stimulation(
    neurons,
    cluster_info,
    burst_interval=7000,
    cluster_fraction=0.7,
    neurons_per_cluster=6,
    stim_amplitude_range=(2.0, 3.5),
    stim_duration_range=(10, 30),
    simulation_duration=60000,
    burst_interval_jitter=1500,
    per_cluster_jitter=75,
    per_neuron_jitter=20,
):
    """Schedule repeated cluster-level stimulation bursts across a recording.

    Args:
        neurons: Sequence of neurons in the simulated network.
        cluster_info: Cluster metadata containing ``cluster_neuron_groups``.
        burst_interval: Nominal interval between successive bursts in milliseconds.
        cluster_fraction: Fraction of clusters stimulated during each burst.
        neurons_per_cluster: Number of neurons stimulated in each selected cluster.
        stim_amplitude_range: Inclusive amplitude range in nA for each stimulus.
        stim_duration_range: Inclusive duration range in ms for each stimulus.
        simulation_duration: Total recording duration in milliseconds.
        burst_interval_jitter: Random jitter applied to each burst onset in ms.
        per_cluster_jitter: Additional cluster-specific burst-time jitter in ms.
        per_neuron_jitter: Additional per-neuron stimulus jitter in ms.

    Returns:
        A tuple ``(stimulation_events, burst_onset_times)`` containing the full
        stimulation schedule and the coarse burst onsets used for alignment.
    """
    if simulation_duration <= 0:
        print("\nStimulation disabled: simulation duration is non-positive.")
        return [], []

    if cluster_fraction <= 0 or neurons_per_cluster <= 0:
        print("\nStimulation disabled: cluster_fraction <= 0 or neurons_per_cluster <= 0.")
        return [], []

    if burst_interval <= 0:
        raise ValueError("burst_interval must be positive.")

    cluster_neuron_groups = cluster_info["cluster_neuron_groups"]
    num_clusters = len(cluster_neuron_groups)
    if num_clusters == 0:
        print("\nStimulation disabled: no clusters available.")
        return [], []

    stimulation_events = []
    burst_onset_times = []
    num_clusters_to_stim = min(
        num_clusters,
        max(1, int(np.ceil(num_clusters * cluster_fraction))),
    )

    num_bursts = int(np.floor(simulation_duration / burst_interval))
    if num_bursts <= 0:
        print(
            "\nNo stimulation bursts scheduled: "
            f"simulation_duration={simulation_duration} ms is shorter than "
            f"burst_interval={burst_interval} ms."
        )
        return [], []

    print("\nCreating periodic cluster stimulation:")
    print(
        f"  Burst interval: {burst_interval} ms ({1000 / burst_interval:.2f} Hz) +/- "
        f"{burst_interval_jitter} ms jitter"
    )
    print(f"  Number of bursts: {num_bursts}")
    print(
        f"  Clusters per burst: {num_clusters_to_stim} / {num_clusters} "
        f"({cluster_fraction * 100:.0f}%)"
    )
    print(f"  Neurons per cluster: {neurons_per_cluster}")
    print(
        f"  Stimulus amplitude: {stim_amplitude_range[0]:.2f} - "
        f"{stim_amplitude_range[1]:.2f} nA"
    )
    print(
        f"  Stimulus duration: {stim_duration_range[0]:.1f} - "
        f"{stim_duration_range[1]:.1f} ms"
    )
    print(f"  Per-cluster jitter: +/-{per_cluster_jitter} ms")
    print(f"  Per-neuron jitter: +/-{per_neuron_jitter} ms")

    for burst_idx in range(num_bursts):
        jitter = np.random.uniform(-burst_interval_jitter, burst_interval_jitter)
        burst_time = max(0.0, burst_idx * burst_interval + jitter)
        events_before_burst = len(stimulation_events)

        selected_clusters = np.random.choice(num_clusters, num_clusters_to_stim, replace=False)
        for cluster_id in selected_clusters:
            cluster_neurons = cluster_neuron_groups[cluster_id]
            cluster_offset = np.random.uniform(-per_cluster_jitter, per_cluster_jitter)

            n_to_stim = min(neurons_per_cluster, len(cluster_neurons))
            if n_to_stim == 0:
                continue

            stim_neuron_ids = np.random.choice(cluster_neurons, n_to_stim, replace=False)
            for neuron_id in stim_neuron_ids:
                amplitude = np.random.uniform(*stim_amplitude_range)
                duration = np.random.uniform(*stim_duration_range)
                neuron_offset = np.random.uniform(-per_neuron_jitter, per_neuron_jitter)
                stim_time = max(0.0, burst_time + cluster_offset + neuron_offset)
                stimulation_events.append((stim_time, neuron_id, amplitude, duration))

        if len(stimulation_events) > events_before_burst:
            burst_onset_times.append(burst_time)

    stimulation_events.sort(key=lambda item: item[0])
    burst_onset_times.sort()

    print(f"  Total stimulation events: {len(stimulation_events)}")
    if burst_onset_times:
        print(f"  Burst onset times (s): {[f'{t / 1000:.1f}' for t in burst_onset_times]}")
    else:
        print("  Burst onset times (s): []")

    return stimulation_events, burst_onset_times