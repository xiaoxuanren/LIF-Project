import numpy as np

from .models import ExpSynapse, LIFNeuron, NetworkWeightParameters


def generate_non_overlapping_cluster_positions(num_clusters, cluster_radius, space_size=50):
    """Place cluster centers while keeping clusters from overlapping spatially.

    Args:
        num_clusters: Number of cluster centers to attempt to place.
        cluster_radius: Radius later used to scatter neurons around each center.
        space_size: Width and height of the square embedding space.

    Returns:
        A NumPy array of shape ``(n_clusters_placed, 2)`` containing the sampled
        cluster-center coordinates. Fewer than ``num_clusters`` centers may be
        returned if the requested layout cannot fit inside ``space_size``.
    """
    positions = []
    min_distance = cluster_radius * 2.0
    max_attempts = 1000
    clusters_placed = 0

    for _ in range(num_clusters):
        placed = False
        for _attempt in range(max_attempts):
            pos = np.random.uniform(0, space_size, 2)
            if not positions:
                positions.append(pos)
                clusters_placed += 1
                placed = True
                break

            distances = np.sqrt(np.sum((np.array(positions) - pos) ** 2, axis=1))
            if np.all(distances >= min_distance):
                positions.append(pos)
                clusters_placed += 1
                placed = True
                break

        if not placed:
            print(
                f"Warning: Could only place {clusters_placed} out of {num_clusters} "
                f"clusters in space_size={space_size}"
            )
            break

    return np.array(positions)


def calculate_lognormal_params(mean_val, sigma):
    """Convert intuitive weight statistics into lognormal distribution parameters.

    Args:
        mean_val: Desired mean synaptic magnitude in linear space.
        sigma: Desired spread of synaptic magnitudes in linear space.

    Returns:
        A ``(mu, sigma_log)`` tuple that can be passed to
        ``np.random.lognormal``.
    """
    mu = np.log(mean_val / np.sqrt(1 + (sigma / mean_val) ** 2))
    sigma_log = np.sqrt(np.log(1 + (sigma / mean_val) ** 2))
    return mu, sigma_log


def generate_lognormal_weight(mean_val, sigma):
    """Sample one positive synaptic magnitude from the configured lognormal law.

    Args:
        mean_val: Desired mean synaptic magnitude in linear space.
        sigma: Desired spread of synaptic magnitudes in linear space.

    Returns:
        A positive floating-point synaptic magnitude.
    """
    mu, sigma_log = calculate_lognormal_params(mean_val, sigma)
    return np.random.lognormal(mu, sigma_log)


def get_connection_weight(pre_cluster, post_cluster, is_inhibitory, weight_params):
    """Sample a signed synaptic weight for one candidate connection.

    Args:
        pre_cluster: Cluster index of the presynaptic neuron.
        post_cluster: Cluster index of the postsynaptic neuron.
        is_inhibitory: Whether the presynaptic neuron is inhibitory.
        weight_params: ``NetworkWeightParameters`` describing the allowed weight
            ranges and whether lognormal sampling is enabled.

    Returns:
        A signed synaptic weight. Inhibitory weights are returned as negative
        values and excitatory weights as positive values.
    """
    if pre_cluster == post_cluster:
        weight_range = (
            weight_params.within_inh_range if is_inhibitory else weight_params.within_exc_range
        )
    else:
        weight_range = (
            weight_params.between_inh_range if is_inhibitory else weight_params.between_exc_range
        )

    if weight_params.use_lognormal:
        mean_val = np.mean(np.abs(weight_range))
        weight = generate_lognormal_weight(mean_val, weight_params.lognormal_sigma)
        weight = np.clip(weight, np.min(np.abs(weight_range)), np.max(np.abs(weight_range)))
        if is_inhibitory:
            weight = -weight
    else:
        weight = np.random.uniform(*weight_range)

    return weight


def calculate_connection_probability(
    distance,
    same_cluster,
    within_cluster_prob=0.5,
    between_cluster_prob=0.1,
    max_distance=3.0,
    distance_decay="gaussian",
    decay_sigma=1.0,
):
    """Compute the probability that a neuron pair should be connected.

    Args:
        distance: Euclidean distance between the presynaptic and postsynaptic neurons.
        same_cluster: Whether the pair belongs to the same cluster.
        within_cluster_prob: Base connection probability for same-cluster pairs.
        between_cluster_prob: Base connection probability for different-cluster pairs.
        max_distance: Hard spatial cutoff beyond which connections are disallowed.
        distance_decay: Distance falloff model, typically ``gaussian`` or
            ``exponential``.
        decay_sigma: Spread parameter used by the distance-decay model.

    Returns:
        A probability between 0 and 1 after applying the cluster-specific base
        probability and the distance-dependent attenuation.
    """
    base_prob = within_cluster_prob if same_cluster else between_cluster_prob

    if distance > max_distance:
        return 0.0

    if distance_decay == "gaussian":
        distance_factor = np.exp(-(distance ** 2) / (2 * decay_sigma ** 2))
    elif distance_decay == "exponential":
        distance_factor = np.exp(-distance / decay_sigma)
    else:
        distance_factor = 1.0

    return base_prob * distance_factor


def designate_hub_neurons(cluster_neuron_groups, hub_fraction=0.1):
    """Choose per-cluster hub neurons used for long-range hub projections.

    Args:
        cluster_neuron_groups: List whose entries contain the neuron ids for each
            cluster.
        hub_fraction: Fraction of neurons in each cluster that should be labeled
            as hubs.

    Returns:
        A tuple ``(hub_neuron_ids, hub_cluster_map)`` containing the global ids of
        all hubs and a mapping from hub neuron id to its cluster index.
    """
    hub_neuron_ids = []
    hub_cluster_map = {}

    for cluster_idx, neuron_ids in enumerate(cluster_neuron_groups):
        n_hubs = max(1, int(len(neuron_ids) * hub_fraction))
        hub_local_indices = np.random.choice(len(neuron_ids), n_hubs, replace=False)
        for local_idx in hub_local_indices:
            global_id = neuron_ids[local_idx]
            hub_neuron_ids.append(global_id)
            hub_cluster_map[global_id] = cluster_idx

    return hub_neuron_ids, hub_cluster_map


def add_hub_connections(
    neurons,
    hub_neuron_ids,
    hub_cluster_map,
    cluster_neuron_groups,
    cluster_assignments,
    neuron_positions,
    synapses,
    connections_list,
    weight_params,
    max_connection_distance=8.0,
    hub_between_prob=0.4,
    hub_weight_scale=1.5,
    hub_reciprocal_factor=2.0,
):
    """Add extra long-range inter-cluster connections from designated hub neurons.

    Args:
        neurons: List of neuron objects in the generated network.
        hub_neuron_ids: Global ids of neurons designated as hubs.
        hub_cluster_map: Mapping from hub neuron id to its home cluster.
        cluster_neuron_groups: Per-cluster lists of neuron ids.
        cluster_assignments: Cluster index for each neuron id.
        neuron_positions: Spatial coordinates for all neurons.
        synapses: Existing list of synapse objects to append to.
        connections_list: Existing connection table to append to.
        weight_params: Weight-sampling configuration.
        max_connection_distance: Maximum distance allowed for added hub edges.
        hub_between_prob: Base probability for a hub to target neurons in other
            clusters.
        hub_weight_scale: Multiplicative factor applied to hub-originating weights.
        hub_reciprocal_factor: Extra probability multiplier used when the target is
            also a hub.

    Returns:
        A list of the newly created hub-connection records that were appended to
        ``connections_list``.
    """
    hub_set = set(hub_neuron_ids)
    hub_connections = []
    existing_pairs = set()

    for conn in connections_list:
        existing_pairs.add((int(conn[0]), int(conn[1])))

    for hub_id in hub_neuron_ids:
        hub_neuron = neurons[hub_id]
        hub_cluster = cluster_assignments[hub_id]
        hub_pos = neuron_positions[hub_id]

        for target_cluster_idx, target_neuron_ids in enumerate(cluster_neuron_groups):
            if target_cluster_idx == hub_cluster:
                continue

            for target_id in target_neuron_ids:
                if (hub_id, target_id) in existing_pairs:
                    continue

                distance = np.sqrt(np.sum((hub_pos - neuron_positions[target_id]) ** 2))
                if distance > max_connection_distance:
                    continue

                prob = hub_between_prob
                if target_id in hub_set:
                    prob = min(1.0, prob * hub_reciprocal_factor)

                if np.random.random() < prob:
                    base_weight = get_connection_weight(
                        hub_cluster,
                        target_cluster_idx,
                        hub_neuron.is_inhibitory,
                        weight_params,
                    )
                    weight = base_weight * hub_weight_scale

                    target_neuron = neurons[target_id]
                    synapse = ExpSynapse(
                        hub_id,
                        target_neuron,
                        weight,
                        hub_neuron.is_inhibitory,
                    )
                    synapses.append(synapse)

                    conn_type = "inh" if hub_neuron.is_inhibitory else "exc"
                    conn_entry = [hub_id, target_id, weight, conn_type]
                    connections_list.append(conn_entry)
                    hub_connections.append(conn_entry)
                    existing_pairs.add((hub_id, target_id))

    return hub_connections


def assign_baseline_drive(neurons, mean=0.11, sd=0.05, seed=0, excitatory_only=True):
    """Assign frozen heterogeneous baseline current to neurons.

    Args:
        neurons: Sequence of ``LIFNeuron`` objects to configure.
        mean: Mean baseline drive in nA-equivalent units. For the default
            excitatory parameters, this should stay below rheobase at about
            0.115 nA-equivalent.
        sd: Standard deviation of the Gaussian baseline-drive distribution.
        seed: Seed for the one-time frozen-disorder draw.
        excitatory_only: Whether inhibitory neurons should receive zero baseline
            drive and be recruited only synaptically.

    Returns:
        None. The function mutates each neuron's ``i_baseline`` in place.
    """
    if sd < 0.0:
        raise ValueError("Baseline-drive standard deviation must be non-negative.")

    default_exc_rheobase = (-50.0 - -61.5) / 100.0
    if mean >= default_exc_rheobase:
        raise ValueError(
            "Baseline-drive mean should stay below the default excitatory "
            f"rheobase (~{default_exc_rheobase:.3f} nA-equivalent)."
        )

    rng = np.random.default_rng(seed)
    for neuron in neurons:
        if excitatory_only and neuron.is_inhibitory:
            neuron.i_baseline = 0.0
        else:
            neuron.i_baseline = max(0.0, float(rng.normal(mean, sd)))


def scale_excitatory_weights(synapses, scale, connections=None):
    """Scale recurrent excitatory synaptic strength in place.

    Args:
        synapses: Sequence of ``ExpSynapse`` objects to update.
        scale: Multiplicative scale applied to excitatory weights and conductance
            increments.
        connections: Optional connection table to keep saved weights consistent
            with the synapse objects.

    Returns:
        None. The function mutates synapses and, when supplied, the connection
        table in place.
    """
    if scale < 0.0:
        raise ValueError("Excitatory weight scale must be non-negative.")

    for synapse in synapses:
        if not synapse.is_inhibitory:
            synapse.weight *= scale
            synapse.g_increment *= scale

    if connections is not None:
        for connection in connections:
            if connection[3] == "exc":
                connection[2] = float(connection[2]) * scale


def create_clustered_network(
    num_clusters=20,
    neurons_per_cluster_range=(12, 18),
    inhibitory_probability=0.2,
    cluster_radius=1.0,
    within_cluster_prob=0.5,
    between_cluster_prob=0.15,
    max_connection_distance=8.0,
    weight_params=None,
    space_size=15,
    hub_fraction=0.1,
    hub_between_prob=0.4,
    hub_weight_scale=1.5,
    hub_reciprocal_factor=2.0,
    use_h_current=True,
    background_noise_sigma=0.0,
):
    """Build the clustered conductance-based network used by the simulation pipeline.

    Args:
        num_clusters: Requested number of spatial clusters.
        neurons_per_cluster_range: Inclusive range used to sample cluster sizes.
        inhibitory_probability: Probability that a neuron is inhibitory.
        cluster_radius: Radius used when scattering neurons around each cluster center.
        within_cluster_prob: Base connection probability for same-cluster pairs.
        between_cluster_prob: Base connection probability for inter-cluster pairs.
        max_connection_distance: Maximum spatial distance allowed for a connection.
        weight_params: Weight-distribution settings for excitatory and inhibitory synapses.
        space_size: Width and height of the square embedding space.
        hub_fraction: Fraction of neurons per cluster promoted to hubs.
        hub_between_prob: Inter-cluster connection probability used for hub projections.
        hub_weight_scale: Multiplier applied to hub-originating weights.
        hub_reciprocal_factor: Extra probability boost for hub-to-hub projections.
        use_h_current: Whether newly created neurons should update the h-current.
        background_noise_sigma: Standard deviation of the additive membrane-noise
            term assigned to each created neuron.

    Returns:
        Neurons, synapses, connection table, neuron positions, and cluster metadata.
    """
    if weight_params is None:
        weight_params = NetworkWeightParameters()

    cluster_centers = generate_non_overlapping_cluster_positions(
        num_clusters,
        cluster_radius,
        space_size,
    )
    num_clusters_actual = len(cluster_centers)

    cluster_sizes = np.random.randint(
        neurons_per_cluster_range[0],
        neurons_per_cluster_range[1] + 1,
        size=num_clusters_actual,
    )

    neurons = []
    neuron_positions = []
    cluster_assignments = []
    cluster_neuron_groups = [[] for _ in range(num_clusters_actual)]

    neuron_id = 0
    for cluster_id in range(num_clusters_actual):
        center = cluster_centers[cluster_id]
        n_neurons = cluster_sizes[cluster_id]

        for _ in range(n_neurons):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, cluster_radius)
            pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            is_inhibitory = np.random.random() < inhibitory_probability

            neuron = LIFNeuron(
                neuron_id,
                is_inhibitory,
                use_h_current=use_h_current,
                noise_sigma=background_noise_sigma,
            )
            neurons.append(neuron)
            neuron_positions.append(pos)
            cluster_assignments.append(cluster_id)
            cluster_neuron_groups[cluster_id].append(neuron_id)
            neuron_id += 1

    neuron_positions = np.array(neuron_positions)
    cluster_assignments = np.array(cluster_assignments)

    print(f"Created {len(neurons)} neurons in {num_clusters_actual} clusters")

    synapses = []
    connections = []
    for pre_id, pre_neuron in enumerate(neurons):
        pre_pos = neuron_positions[pre_id]
        pre_cluster = cluster_assignments[pre_id]

        for post_id, post_neuron in enumerate(neurons):
            if pre_id == post_id:
                continue

            post_pos = neuron_positions[post_id]
            post_cluster = cluster_assignments[post_id]
            distance = np.sqrt(np.sum((pre_pos - post_pos) ** 2))
            same_cluster = pre_cluster == post_cluster
            prob = calculate_connection_probability(
                distance,
                same_cluster,
                within_cluster_prob,
                between_cluster_prob,
                max_connection_distance,
            )

            if np.random.random() < prob:
                weight = get_connection_weight(
                    pre_cluster,
                    post_cluster,
                    pre_neuron.is_inhibitory,
                    weight_params,
                )
                synapse = ExpSynapse(
                    pre_id,
                    post_neuron,
                    weight,
                    pre_neuron.is_inhibitory,
                )
                synapses.append(synapse)

                conn_type = "inh" if pre_neuron.is_inhibitory else "exc"
                connections.append([pre_id, post_id, weight, conn_type])

    n_base_connections = len(connections)
    print(f"Created {n_base_connections} base synaptic connections")

    n_within = sum(
        1
        for conn in connections
        if cluster_assignments[int(conn[0])] == cluster_assignments[int(conn[1])]
    )
    n_between = n_base_connections - n_within
    print(f"  Within-cluster: {n_within}, Between-cluster: {n_between}")

    hub_neuron_ids, hub_cluster_map = designate_hub_neurons(
        cluster_neuron_groups,
        hub_fraction,
    )
    print(
        f"\nDesignated {len(hub_neuron_ids)} hub neurons "
        f"({100 * len(hub_neuron_ids) / len(neurons):.1f}% of network)"
    )

    hub_connections = add_hub_connections(
        neurons=neurons,
        hub_neuron_ids=hub_neuron_ids,
        hub_cluster_map=hub_cluster_map,
        cluster_neuron_groups=cluster_neuron_groups,
        cluster_assignments=cluster_assignments,
        neuron_positions=neuron_positions,
        synapses=synapses,
        connections_list=connections,
        weight_params=weight_params,
        max_connection_distance=max_connection_distance,
        hub_between_prob=hub_between_prob,
        hub_weight_scale=hub_weight_scale,
        hub_reciprocal_factor=hub_reciprocal_factor,
    )

    print(f"Added {len(hub_connections)} hub connections")
    print(
        f"Total connections: {len(connections)} "
        f"(base: {n_base_connections}, hub: {len(hub_connections)})"
    )

    connections = np.array(connections, dtype=object)
    cluster_info = {
        "cluster_centers": cluster_centers,
        "cluster_sizes": cluster_sizes,
        "cluster_assignments": cluster_assignments,
        "cluster_neuron_groups": cluster_neuron_groups,
        "hub_neuron_ids": hub_neuron_ids,
        "hub_cluster_map": hub_cluster_map,
        "n_hub_connections": len(hub_connections),
        "hub_fraction": hub_fraction,
        "hub_between_prob": hub_between_prob,
        "hub_weight_scale": hub_weight_scale,
        "hub_reciprocal_factor": hub_reciprocal_factor,
        "use_h_current": bool(use_h_current),
    }
    return neurons, synapses, connections, neuron_positions, cluster_info