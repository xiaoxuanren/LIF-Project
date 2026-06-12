"""Adapter: build a LIF-Project network from an nc_sim grown topology.

This module lets you generate the network *topology* with the spatial axon-growth
model of nc_sim (Houben, Garcia-Ojalvo & Soriano 2025;
github.com/akkeh/nc_sim) while keeping the LIF-Project conductance-based
``LIFNeuron`` / ``ExpSynapse`` dynamics, saving, and inference unchanged.

``build_network_from_nc_sim`` mirrors the return contract of
``create_clustered_network`` exactly::

    neurons, synapses, connections, neuron_positions, cluster_info

so it is a drop-in replacement at the two call sites
(``workflows.py`` and ``probes.py``). Everything downstream
(``save_network_structure``, the simulation loop, ``build_ground_truth``) works
without modification.

Why use this instead of ``create_clustered_network``:
    * Connections are grown from distance-dependent axon paths, so nearby
      neurons share *spatially structured* presynaptic drive -- the realistic
      form of the common-input confound that dominates your false positives,
      rather than the i.i.d. drive an Erdos-Renyi / pure-probability rule gives.
    * The average axon length ``L`` is a *structural* burst-synchrony dial
      (short L -> sparse, fragmented, weak global bursting; long L -> dense,
      globally synchronized). That is an axis *independent* of your dynamical
      knobs (adaptation, baseline drive, stimulus mode), giving you two
      orthogonal handles to decouple burst statistics from wiring on the
      validation ladder.
    * The obstacle grid ``H`` adds tunable anisotropy/modularity that suppresses
      pathological whole-culture bursting when you want richer regimes.

What is imported vs. native:
    * TOPOLOGY (who connects to whom, directionality, E/I identity, positions)
      comes entirely from nc_sim.
    * WEIGHTS are *not* taken from nc_sim (which only emits +/-1). They are drawn
      from LIF-Project's own ``NetworkWeightParameters`` via
      ``get_connection_weight`` so the synaptic-strength distribution matches
      your usual networks. Swap ``weight_params`` to change this.
    * DYNAMICS are 100% LIF-Project: ``ExpSynapse`` has no short-term depression
      (unlike nc_sim's synapse), so burst structure here comes from your
      ``LIFNeuron`` adaptation + h-current + stimulus mode, exactly as in your
      native pipeline.

Requirements:
    nc_sim must be importable (``pip install`` the repo, or vendor its two
    numpy-only files ``axons/grow_axons.py`` and ``__init__.py`` into your tree).

Convention notes (verified against both codebases):
    * nc_sim ``grow_W`` returns a dense ``W[post, pre]`` matrix (row = target /
      dendrite, col = source / axon) with entries in {-1, 0, +1}; the last
      ``(1 - EIratio)`` fraction of *columns* are inhibitory sources (negative).
    * LIF-Project's ``connections`` table is source-first: ``[pre, post, w, type]``.
    * Mapping is therefore: for each nonzero ``W[post, pre]``, emit an edge with
      ``pre = col``, ``post = row``. ``build_ground_truth`` then rebuilds
      ``W[post, pre]`` identically -- the round-trip is the identity, so there is
      no transpose ambiguity.
"""

from __future__ import annotations

import numpy as np

from .models import ExpSynapse, LIFNeuron, NetworkWeightParameters
from .network import get_connection_weight


def _import_nc_sim():
    """Import nc_sim with a helpful error if it is not on the path."""
    try:
        import nc_sim  # noqa: WPS433 (runtime import is intentional)

        return nc_sim
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "nc_sim is required by nc_sim_adapter. Install it from "
            "https://github.com/akkeh/nc_sim (numpy-only) or vendor its two "
            "files into your project so `import nc_sim` succeeds."
        ) from exc


def _assign_spatial_clusters(neuron_positions, num_clusters):
    """Partition neurons into a spatial grid for weight sampling / bookkeeping.

    These clusters do **not** drive connectivity (nc_sim growth already fixed
    that). They exist only so that (a) ``get_connection_weight`` can use the
    within- vs between-cluster weight ranges and (b) ``cluster_info`` is
    non-degenerate for saving and any cluster-aware plotting/metrics.

    Args:
        neuron_positions: ``(N, 2)`` array of neuron coordinates.
        num_clusters: Approximate number of clusters. ``1`` puts every neuron in
            a single cluster (all weights then use the within-cluster ranges).

    Returns:
        A tuple ``(cluster_assignments, cluster_centers, cluster_sizes,
        cluster_neuron_groups)`` with contiguous cluster ids and no empty
        clusters.
    """
    n = len(neuron_positions)
    if num_clusters <= 1 or n == 0:
        assignments = np.zeros(n, dtype=int)
    else:
        # Square-ish grid that yields ~num_clusters cells.
        side = max(1, int(np.ceil(np.sqrt(num_clusters))))
        xs, ys = neuron_positions[:, 0], neuron_positions[:, 1]

        def _bin(values, n_bins):
            lo, hi = float(np.min(values)), float(np.max(values))
            if hi <= lo:
                return np.zeros(len(values), dtype=int)
            idx = ((values - lo) / (hi - lo) * n_bins).astype(int)
            return np.clip(idx, 0, n_bins - 1)

        raw = _bin(xs, side) * side + _bin(ys, side)
        # Relabel to contiguous ids, dropping empty cells.
        _, assignments = np.unique(raw, return_inverse=True)

    num_actual = int(assignments.max()) + 1 if n else 0
    cluster_neuron_groups = [[] for _ in range(num_actual)]
    for nid, cid in enumerate(assignments):
        cluster_neuron_groups[int(cid)].append(nid)

    cluster_centers = np.array(
        [
            neuron_positions[group].mean(axis=0) if group else np.zeros(2)
            for group in cluster_neuron_groups
        ]
    )
    cluster_sizes = np.array([len(group) for group in cluster_neuron_groups], dtype=int)
    return assignments, cluster_centers, cluster_sizes, cluster_neuron_groups


def build_network_from_nc_sim(
    *,
    width=1.0,
    height=1.0,
    obstacles=None,
    rho=100.0,
    axon_length=1.0,
    ei_ratio=0.8,
    exc_axon_length=-1.0,
    inh_axon_length=-1.0,
    phi_sd=0.1,
    r_dendrite_mu=150e-3,
    r_dendrite_sd=20e-3,
    weight_params=None,
    num_clusters=20,
    use_h_current=True,
    background_noise_sigma=0.0,
    seed=None,
    verbose=True,
):
    """Grow a topology with nc_sim and wrap it as a LIF-Project network.

    Drop-in replacement for ``create_clustered_network``: same return tuple,
    same downstream compatibility.

    Args:
        width: Culture width in mm (nc_sim units).
        height: Culture height in mm.
        obstacles: ``H`` obstacle grid for nc_sim (``M x N`` of heights in mm;
            ``0`` = flat, ``h > 0`` = PDMS obstacle, ``h < 0`` = forbidden). Use
            ``None`` for a flat isotropic culture.
        rho: Neuron density (neurons / mm^2). With ``width = height = 1`` this is
            roughly the neuron count. ``rho`` and ``axon_length`` are the density
            knobs.
        axon_length: Average axon length ``L`` in mm. Doubles as the structural
            burst-synchrony / maturity dial. ~1 mm ~ mature culture; ~0.2 mm ~
            early, fragmented. Shorter also means sparser.
        ei_ratio: Excitatory fraction. The last ``(1 - ei_ratio)`` of neuron ids
            are inhibitory, matching nc_sim's column-sign convention.
        exc_axon_length: Optional separate excitatory axon length (``-1`` = use
            ``axon_length``).
        inh_axon_length: Optional separate inhibitory axon length (``-1`` = use
            ``axon_length``).
        phi_sd: Axon random-walk angular std (nc_sim ``phi_sd``).
        r_dendrite_mu: Mean dendritic radius in mm (connection capture radius).
        r_dendrite_sd: Std of dendritic radius in mm.
        weight_params: ``NetworkWeightParameters`` controlling synaptic-strength
            sampling. ``None`` uses the project defaults.
        num_clusters: Spatial partition used only for weight ranges and
            bookkeeping (not connectivity). ``1`` -> single cluster.
        use_h_current: Passed to every ``LIFNeuron``.
        background_noise_sigma: Per-neuron additive membrane-noise std.
        seed: Optional seed for reproducible growth + weights.
        verbose: Print a short summary.

    Returns:
        ``(neurons, synapses, connections, neuron_positions, cluster_info)`` in
        the exact format of ``create_clustered_network``.
    """
    if seed is not None:
        np.random.seed(seed)
    if weight_params is None:
        weight_params = NetworkWeightParameters()

    nc_sim = _import_nc_sim()
    H = np.zeros((1, 1)) if obstacles is None else np.asarray(obstacles, dtype=float)

    # --- Stage 1: nc_sim spatial growth -> dense W[post, pre] in {-1, 0, +1} ---
    X, Y = nc_sim.axons.place_neurons(width, height, H, rho=rho)
    W = nc_sim.axons.grow_W(
        width,
        height,
        X,
        Y,
        H=H,
        L=axon_length,
        Le=exc_axon_length,
        Li=inh_axon_length,
        EIratio=ei_ratio,
        phi_sd=phi_sd,
        r_dendrite_mu=r_dendrite_mu,
        r_dendrite_sd=r_dendrite_sd,
    )

    n_neurons = len(X)
    n_exc = int(n_neurons * ei_ratio)
    # nc_sim marks neuron k inhibitory by making its outgoing column negative,
    # and those are exactly the last (n_neurons - n_exc) indices.
    is_inhibitory = np.arange(n_neurons) >= n_exc

    neuron_positions = np.column_stack([X, Y])  # mm

    # --- Spatial partition for weight sampling / bookkeeping only ---
    (
        cluster_assignments,
        cluster_centers,
        cluster_sizes,
        cluster_neuron_groups,
    ) = _assign_spatial_clusters(neuron_positions, num_clusters)

    # --- Build neurons (LIF-Project conductance LIF) ---
    neurons = [
        LIFNeuron(
            nid,
            bool(is_inhibitory[nid]),
            use_h_current=use_h_current,
            noise_sigma=background_noise_sigma,
        )
        for nid in range(n_neurons)
    ]

    # --- Build synapses + connection table from grown topology ---
    # W[post, pre] != 0  ==>  edge pre -> post.
    post_ids, pre_ids = np.nonzero(W)
    synapses = []
    connections = []
    for pre, post in zip(pre_ids.tolist(), post_ids.tolist()):
        inh = bool(is_inhibitory[pre])
        # Weight magnitude from LIF-Project's own distribution; sign handled
        # inside get_connection_weight (negative for inhibitory).
        weight = get_connection_weight(
            int(cluster_assignments[pre]),
            int(cluster_assignments[post]),
            inh,
            weight_params,
        )
        synapses.append(ExpSynapse(pre, neurons[post], weight, inh))
        connections.append([pre, post, weight, "inh" if inh else "exc"])

    connections = np.array(connections, dtype=object)

    # --- cluster_info in the exact shape create_clustered_network returns ---
    # nc_sim has no hub concept; leave hub fields empty/default for save+metric
    # compatibility.
    cluster_info = {
        "cluster_centers": cluster_centers,
        "cluster_sizes": cluster_sizes,
        "cluster_assignments": cluster_assignments,
        "cluster_neuron_groups": cluster_neuron_groups,
        "hub_neuron_ids": [],
        "hub_cluster_map": {},
        "n_hub_connections": 0,
        "hub_fraction": 0.0,
        "hub_between_prob": 0.0,
        "hub_weight_scale": 1.0,
        "hub_reciprocal_factor": 1.0,
        "use_h_current": bool(use_h_current),
    }

    if verbose:
        n_conn = len(connections)
        density = n_conn / (n_neurons * (n_neurons - 1)) if n_neurons > 1 else 0.0
        n_inh = int(is_inhibitory.sum())
        print(
            f"[nc_sim_adapter] {n_neurons} neurons "
            f"({n_neurons - n_inh} exc / {n_inh} inh), "
            f"{n_conn} synapses, density {density:.3f}, "
            f"{len(cluster_neuron_groups)} spatial clusters, "
            f"L={axon_length} mm, rho={rho}/mm^2"
        )

    return neurons, synapses, connections, neuron_positions, cluster_info
