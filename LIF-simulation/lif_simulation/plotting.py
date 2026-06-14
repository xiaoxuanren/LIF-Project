import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def _resolve_time_window(times, time_range):
    """Map a millisecond time window into a contiguous array slice."""
    if time_range is None:
        return slice(None), np.asarray(times)

    start_ms = float(time_range[0])
    end_ms = float(time_range[1])
    start_idx = int(np.searchsorted(times, start_ms, side="left"))
    end_idx = int(np.searchsorted(times, end_ms, side="right"))
    return slice(start_idx, end_idx), np.asarray(times[start_idx:end_idx])


def plot_raster(spike_data, cluster_assignments, duration, title="Network Activity"):
    """Plot a spike raster sorted by cluster assignment.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        cluster_assignments: Cluster index for each neuron id.
        duration: Displayed recording duration in milliseconds.
        title: Figure title.

    Returns:
        The created Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    neuron_ids = sorted(spike_data.keys(), key=lambda idx: cluster_assignments[idx])

    for plot_idx, neuron_id in enumerate(neuron_ids):
        spikes = spike_data[neuron_id]
        if len(spikes) > 0:
            ax.scatter(spikes, [plot_idx] * len(spikes), s=1, c="k", marker="|")

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Neuron", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, len(neuron_ids))
    plt.tight_layout()
    return fig


def plot_voltage_traces(voltage_data, neuron_ids=None, time_range=None, spike_data=None, title="Voltage Traces"):
    """Plot stored membrane traces for selected neurons over a time window.

    Args:
        voltage_data: Voltage bundle containing ``traces`` and ``times`` arrays.
        neuron_ids: Optional list of neuron ids to plot. Defaults to the first few
            stored traces.
        time_range: Optional ``(start_ms, end_ms)`` window used to crop the plot.
        spike_data: Optional spike-time mapping used to overlay spike markers.
        title: Figure title.

    Returns:
        The created Matplotlib figure.
    """
    traces = voltage_data["traces"]
    times = voltage_data["times"]
    if neuron_ids is None:
        neuron_ids = list(range(min(10, traces.shape[0])))

    time_slice, times_plot = _resolve_time_window(times, time_range)
    if len(times_plot) == 0:
        raise ValueError("Selected time_range does not overlap the stored voltage trace times.")

    fig, axes = plt.subplots(len(neuron_ids), 1, figsize=(15, 2 * len(neuron_ids)), sharex=True)
    if len(neuron_ids) == 1:
        axes = [axes]

    for idx, neuron_id in enumerate(neuron_ids):
        ax = axes[idx]
        voltage_trace = np.asarray(traces[neuron_id, time_slice], dtype=np.float32)
        ax.plot(times_plot, voltage_trace, "b-", linewidth=0.8)

        spike_marker_y = None
        if spike_data is not None and neuron_id in spike_data:
            spikes = np.asarray(spike_data[neuron_id], dtype=float)
            if time_range is not None:
                spikes = spikes[(spikes >= time_range[0]) & (spikes <= time_range[1])]
            if len(spikes) > 0:
                spike_marker_y = max(float(voltage_trace.max()) + 1.5, -48.0)
                ax.plot(
                    spikes,
                    np.full(len(spikes), spike_marker_y),
                    "rv",
                    markersize=4,
                    markeredgewidth=1,
                    alpha=0.7,
                    label="Spike time" if idx == 0 else None,
                )

        ax.axhline(-50, color="r", linestyle="--", linewidth=0.5, alpha=0.5, label="Threshold" if idx == 0 else None)
        lower_y = min(float(voltage_trace.min()) - 2.0, -80.0)
        upper_y = max(float(voltage_trace.max()) + 2.0, -45.0)
        if spike_marker_y is not None:
            upper_y = max(upper_y, spike_marker_y + 2.0)

        ax.set_ylabel(f"Neuron {neuron_id}\n(mV)", fontsize=10)
        ax.set_ylim(lower_y, upper_y)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.set_title(title, fontsize=14)
            handles, _labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    plt.tight_layout()
    return fig


def plot_voltage_heatmap(voltage_data, neuron_ids=None, time_range=None, cluster_assignments=None, title="Voltage Activity Heatmap"):
    """Render a neuron-by-time voltage heatmap, optionally sorted by cluster.

    Args:
        voltage_data: Voltage bundle containing ``traces`` and ``times`` arrays.
        neuron_ids: Optional subset of neuron ids to include in the heatmap.
        time_range: Optional ``(start_ms, end_ms)`` window used to crop the plot.
        cluster_assignments: Optional cluster index for each neuron, used to sort
            rows by cluster identity.
        title: Figure title.

    Returns:
        The created Matplotlib figure.
    """
    traces = voltage_data["traces"]
    times = voltage_data["times"]

    time_slice, times_plot = _resolve_time_window(times, time_range)
    if len(times_plot) == 0:
        raise ValueError("Selected time_range does not overlap the stored voltage trace times.")

    if neuron_ids is None:
        neuron_ids = list(range(traces.shape[0]))
    if cluster_assignments is not None:
        neuron_ids = sorted(neuron_ids, key=lambda idx: cluster_assignments[idx])

    traces_plot = np.asarray(traces[neuron_ids, time_slice], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(15, 8))
    image = ax.imshow(
        traces_plot,
        aspect="auto",
        cmap="RdBu_r",
        extent=[times_plot[0], times_plot[-1], len(neuron_ids), 0],
        vmin=-75,
        vmax=-45,
    )
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Neuron (sorted by cluster)", fontsize=12)
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Voltage (mV)", fontsize=12)
    plt.tight_layout()
    return fig


def _infer_cluster_assignments(neuron_positions, cluster_info, cluster_assignments=None):
    """Recover cluster assignments from explicit input or saved cluster metadata.

    Args:
        neuron_positions: Spatial coordinates for each neuron.
        cluster_info: Cluster metadata that may contain assignments or groups.
        cluster_assignments: Optional explicit cluster-assignment array.

    Returns:
        A NumPy array giving the cluster index for each neuron.
    """
    if cluster_assignments is not None:
        return cluster_assignments

    if "cluster_assignments" in cluster_info:
        return cluster_info["cluster_assignments"]

    if "cluster_neuron_groups" not in cluster_info:
        return np.zeros(len(neuron_positions), dtype=int)

    inferred = np.zeros(len(neuron_positions), dtype=int)
    for cluster_id, neuron_ids in enumerate(cluster_info["cluster_neuron_groups"]):
        for neuron_id in neuron_ids:
            inferred[neuron_id] = cluster_id
    return inferred


def plot_network_positions(
    neuron_positions,
    cluster_info,
    connections,
    cluster_assignments=None,
    show_connections="sample",
    max_connections=500,
):
    """Plot within-cluster and between-cluster connectivity in separate panels.

    Args:
        neuron_positions: Spatial coordinates for every neuron.
        cluster_info: Cluster metadata containing centers and neuron groups.
        connections: Connection table describing the generated synapses.
        cluster_assignments: Optional cluster index for each neuron.
        show_connections: Which connection subset to draw: ``all``, ``sample``,
            ``within``, or ``between``.
        max_connections: Maximum number of sampled connections to render when
            ``show_connections`` is ``sample``.

    Returns:
        The created Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    within_line_alpha = 0.55
    within_line_width = 1.1
    between_line_alpha = 0.7
    between_line_width = 1.45

    cluster_assignments = _infer_cluster_assignments(neuron_positions, cluster_info, cluster_assignments)
    within_conns = []
    between_conns = []
    for conn in connections:
        pre_id, post_id = int(conn[0]), int(conn[1])
        weight = float(conn[2])
        conn_type = conn[3]
        if cluster_assignments[pre_id] == cluster_assignments[post_id]:
            within_conns.append((pre_id, post_id, weight, conn_type))
        else:
            between_conns.append((pre_id, post_id, weight, conn_type))

    ax1 = axes[0]
    if show_connections in ["all", "sample", "within"]:
        conns_to_plot = within_conns
        if show_connections == "sample" and len(conns_to_plot) > max_connections:
            indices = np.random.choice(len(conns_to_plot), max_connections, replace=False)
            conns_to_plot = [within_conns[i] for i in indices]

        for pre_id, post_id, _weight, conn_type in conns_to_plot:
            pos_pre = neuron_positions[pre_id]
            pos_post = neuron_positions[post_id]
            color = "royalblue" if conn_type == "exc" else "firebrick"
            ax1.plot(
                [pos_pre[0], pos_post[0]],
                [pos_pre[1], pos_post[1]],
                color=color,
                alpha=within_line_alpha,
                linewidth=within_line_width,
            )

    colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_info["cluster_neuron_groups"])))
    for cluster_id, neuron_ids in enumerate(cluster_info["cluster_neuron_groups"]):
        positions = neuron_positions[neuron_ids]
        ax1.scatter(
            positions[:, 0],
            positions[:, 1],
            s=60,
            c=[colors[cluster_id]],
            edgecolors="black",
            linewidths=0.5,
            label=f"C{cluster_id}",
        )

    centers = cluster_info["cluster_centers"]
    ax1.scatter(centers[:, 0], centers[:, 1], s=150, c="black", marker="x", linewidths=2)
    ax1.set_xlabel("X Position", fontsize=12)
    ax1.set_ylabel("Y Position", fontsize=12)
    ax1.set_title(f"Within-Cluster Connections ({len(within_conns)} total)", fontsize=12)
    ax1.set_aspect("equal")

    ax2 = axes[1]
    if show_connections in ["all", "sample", "between"]:
        conns_to_plot = between_conns
        if show_connections == "sample" and len(conns_to_plot) > max_connections:
            indices = np.random.choice(len(conns_to_plot), max_connections, replace=False)
            conns_to_plot = [between_conns[i] for i in indices]

        for pre_id, post_id, _weight, conn_type in conns_to_plot:
            pos_pre = neuron_positions[pre_id]
            pos_post = neuron_positions[post_id]
            color = "midnightblue" if conn_type == "exc" else "darkred"
            ax2.plot(
                [pos_pre[0], pos_post[0]],
                [pos_pre[1], pos_post[1]],
                color=color,
                alpha=between_line_alpha,
                linewidth=between_line_width,
            )

    for cluster_id, neuron_ids in enumerate(cluster_info["cluster_neuron_groups"]):
        positions = neuron_positions[neuron_ids]
        ax2.scatter(
            positions[:, 0],
            positions[:, 1],
            s=60,
            c=[colors[cluster_id]],
            edgecolors="black",
            linewidths=0.5,
        )

    ax2.scatter(centers[:, 0], centers[:, 1], s=150, c="black", marker="x", linewidths=2)
    ax2.set_xlabel("X Position", fontsize=12)
    ax2.set_ylabel("Y Position", fontsize=12)
    ax2.set_title(f"Between-Cluster Connections ({len(between_conns)} total)", fontsize=12)
    ax2.set_aspect("equal")

    legend_elements = [
        Line2D([0], [0], color="midnightblue", linewidth=2.5, label="Excitatory"),
        Line2D([0], [0], color="darkred", linewidth=2.5, label="Inhibitory"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right")
    plt.suptitle("Network Spatial Layout", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_combined_network_layout(
    neuron_positions,
    cluster_info,
    connections,
    cluster_assignments=None,
    max_connections=600,
    title="Network Structure",
):
    """Plot the full network layout with sampled connections and highlighted hubs.

    Args:
        neuron_positions: Spatial coordinates for every neuron.
        cluster_info: Cluster metadata containing centers, groups, and optional hubs.
        connections: Connection table describing the generated synapses.
        cluster_assignments: Optional cluster index for each neuron.
        max_connections: Maximum number of connections to draw for readability.
        title: Figure title.

    Returns:
        The created Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cluster_assignments = _infer_cluster_assignments(neuron_positions, cluster_info, cluster_assignments)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_info["cluster_neuron_groups"])))

    connections_to_plot = connections
    if len(connections_to_plot) > max_connections:
        sampled_indices = np.random.choice(len(connections_to_plot), max_connections, replace=False)
        connections_to_plot = connections_to_plot[sampled_indices]

    for conn in connections_to_plot:
        pre_id = int(conn[0])
        post_id = int(conn[1])
        conn_type = conn[3]
        pos_pre = neuron_positions[pre_id]
        pos_post = neuron_positions[post_id]
        line_color = "steelblue" if conn_type == "exc" else "indianred"
        line_alpha = 0.18 if cluster_assignments[pre_id] == cluster_assignments[post_id] else 0.32
        ax.plot(
            [pos_pre[0], pos_post[0]],
            [pos_pre[1], pos_post[1]],
            color=line_color,
            alpha=line_alpha,
            linewidth=0.65,
        )

    for cluster_idx, neuron_ids in enumerate(cluster_info["cluster_neuron_groups"]):
        cluster_positions = neuron_positions[neuron_ids]
        ax.scatter(
            cluster_positions[:, 0],
            cluster_positions[:, 1],
            s=28,
            c=[cluster_colors[cluster_idx]],
            edgecolors="black",
            linewidths=0.25,
        )

    hub_ids = cluster_info.get("hub_neuron_ids", [])
    if hub_ids:
        hub_positions = neuron_positions[hub_ids]
        ax.scatter(
            hub_positions[:, 0],
            hub_positions[:, 1],
            s=140,
            marker="*",
            c="gold",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=f"Hub neurons ({len(hub_ids)})",
        )
        ax.legend(loc="upper right", fontsize=10)

    centers = cluster_info["cluster_centers"]
    ax.scatter(centers[:, 0], centers[:, 1], s=110, c="black", marker="x", linewidths=1.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X position", fontsize=12)
    ax.set_ylabel("Y position", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_firing_rates(spike_data, cluster_assignments, duration):
    """Plot firing-rate histograms and neuron-wise rates sorted by cluster.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        cluster_assignments: Cluster index for each neuron id.
        duration: Recording duration in milliseconds.

    Returns:
        The created Matplotlib figure.
    """
    firing_rates = []
    for _neuron_id, spikes in spike_data.items():
        firing_rates.append(len(spikes) / (duration / 1000.0))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(firing_rates, bins=50, edgecolor="black")
    axes[0].set_xlabel("Firing Rate (Hz)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Firing Rate Distribution", fontsize=14)

    neuron_ids = sorted(spike_data.keys(), key=lambda idx: cluster_assignments[idx])
    rates_sorted = [len(spike_data[nid]) / (duration / 1000.0) for nid in neuron_ids]
    axes[1].plot(rates_sorted, "o-", markersize=3)
    axes[1].set_xlabel("Neuron (sorted by cluster)", fontsize=12)
    axes[1].set_ylabel("Firing Rate (Hz)", fontsize=12)
    axes[1].set_title("Firing Rate by Neuron", fontsize=14)
    plt.tight_layout()
    return fig


def plot_hub_network(neuron_positions, connections, cluster_info, figsize=(14, 10)):
    """Visualize hub neurons and emphasize their long-range connections.

    Args:
        neuron_positions: Spatial coordinates for every neuron.
        connections: Connection table describing the generated synapses.
        cluster_info: Cluster metadata containing assignments, groups, and hub ids.
        figsize: Requested Matplotlib figure size.

    Returns:
        A tuple ``(fig, ax)`` containing the created figure and main axes.
    """
    hub_neuron_ids = cluster_info.get("hub_neuron_ids", [])
    hub_set = set(hub_neuron_ids)
    cluster_assignments = cluster_info["cluster_assignments"]
    cluster_neuron_groups = cluster_info["cluster_neuron_groups"]
    n_clusters = len(cluster_neuron_groups)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.tab20
    cluster_colors = [cmap(idx / n_clusters) for idx in range(n_clusters)]

    normal_conns = []
    hub_conns = []
    for conn in connections:
        pre_id, post_id = int(conn[0]), int(conn[1])
        if pre_id in hub_set or post_id in hub_set:
            if cluster_assignments[pre_id] != cluster_assignments[post_id]:
                hub_conns.append((pre_id, post_id))
            else:
                normal_conns.append((pre_id, post_id))
        else:
            normal_conns.append((pre_id, post_id))

    conns_to_plot = normal_conns
    if len(conns_to_plot) > 500:
        indices = np.random.choice(len(conns_to_plot), 500, replace=False)
        conns_to_plot = [normal_conns[idx] for idx in indices]
    for pre_id, post_id in conns_to_plot:
        pre_pos = neuron_positions[pre_id]
        post_pos = neuron_positions[post_id]
        ax.plot([pre_pos[0], post_pos[0]], [pre_pos[1], post_pos[1]], "gray", alpha=0.12, linewidth=0.45)

    hub_conns_to_plot = hub_conns
    if len(hub_conns_to_plot) > 500:
        indices = np.random.choice(len(hub_conns_to_plot), 500, replace=False)
        hub_conns_to_plot = [hub_conns[idx] for idx in indices]
    for pre_id, post_id in hub_conns_to_plot:
        pre_pos = neuron_positions[pre_id]
        post_pos = neuron_positions[post_id]
        ax.plot(
            [pre_pos[0], post_pos[0]],
            [pre_pos[1], post_pos[1]],
            "red",
            alpha=0.35,
            linewidth=1.0,
            linestyle="--",
        )

    for cluster_idx, neuron_ids in enumerate(cluster_neuron_groups):
        non_hub_ids = [neuron_id for neuron_id in neuron_ids if neuron_id not in hub_set]
        if non_hub_ids:
            positions = neuron_positions[non_hub_ids]
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                s=40,
                c=[cluster_colors[cluster_idx]],
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )

    if hub_neuron_ids:
        hub_positions = neuron_positions[hub_neuron_ids]
        ax.scatter(
            hub_positions[:, 0],
            hub_positions[:, 1],
            s=200,
            marker="*",
            c="red",
            edgecolors="black",
            linewidth=0.8,
            zorder=5,
            label=f"Hub neurons ({len(hub_neuron_ids)})",
        )

    ax.legend(loc="upper right", fontsize=11)
    ax.set_title(
        "Network Connectivity with Hub Neurons\n"
        f"{len(neuron_positions)} neurons, {len(connections)} connections, "
        f"{len(hub_neuron_ids)} hubs, {len(hub_conns)} hub connections",
        fontsize=13,
    )
    ax.set_xlabel("X position", fontsize=12)
    ax.set_ylabel("Y position", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig, ax


def plot_resampled_raster(
    resampled_spikes,
    resampled_times,
    cluster_assignments,
    resampling_frequency,
    burst_onset_times=None,
    title=None,
):
    """Plot the saved resampled spike raster and optional stimulus markers.

    Args:
        resampled_spikes: Binary neuron-by-time spike matrix.
        resampled_times: Time axis corresponding to the resampled spike bins.
        cluster_assignments: Cluster index for each neuron.
        resampling_frequency: Sampling frequency used to build the raster.
        burst_onset_times: Optional stimulation onset times in milliseconds.
        title: Optional figure title. A default title is generated when omitted.

    Returns:
        The created Matplotlib figure.
    """
    sorted_indices = np.argsort(cluster_assignments)
    sorted_spikes = resampled_spikes[sorted_indices]
    sorted_clusters = cluster_assignments[sorted_indices]
    time_sec = resampled_times / 1000.0
    bin_width_ms = (
        float(resampled_times[1] - resampled_times[0])
        if len(resampled_times) > 1
        else 1000.0 / resampling_frequency
    )
    stimulation_enabled = burst_onset_times is not None and len(burst_onset_times) > 0

    fig, ax = plt.subplots(figsize=(16, 8))
    unique_clusters = np.unique(sorted_clusters)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for neuron_idx in range(sorted_spikes.shape[0]):
        spike_times_idx = np.where(sorted_spikes[neuron_idx] == 1)[0]
        if len(spike_times_idx) > 0:
            cluster_id = sorted_clusters[neuron_idx]
            color = cmap[cluster_id % len(cmap)]
            ax.scatter(
                time_sec[spike_times_idx],
                np.full_like(spike_times_idx, neuron_idx, dtype=float),
                c=[color],
                s=1,
                marker="|",
                linewidths=0.5,
            )

    for cluster_id in unique_clusters[:-1]:
        boundary = np.max(np.where(sorted_clusters == cluster_id)[0]) + 0.5
        ax.axhline(y=boundary, color="gray", linewidth=0.3, alpha=0.5)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Neuron (sorted by cluster)", fontsize=12)
    if title is None:
        title = f"Resampled Raster Plot ({resampling_frequency:.1f} Hz, {bin_width_ms:.0f} ms bins)"
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, time_sec[-1] if len(time_sec) > 0 else 0)
    ax.set_ylim(-0.5, sorted_spikes.shape[0] - 0.5)

    if stimulation_enabled:
        for idx, onset in enumerate(burst_onset_times):
            ax.axvline(
                x=onset / 1000.0,
                color="red",
                linewidth=0.8,
                alpha=0.5,
                linestyle="--",
                label="Stimulus" if idx == 0 else "",
            )
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def plot_sag_probe(sag_with_h, sag_without_h):
    """Plot paired h-current sag and rebound probe traces.

    Args:
        sag_with_h: Probe output dictionary generated with the h-current enabled.
        sag_without_h: Probe output dictionary generated with the h-current disabled.

    Returns:
        The created Matplotlib figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(sag_with_h["times"], sag_with_h["voltage"], color="navy", linewidth=1.2, label="With I_h")
    axes[0].plot(
        sag_without_h["times"],
        sag_without_h["voltage"],
        color="darkorange",
        linewidth=1.2,
        linestyle="--",
        label="Without I_h",
    )
    axes[0].axhline(sag_with_h["v_rest"], color="gray", linewidth=0.8, linestyle=":")
    axes[0].axvspan(sag_with_h["step_start_ms"], sag_with_h["step_end_ms"], color="lightsteelblue", alpha=0.2)
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("Hyperpolarizing Step Test: Voltage Sag and Rebound")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(sag_with_h["times"], sag_with_h["h_gate"], color="teal", linewidth=1.2)
    axes[1].axvspan(sag_with_h["step_start_ms"], sag_with_h["step_end_ms"], color="lightsteelblue", alpha=0.2)
    axes[1].set_ylabel("h-gate")
    axes[1].set_title("Slow h-current activation")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(sag_with_h["times"], sag_with_h["i_h"], color="purple", linewidth=1.2, label="I_h")
    axes[2].plot(sag_with_h["times"], sag_with_h["i_ext"], color="black", linewidth=1.0, linestyle="--", label="I_ext")
    axes[2].axvspan(sag_with_h["step_start_ms"], sag_with_h["step_end_ms"], color="lightsteelblue", alpha=0.2)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Current")
    axes[2].set_title("Hyperpolarizing input and inward h-current response")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


def plot_spike_train_analysis_summary(analysis_results):
    """Summarize spike-train analysis metrics in a four-panel diagnostic figure.

    Args:
        analysis_results: Dictionary returned by ``analyze_spike_trains``.

    Returns:
        The created Matplotlib figure.
    """
    mode_title = analysis_results["expected_mode"].replace("_", " ").title()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.hist(analysis_results["firing_rates"], bins=30, edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.mean(analysis_results["firing_rates"]),
        color="r",
        linestyle="--",
        label=f"Mean={np.mean(analysis_results['firing_rates']):.2f} Hz",
    )
    ax1.set_xlabel("Firing Rate (Hz)")
    ax1.set_ylabel("Count")
    ax1.set_title("Firing Rate Distribution")
    ax1.legend()

    ax2 = axes[0, 1]
    if len(analysis_results["all_isis"]) > 0:
        isis_to_plot = analysis_results["all_isis"][analysis_results["all_isis"] < 500]
        ax2.hist(isis_to_plot, bins=50, edgecolor="black", alpha=0.7)
        ax2.axvline(2, color="r", linestyle="--", label="Refractory (2 ms)")
        ax2.set_xlabel("Inter-Spike Interval (ms)")
        ax2.set_ylabel("Count")
        ax2.set_title("ISI Distribution")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Not enough spikes", ha="center", va="center", transform=ax2.transAxes)

    ax3 = axes[1, 0]
    time_bins = np.arange(len(analysis_results["population_rate"])) * 100
    ax3.plot(time_bins / 1000.0, analysis_results["population_rate"], "b-", linewidth=1)
    ax3.fill_between(time_bins / 1000.0, analysis_results["population_rate"], alpha=0.3)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Population Rate (Hz per neuron)")
    ax3.set_title(f"Network Activity Over Time ({mode_title})")
    ax3.axhline(
        np.mean(analysis_results["population_rate"]),
        color="r",
        linestyle="--",
        label=f"Mean={np.mean(analysis_results['population_rate']):.2f} Hz",
    )
    if analysis_results["stimulus_enabled"]:
        for idx, onset in enumerate(analysis_results["burst_onset_times"]):
            ax3.axvline(
                onset / 1000.0,
                color="k",
                linewidth=0.8,
                alpha=0.4,
                linestyle="--",
                label="Stimulus" if idx == 0 else "",
            )
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.bar(range(len(analysis_results["cluster_rates"])), analysis_results["cluster_rates"], edgecolor="black", alpha=0.7)
    ax4.axhline(
        np.mean(analysis_results["cluster_rates"]),
        color="r",
        linestyle="--",
        label=f"Mean={np.mean(analysis_results['cluster_rates']):.2f} Hz",
    )
    ax4.set_xlabel("Cluster ID")
    ax4.set_ylabel("Mean Firing Rate (Hz)")
    ax4.set_title("Firing Rate by Cluster")
    ax4.legend()

    plt.suptitle(f"Spike Train Analysis - {mode_title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_hub_degree_distributions(hub_stats):
    """Compare hub and non-hub degree distributions in histogram form.

    Args:
        hub_stats: Dictionary returned by ``validate_hub_structure``.

    Returns:
        The created Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(
        hub_stats["non_hub_degrees"],
        bins=30,
        alpha=0.6,
        label=f"Non-hub (n={len(hub_stats['non_hub_degrees'])})",
        color="steelblue",
    )
    ax1.hist(
        hub_stats["hub_degrees"],
        bins=15,
        alpha=0.7,
        label=f"Hub (n={len(hub_stats['hub_degrees'])})",
        color="red",
    )
    ax1.axvline(
        np.mean(hub_stats["non_hub_degrees"]),
        color="steelblue",
        linestyle="--",
        linewidth=2,
        label=f"Non-hub mean={np.mean(hub_stats['non_hub_degrees']):.0f}",
    )
    ax1.axvline(
        np.mean(hub_stats["hub_degrees"]),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Hub mean={np.mean(hub_stats['hub_degrees']):.0f}",
    )
    ax1.set_xlabel("Total Degree (in + out)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Degree Distribution: Hub vs Non-Hub", fontsize=13)
    ax1.legend(fontsize=10)

    ax2 = axes[1]
    ax2.hist(hub_stats["non_hub_out_degree"], bins=30, alpha=0.6, label="Non-hub", color="steelblue")
    ax2.hist(hub_stats["hub_out_degree"], bins=15, alpha=0.7, label="Hub", color="red")
    ax2.axvline(np.mean(hub_stats["non_hub_out_degree"]), color="steelblue", linestyle="--", linewidth=2)
    ax2.axvline(np.mean(hub_stats["hub_out_degree"]), color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Out-Degree", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Out-Degree Distribution: Hub vs Non-Hub", fontsize=13)
    ax2.legend(fontsize=10)

    plt.suptitle(
        f"Hub Structure Analysis (degree ratio: {hub_stats['degree_ratio']:.2f}x)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_hub_firing_rate_histogram(rate_groups):
    """Compare hub and non-hub firing-rate distributions.

    Args:
        rate_groups: Dictionary returned by ``compute_hub_firing_rate_groups``.

    Returns:
        The created Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        rate_groups["non_hub_rates"],
        bins=30,
        alpha=0.6,
        label=f"Non-hub (mean={rate_groups['non_hub_mean_rate']:.2f} Hz)",
        color="steelblue",
    )
    ax.hist(
        rate_groups["hub_rates"],
        bins=15,
        alpha=0.7,
        label=f"Hub (mean={rate_groups['hub_mean_rate']:.2f} Hz)",
        color="red",
    )
    ax.set_xlabel("Firing Rate (Hz)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Firing Rate: Hub vs Non-Hub Neurons", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    return fig


# net900-style network-figure palette (shared by the four panel plotters below)
_NET900_EXC = "#3b6fb0"
_NET900_INH = "#c0392b"
_NET900_HIGHLIGHT = "#13b955"
_NET900_INDEG = "#2c7fb8"
_NET900_OUTDEG = "#d95f0e"
_NET900_LENGTH = "#5e3c99"


def plot_spatial_connectivity(neuron_positions, connections, highlight_neuron=None,
                              edge_alpha_exc=0.14, edge_alpha_inh=0.28, figsize=(8, 8)):
    """Plot the spatial network layout with one neuron's full axonal output.

    Renders panel A of the net900-style network figure: every synapse is drawn as
    a distance-spanning line segment (excitatory and inhibitory colored
    separately), all neurons are scattered on top, and a single chosen neuron's
    outgoing axonal projections are highlighted.

    Args:
        neuron_positions: ``(N, 2)`` array of neuron coordinates in mm.
        connections: Connection table whose rows are
            ``[pre_id, post_id, weight, conn_type]`` with ``conn_type`` in
            ``{'exc', 'inh'}``.
        highlight_neuron: Optional neuron id whose outgoing connections are drawn
            in the highlight color. When ``None`` a well-connected, central,
            excitatory neuron is chosen automatically.
        edge_alpha_exc: Opacity of excitatory edge segments.
        edge_alpha_inh: Opacity of inhibitory edge segments.
        figsize: Figure size in inches.

    Returns:
        The created Matplotlib figure.
    """
    pos = np.asarray(neuron_positions, dtype=float)
    connections = np.asarray(connections, dtype=object)
    pre = connections[:, 0].astype(int)
    post = connections[:, 1].astype(int)
    conn_type = connections[:, 3].astype(str)
    N = len(pos)

    # A neuron is inhibitory iff it sources at least one inhibitory connection.
    is_inh = np.zeros(N, dtype=bool)
    is_inh[np.unique(pre[conn_type == "inh"])] = True

    fig, ax = plt.subplots(figsize=figsize)

    exc_mask = conn_type == "exc"
    inh_mask = conn_type == "inh"
    if np.any(exc_mask):
        segments_exc = np.stack([pos[pre[exc_mask]], pos[post[exc_mask]]], axis=1)
        ax.add_collection(LineCollection(segments_exc, colors=_NET900_EXC,
                                         linewidths=0.3, alpha=edge_alpha_exc))
    if np.any(inh_mask):
        segments_inh = np.stack([pos[pre[inh_mask]], pos[post[inh_mask]]], axis=1)
        ax.add_collection(LineCollection(segments_inh, colors=_NET900_INH,
                                         linewidths=0.35, alpha=edge_alpha_inh))

    out_degree = np.bincount(pre, minlength=N)
    if highlight_neuron is None:
        center = (pos.min(0) + pos.max(0)) / 2
        extent = (pos.max(0) - pos.min(0)).max()
        near = np.where((np.linalg.norm(pos - center, axis=1) < 0.15 * extent) & (~is_inh)
                        & (out_degree > np.percentile(out_degree, 75)))[0]
        hub = near[np.argmax(out_degree[near])] if len(near) else int(np.argmax(out_degree))
    else:
        hub = int(highlight_neuron)

    targets = post[pre == hub]
    if len(targets):
        hub_starts = np.repeat(pos[hub][None, :], len(targets), axis=0)
        segments_hub = np.stack([hub_starts, pos[targets]], axis=1)
        ax.add_collection(LineCollection(segments_hub, colors=_NET900_HIGHLIGHT,
                                         linewidths=0.8, alpha=0.95, zorder=4))

    exc_neurons = ~is_inh
    n_exc = int(exc_neurons.sum())
    n_inh = int(is_inh.sum())
    ax.scatter(pos[exc_neurons, 0], pos[exc_neurons, 1], s=5, c=_NET900_EXC,
               alpha=0.85, linewidths=0, label=f"excitatory ({n_exc})")
    ax.scatter(pos[is_inh, 0], pos[is_inh, 1], s=7, c=_NET900_INH,
               alpha=0.9, linewidths=0, label=f"inhibitory ({n_inh})")
    ax.scatter(*pos[hub], s=70, facecolors="none", edgecolors=_NET900_HIGHLIGHT,
               linewidths=1.6, zorder=5)

    ax.set_aspect("equal")
    x_margin = 0.02 * (pos[:, 0].max() - pos[:, 0].min())
    y_margin = 0.02 * (pos[:, 1].max() - pos[:, 1].min())
    ax.set_xlim(pos[:, 0].min() - x_margin, pos[:, 0].max() + x_margin)
    ax.set_ylim(pos[:, 1].min() - y_margin, pos[:, 1].max() + y_margin)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.02, "green = one neuron's full axonal output",
            color=_NET900_HIGHLIGHT, fontweight="bold", transform=ax.transAxes)
    plt.tight_layout()
    return fig


def plot_position_sorted_adjacency(neuron_positions, connections, figsize=(8, 8)):
    """Plot the x-position-sorted adjacency scatter (net900 panel B).

    Neurons are ranked by x coordinate and each synapse is plotted at
    ``(rank[pre], rank[post])``; distance-dependent wiring appears as a band
    around the diagonal.

    Args:
        neuron_positions: ``(N, 2)`` array of neuron coordinates in mm.
        connections: Connection table whose rows are
            ``[pre_id, post_id, weight, conn_type]`` with ``conn_type`` in
            ``{'exc', 'inh'}``.
        figsize: Figure size in inches.

    Returns:
        The created Matplotlib figure.
    """
    pos = np.asarray(neuron_positions, dtype=float)
    connections = np.asarray(connections, dtype=object)
    pre = connections[:, 0].astype(int)
    post = connections[:, 1].astype(int)
    conn_type = connections[:, 3].astype(str)
    N = len(pos)

    order = np.argsort(pos[:, 0])
    rank = np.empty(N, dtype=int)
    rank[order] = np.arange(N)

    fig, ax = plt.subplots(figsize=figsize)
    exc_mask = conn_type == "exc"
    inh_mask = conn_type == "inh"
    ax.scatter(rank[pre[exc_mask]], rank[post[exc_mask]], s=1.4, c=_NET900_EXC,
               alpha=0.5, linewidths=0)
    ax.scatter(rank[pre[inh_mask]], rank[post[inh_mask]], s=1.8, c=_NET900_INH,
               alpha=0.65, linewidths=0)

    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("presynaptic neuron (sorted by x position)")
    ax.set_ylabel("postsynaptic neuron")
    plt.tight_layout()
    return fig


def plot_degree_distributions(connections, n_neurons=None, figsize=(8, 6)):
    """Plot in- and out-degree synaptic distributions (net900 panel C).

    Args:
        connections: Connection table whose rows are
            ``[pre_id, post_id, weight, conn_type]``.
        n_neurons: Total neuron count. When ``None`` it is inferred as one past
            the largest neuron id appearing in the connection table.
        figsize: Figure size in inches.

    Returns:
        The created Matplotlib figure.
    """
    connections = np.asarray(connections, dtype=object)
    pre = connections[:, 0].astype(int)
    post = connections[:, 1].astype(int)
    N = n_neurons or int(max(pre.max(), post.max())) + 1

    indeg = np.bincount(post, minlength=N)
    outdeg = np.bincount(pre, minlength=N)
    bins = np.arange(0, max(indeg.max(), outdeg.max()) + 2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(indeg, bins=bins, alpha=0.6, color=_NET900_INDEG,
            label=f"in-degree (mean {indeg.mean():.1f})")
    ax.hist(outdeg, bins=bins, alpha=0.55, color=_NET900_OUTDEG,
            label=f"out-degree (mean {outdeg.mean():.1f})")
    ax.set_xlabel("synaptic degree")
    ax.set_ylabel("number of neurons")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_connection_length_distribution(neuron_positions, connections, bins=60, figsize=(8, 6)):
    """Plot the distribution of Euclidean connection lengths (net900 panel D).

    Args:
        neuron_positions: ``(N, 2)`` array of neuron coordinates in mm.
        connections: Connection table whose rows are
            ``[pre_id, post_id, weight, conn_type]``.
        bins: Number of histogram bins.
        figsize: Figure size in inches.

    Returns:
        The created Matplotlib figure.
    """
    pos = np.asarray(neuron_positions, dtype=float)
    connections = np.asarray(connections, dtype=object)
    pre = connections[:, 0].astype(int)
    post = connections[:, 1].astype(int)

    lengths = np.linalg.norm(pos[pre] - pos[post], axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(lengths, bins=bins, color=_NET900_LENGTH, alpha=0.8)
    ax.axvline(lengths.mean(), color="k", ls="--", lw=1.2,
               label=f"mean {lengths.mean():.2f} mm")
    ax.axvline(np.median(lengths), color="#888", ls=":", lw=1.2,
               label=f"median {np.median(lengths):.2f} mm")
    ax.set_xlabel("connection length (mm)")
    ax.set_ylabel("number of synapses")
    ax.legend()
    plt.tight_layout()
    return fig