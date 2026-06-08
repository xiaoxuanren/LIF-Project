# Hub Neurons Implementation Guide

## Context

This document provides instructions for adding **hub neurons** to an existing NEURON-based clustered neural network simulation. The simulation uses Hodgkin-Huxley neurons organized in spatial clusters with lognormal weight distributions, presynaptic input populations, and calcium imaging conversion.

Use this file as a reference when working with Claude Code in VS Code on the simulation codebase.

---

## What Are Hub Neurons?

Hub neurons are highly connected neurons that serve as critical nodes for information integration and routing across the network. They have:

- **High degree**: 3–10× the mean connectivity of normal neurons
- **Long-range connections**: Bridge multiple clusters (connector hubs)
- **Stronger synapses**: Scaled-up synaptic weights
- **Disproportionate influence**: Drive burst propagation and cross-cluster synchronization

In biological cortical networks, hubs make up ~5–15% of neurons but account for a large share of total connectivity and burst initiation.

---

## Parameters to Add

Add these to your network creation function or parameter class:

```python
# Hub neuron parameters
hub_fraction = 0.1            # Fraction of neurons per cluster designated as hubs (5-15%)
hub_between_prob = 0.4        # Inter-cluster connection probability for hubs (vs ~0.15 baseline)
hub_weight_scale = 1.5        # Multiplicative scaling of hub synaptic weights
hub_reciprocal_factor = 2.0   # Additional reciprocal connection boost for hub-hub pairs
```

---

## Implementation Steps

### Step 1: Designate Hub Neurons

After creating clusters and assigning neurons, select hub neurons within each cluster:

```python
import numpy as np

def designate_hub_neurons(neuron_clusters, hub_fraction=0.1):
    """
    Designate a fraction of neurons in each cluster as hub neurons.
    
    Parameters
    ----------
    neuron_clusters : list of lists
        Each sublist contains NEURON section objects for one cluster.
    hub_fraction : float
        Fraction of neurons per cluster to designate as hubs.
    
    Returns
    -------
    hub_neurons : list of tuples
        Each tuple is (cluster_index, local_neuron_index).
    hub_global_indices : list of int
        Global neuron indices of all hub neurons.
    """
    hub_neurons = []
    hub_global_indices = []
    offset = 0

    for cluster_idx, cluster in enumerate(neuron_clusters):
        n_hubs = max(1, int(len(cluster) * hub_fraction))
        hub_local_indices = np.random.choice(len(cluster), n_hubs, replace=False)

        for local_idx in hub_local_indices:
            hub_neurons.append((cluster_idx, int(local_idx)))
            hub_global_indices.append(offset + int(local_idx))

        offset += len(cluster)

    return hub_neurons, hub_global_indices
```

### Step 2: Add Hub Inter-Cluster Connections

After the main connectivity loop in `create_clustered_network`, add extra between-cluster connections for hub neurons:

```python
def add_hub_connections(neuron_clusters, hub_neurons, connections, synapses, netcons,
                        inhibitory_indices, hub_between_prob=0.4, hub_weight_scale=1.5,
                        hub_reciprocal_factor=2.0, weight_params=None):
    """
    Add elevated inter-cluster connections for hub neurons.
    
    This function should be called AFTER the base network connectivity is created.
    It adds additional between-cluster connections for designated hub neurons
    with higher probability and scaled weights.
    
    Parameters
    ----------
    neuron_clusters : list of lists
        Cluster structure from create_clustered_network.
    hub_neurons : list of tuples
        (cluster_idx, local_idx) for each hub neuron.
    connections : list of dicts
        Existing connection list to append to.
    synapses : list
        Existing synapse list to append to.
    netcons : list
        Existing NetCon list to append to.
    inhibitory_indices : dict or list of lists
        Which local indices in each cluster are inhibitory.
    hub_between_prob : float
        Connection probability from hub to neurons in other clusters.
    hub_weight_scale : float
        Multiplicative weight scaling for hub connections.
    hub_reciprocal_factor : float
        Additional factor for reciprocal hub-hub connections.
    weight_params : dict, optional
        Parameters for lognormal weight generation.
        Expected keys: 'exc_mean', 'exc_std', 'inh_mean', 'inh_std'
        Defaults provided if None.
    
    Returns
    -------
    hub_connections : list of dicts
        Only the newly added hub connections.
    """
    from neuron import h

    if weight_params is None:
        weight_params = {
            'exc_mean': 0.003,
            'exc_std': 0.001,
            'inh_mean': 0.008,
            'inh_std': 0.002,
        }

    # Build a set of hub global indices for reciprocal detection
    hub_global_set = set()
    offset = 0
    cluster_offsets = []
    for cluster in neuron_clusters:
        cluster_offsets.append(offset)
        offset += len(cluster)
    for (ci, li) in hub_neurons:
        hub_global_set.add(cluster_offsets[ci] + li)

    hub_connections = []

    for (hub_cluster_idx, hub_local_idx) in hub_neurons:
        hub_section = neuron_clusters[hub_cluster_idx][hub_local_idx]
        hub_global = cluster_offsets[hub_cluster_idx] + hub_local_idx

        # Determine if this hub is inhibitory
        is_hub_inhib = hub_local_idx in inhibitory_indices[hub_cluster_idx]

        for target_cluster_idx, target_cluster in enumerate(neuron_clusters):
            if target_cluster_idx == hub_cluster_idx:
                continue  # Skip own cluster (already connected by base network)

            for target_local_idx, target_section in enumerate(target_cluster):
                target_global = cluster_offsets[target_cluster_idx] + target_local_idx

                # Determine connection probability
                prob = hub_between_prob

                # Boost for hub-to-hub reciprocal connections
                if target_global in hub_global_set:
                    prob = min(1.0, prob * hub_reciprocal_factor)

                if np.random.random() < prob:
                    # Create synapse on target
                    syn = h.ExpSyn(target_section(0.5))
                    if is_hub_inhib:
                        syn.e = -75.0
                        syn.tau = 5.0
                        base_weight = np.random.lognormal(
                            np.log(weight_params['inh_mean']),
                            weight_params['inh_std']
                        )
                        weight = -abs(base_weight * hub_weight_scale)
                    else:
                        syn.e = 0.0
                        syn.tau = 2.0
                        base_weight = np.random.lognormal(
                            np.log(weight_params['exc_mean']),
                            weight_params['exc_std']
                        )
                        weight = abs(base_weight * hub_weight_scale)

                    # Create NetCon
                    nc = h.NetCon(hub_section(0.5)._ref_v, syn, sec=hub_section)
                    nc.weight[0] = abs(weight)
                    nc.delay = 1.0
                    nc.threshold = -20

                    synapses.append(syn)
                    netcons.append(nc)

                    conn_info = {
                        'pre_cluster': hub_cluster_idx,
                        'post_cluster': target_cluster_idx,
                        'pre_local_idx': hub_local_idx,
                        'post_local_idx': target_local_idx,
                        'pre_global_idx': hub_global,
                        'post_global_idx': target_global,
                        'weight': weight,
                        'is_hub_connection': True,
                        'is_inhibitory': is_hub_inhib,
                    }
                    connections.append(conn_info)
                    hub_connections.append(conn_info)

    return hub_connections
```

### Step 3: Integrate Into `create_clustered_network`

Where your existing function finishes building base connections, add:

```python
# --- Inside create_clustered_network, after base connectivity loop ---

# Designate hubs
hub_neurons, hub_global_indices = designate_hub_neurons(neuron_clusters, hub_fraction)

# Add hub connections
hub_connections = add_hub_connections(
    neuron_clusters=neuron_clusters,
    hub_neurons=hub_neurons,
    connections=connections,        # your existing connections list
    synapses=all_synapses,          # your existing synapses list
    netcons=all_netcons,            # your existing netcons list
    inhibitory_indices=inhibitory_indices,
    hub_between_prob=hub_between_prob,
    hub_weight_scale=hub_weight_scale,
    hub_reciprocal_factor=hub_reciprocal_factor,
    weight_params=weight_params,
)

# Store hub info in your network data dictionary
network_data['hub_neurons'] = hub_neurons
network_data['hub_global_indices'] = hub_global_indices
network_data['n_hub_connections'] = len(hub_connections)
```

---

## Visualization: Highlight Hubs in Network Plot

Update your `plot_network_connectivity` function:

```python
def plot_hub_network(neuron_positions, connections, hub_neurons, neuron_clusters,
                     cluster_colors=None, figsize=(12, 10)):
    """
    Plot network connectivity with hub neurons highlighted.
    
    Hub neurons are shown as large red stars.
    Hub connections are shown as red dashed lines.
    Normal connections are shown as usual.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    n_clusters = len(neuron_clusters)
    if cluster_colors is None:
        cmap = plt.cm.tab10
        cluster_colors = [cmap(i / n_clusters) for i in range(n_clusters)]

    # Plot normal connections (thin, gray)
    for conn in connections:
        if conn.get('is_hub_connection', False):
            continue
        pre_pos = neuron_positions[conn['pre_cluster']][conn['pre_local_idx']]
        post_pos = neuron_positions[conn['post_cluster']][conn['post_local_idx']]
        ax.plot([pre_pos[0], post_pos[0]], [pre_pos[1], post_pos[1]],
                'gray', alpha=0.1, linewidth=0.3)

    # Plot hub connections (red, dashed)
    for conn in connections:
        if not conn.get('is_hub_connection', False):
            continue
        pre_pos = neuron_positions[conn['pre_cluster']][conn['pre_local_idx']]
        post_pos = neuron_positions[conn['post_cluster']][conn['post_local_idx']]
        ax.plot([pre_pos[0], post_pos[0]], [pre_pos[1], post_pos[1]],
                'red', alpha=0.15, linewidth=0.5, linestyle='--')

    # Plot normal neurons
    hub_set = set(hub_neurons)
    for cluster_idx in range(n_clusters):
        for local_idx in range(len(neuron_clusters[cluster_idx])):
            if (cluster_idx, local_idx) in hub_set:
                continue
            pos = neuron_positions[cluster_idx][local_idx]
            ax.scatter(pos[0], pos[1], s=40, c=[cluster_colors[cluster_idx]],
                       edgecolors='black', linewidth=0.5, zorder=3)

    # Plot hub neurons (large red stars)
    for i, (cluster_idx, local_idx) in enumerate(hub_neurons):
        pos = neuron_positions[cluster_idx][local_idx]
        ax.scatter(pos[0], pos[1], s=200, marker='*', c='red',
                   edgecolors='black', linewidth=0.8, zorder=5,
                   label='Hub neuron' if i == 0 else '')

    ax.legend(loc='upper right')
    ax.set_title('Network Connectivity with Hub Neurons')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    plt.tight_layout()
    return fig, ax
```

---

## Validation: Verify Hub Structure

```python
def validate_hub_structure(connections, hub_global_indices, n_neurons):
    """
    Validate that hub neurons have expected elevated connectivity.
    Prints degree statistics and rich-club coefficient.
    """
    import networkx as nx
    import numpy as np

    G = nx.DiGraph()
    G.add_nodes_from(range(n_neurons))

    for conn in connections:
        G.add_edge(conn['pre_global_idx'], conn['post_global_idx'],
                   weight=conn.get('weight', 1.0))

    degrees = dict(G.degree())
    hub_set = set(hub_global_indices)

    hub_degrees = [degrees.get(i, 0) for i in hub_global_indices]
    non_hub_degrees = [degrees.get(i, 0) for i in range(n_neurons) if i not in hub_set]

    print("=== Hub Neuron Validation ===")
    print(f"Total neurons: {n_neurons}")
    print(f"Hub neurons: {len(hub_global_indices)} ({100*len(hub_global_indices)/n_neurons:.1f}%)")
    print(f"Hub mean degree: {np.mean(hub_degrees):.1f} ± {np.std(hub_degrees):.1f}")
    print(f"Non-hub mean degree: {np.mean(non_hub_degrees):.1f} ± {np.std(non_hub_degrees):.1f}")
    print(f"Degree ratio (hub / non-hub): {np.mean(hub_degrees)/np.mean(non_hub_degrees):.2f}x")

    # Betweenness centrality
    bc = nx.betweenness_centrality(G)
    hub_bc = [bc[i] for i in hub_global_indices]
    non_hub_bc = [bc[i] for i in range(n_neurons) if i not in hub_set]
    print(f"Hub mean betweenness: {np.mean(hub_bc):.4f}")
    print(f"Non-hub mean betweenness: {np.mean(non_hub_bc):.4f}")

    # Rich-club coefficient
    try:
        rc = nx.rich_club_coefficient(G.to_undirected(), normalized=False)
        print(f"Rich-club coefficients (top 5 degree thresholds):")
        sorted_keys = sorted(rc.keys())[-5:]
        for k in sorted_keys:
            print(f"  degree >= {k}: {rc[k]:.3f}")
    except Exception as e:
        print(f"Rich-club computation skipped: {e}")

    print("=============================")
    return {
        'hub_degrees': hub_degrees,
        'non_hub_degrees': non_hub_degrees,
        'hub_betweenness': hub_bc,
        'non_hub_betweenness': non_hub_bc,
    }
```

---

## Saving Hub Data

Update your `save_simulation_data_colab` function to include hub info:

```python
# Add these to your np.savez call:
np.savez(filepath,
    # ... existing fields ...
    hub_neurons=np.array(hub_neurons),              # (n_hubs, 2) array of (cluster, local_idx)
    hub_global_indices=np.array(hub_global_indices), # (n_hubs,) array
)

# Add to your JSON parameters:
params['hub_fraction'] = hub_fraction
params['hub_between_prob'] = hub_between_prob
params['hub_weight_scale'] = hub_weight_scale
params['hub_reciprocal_factor'] = hub_reciprocal_factor
params['n_hub_neurons'] = len(hub_neurons)
params['n_hub_connections'] = len(hub_connections)
```

---

## Quick Reference: Parameter Tuning

| Parameter | Range | Effect |
|---|---|---|
| `hub_fraction` | 0.05 – 0.15 | More hubs → more synchronized bursting across clusters |
| `hub_between_prob` | 0.2 – 0.5 | Higher → stronger inter-cluster coupling |
| `hub_weight_scale` | 1.2 – 2.0 | Amplifies hub influence on postsynaptic targets |
| `hub_reciprocal_factor` | 1.5 – 3.0 | Strengthens hub-to-hub reciprocal connections |

### Expected Outcomes

- Hub degree should be **3–5×** non-hub degree
- Hub betweenness centrality should be significantly higher
- Network should show more cross-cluster burst propagation
- Removing hubs (lesion experiment) should reduce global synchrony

---

## Usage with Claude Code

When working in Claude Code in VS Code, you can reference this file:

```
@hub_neurons_instructions.md Add hub neurons to my create_clustered_network function
```

Or for specific sections:

```
@hub_neurons_instructions.md Implement the validation function for hub structure
```

Claude Code will read this file and apply the instructions to your actual codebase.
