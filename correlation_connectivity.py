"""
Correlation-based connectivity maps from spike trains.

Builds a lagged correlation score matrix from binned spike trains, then saves:
  - lagged correlation heatmap
  - lagged correlation connectivity map
  - true connectivity map
  - learned-LIF connectivity map (when available)
  - side-by-side comparison figure

Usage:
    python correlation_connectivity.py [--session path] [--bin-size-ms 50]
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def spike_times_to_binned_matrix(spike_times, duration_ms, bin_size_ms, mode='binary'):
    """Convert per-neuron spike times to a binned spike matrix."""
    n_neurons = len(spike_times)
    n_bins = max(1, int(np.ceil(float(duration_ms) / float(bin_size_ms))))
    matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)

    for neuron_idx in range(n_neurons):
        neuron_spikes = np.asarray(spike_times[neuron_idx], dtype=np.float64)
        if neuron_spikes.size == 0:
            continue
        bin_indices = np.floor(neuron_spikes / float(bin_size_ms)).astype(np.int64)
        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        if not np.any(valid):
            continue
        counts = np.bincount(bin_indices[valid], minlength=n_bins).astype(np.float32)
        matrix[neuron_idx] = counts

    if mode == 'binary':
        np.minimum(matrix, 1.0, out=matrix)
    return matrix


def load_network_data(session_dir):
    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f"No network file found in {session_dir}")
    return np.load(net_files[0], allow_pickle=True)


def load_all_recordings(session_dir, bin_size_ms, mode='binary', use_combined=False):
    """Load a session and return a concatenated binned spike matrix."""
    net_data = load_network_data(session_dir)
    source_notes = []

    if use_combined:
        combined_path = os.path.join(session_dir, 'recording_combined.npz')
        if not os.path.exists(combined_path):
            raise FileNotFoundError(
                f"No recording_combined.npz found in {session_dir}; run without --use-combined-recording"
            )
        data = np.load(combined_path, allow_pickle=True)
        duration_ms = float(data['duration_ms'])
        spike_matrix = spike_times_to_binned_matrix(
            data['spike_times'], duration_ms, bin_size_ms, mode=mode,
        )
        source_notes.append('combined raw spike_times')
        return {
            'spike_matrix': spike_matrix,
            'total_duration_ms': duration_ms,
            'n_recordings': 1,
            'connections': net_data['connections'],
            'neuron_positions': net_data['neuron_positions'],
            'cluster_assignments': net_data['cluster_assignments']
            if 'cluster_assignments' in net_data else None,
            'source_note': ', '.join(source_notes),
        }

    rec_files = sorted(glob.glob(os.path.join(session_dir, 'recording[0-9][0-9][0-9].npz')))
    if not rec_files:
        raise FileNotFoundError(f"No recordings found in {session_dir}")

    all_matrices = []
    total_duration_ms = 0.0
    used_saved_resample = True

    for rec_file in rec_files:
        data = np.load(rec_file, allow_pickle=True)
        duration_ms = float(data['duration'])
        if (
            mode == 'binary'
            and 'resampled_spikes' in data
            and 'resampling_interval_ms' in data
            and np.isclose(float(data['resampling_interval_ms']), float(bin_size_ms))
        ):
            spike_matrix = data['resampled_spikes'].astype(np.float32)
        else:
            spike_matrix = spike_times_to_binned_matrix(
                data['spike_times'], duration_ms, bin_size_ms, mode=mode,
            )
            used_saved_resample = False
        all_matrices.append(spike_matrix)
        total_duration_ms += duration_ms

    if used_saved_resample:
        source_notes.append('saved resampled_spikes')
    else:
        source_notes.append('re-binned raw spike_times')

    return {
        'spike_matrix': np.concatenate(all_matrices, axis=1),
        'total_duration_ms': total_duration_ms,
        'n_recordings': len(rec_files),
        'connections': net_data['connections'],
        'neuron_positions': net_data['neuron_positions'],
        'cluster_assignments': net_data['cluster_assignments']
        if 'cluster_assignments' in net_data else None,
        'source_note': ', '.join(source_notes),
    }


def normalize_binned_traces(spike_matrix):
    """Center and standardize each neuron's binned spike train."""
    data = spike_matrix.astype(np.float32, copy=False)
    centered = data - data.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True)
    valid = std[:, 0] > 0

    normalized = np.zeros_like(centered, dtype=np.float32)
    normalized[valid] = centered[valid] / std[valid]
    return normalized


def compute_zero_lag_correlation_matrix(spike_matrix):
    """Compute a zero-lag Pearson correlation matrix."""
    normalized = normalize_binned_traces(spike_matrix)
    n_neurons, n_bins = normalized.shape
    if n_bins < 2:
        return np.zeros((n_neurons, n_neurons), dtype=np.float32)

    corr = (normalized @ normalized.T) / float(n_bins)
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr


def compute_lagged_correlation_matrix(spike_matrix, max_lag_bins=1):
    """
    Compute a directed lagged correlation score matrix.

    Entry [i, j] is the strongest signed correlation where neuron i leads neuron j
    by a positive lag in [1, max_lag_bins]. If max_lag_bins <= 0, this falls back
    to zero-lag correlation.
    """
    normalized = normalize_binned_traces(spike_matrix)
    n_neurons, n_bins = normalized.shape
    zero_lag = compute_zero_lag_correlation_matrix(spike_matrix)

    if n_bins < 2 or max_lag_bins <= 0:
        best_lags = np.zeros((n_neurons, n_neurons), dtype=np.int16)
        return zero_lag, best_lags, zero_lag

    best_scores = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    best_lags = np.zeros((n_neurons, n_neurons), dtype=np.int16)
    lag_limit = min(int(max_lag_bins), n_bins - 1)

    for lag in range(1, lag_limit + 1):
        lead = normalized[:, :-lag]
        follow = normalized[:, lag:]
        lag_scores = (lead @ follow.T) / float(lead.shape[1])
        lag_scores = np.clip(lag_scores, -1.0, 1.0)
        lag_scores = np.nan_to_num(lag_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        np.fill_diagonal(lag_scores, 0.0)

        update = np.abs(lag_scores) > np.abs(best_scores)
        best_scores[update] = lag_scores[update]
        best_lags[update] = lag

    np.fill_diagonal(best_scores, 0.0)
    np.fill_diagonal(best_lags, 0)
    return best_scores, best_lags, zero_lag


def build_true_directed_weight_matrix(connections, n_neurons):
    """Build a directed source-to-target weight matrix from the saved connections."""
    weight_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    for conn in connections:
        pre_id = int(conn[0])
        post_id = int(conn[1])
        weight_matrix[pre_id, post_id] = float(conn[2])
    np.fill_diagonal(weight_matrix, 0.0)
    return weight_matrix


def directed_to_undirected_score_matrix(directed_matrix):
    """Collapse a directed score matrix to a symmetric undirected matrix."""
    directed = np.asarray(directed_matrix, dtype=np.float32)
    if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
        raise ValueError('directed_matrix must be square')

    n_neurons = directed.shape[0]
    upper_i, upper_j = np.triu_indices(n_neurons, k=1)
    forward_scores = directed[upper_i, upper_j]
    reverse_scores = directed[upper_j, upper_i]
    use_forward = np.abs(forward_scores) >= np.abs(reverse_scores)
    chosen_scores = np.where(use_forward, forward_scores, reverse_scores).astype(np.float32)

    undirected = np.zeros_like(directed, dtype=np.float32)
    undirected[upper_i, upper_j] = chosen_scores
    undirected[upper_j, upper_i] = chosen_scores
    np.fill_diagonal(undirected, 0.0)
    return undirected


def build_true_undirected_matrix(true_directed_matrix):
    """Convert a directed ground-truth matrix into an undirected adjacency matrix."""
    true_matrix = ((np.abs(true_directed_matrix) > 0) |
                   (np.abs(true_directed_matrix.T) > 0)).astype(np.int32)
    np.fill_diagonal(true_matrix, 0)
    return true_matrix


def select_undirected_edges(score_matrix, positions, true_matrix=None, top_k=500, threshold=None):
    """Select undirected edges by absolute score from the upper triangle."""
    n_neurons = score_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n_neurons, k=1)
    score_values = score_matrix[upper_i, upper_j]
    abs_values = np.abs(score_values)
    distances = np.linalg.norm(positions[upper_i] - positions[upper_j], axis=1)

    if threshold is not None:
        keep = abs_values >= float(threshold)
    else:
        keep = np.ones_like(abs_values, dtype=bool)

    candidate_indices = np.flatnonzero(keep)
    if top_k is not None:
        if top_k > 0 and len(candidate_indices) > top_k:
            ranked = np.argsort(abs_values[candidate_indices])[::-1][:top_k]
            candidate_indices = candidate_indices[ranked]
        elif top_k == 0:
            candidate_indices = np.array([], dtype=np.int64)

    if len(candidate_indices) > 0:
        order = np.argsort(abs_values[candidate_indices])[::-1]
        candidate_indices = candidate_indices[order]

    edge_indices = np.column_stack((upper_i[candidate_indices], upper_j[candidate_indices]))
    edge_scores = score_values[candidate_indices].astype(np.float32)
    edge_abs_scores = abs_values[candidate_indices].astype(np.float32)
    edge_distances = distances[candidate_indices].astype(np.float32)
    edge_true = None
    if true_matrix is not None:
        edge_true = true_matrix[upper_i[candidate_indices], upper_j[candidate_indices]].astype(np.int32)

    adjacency = np.zeros_like(score_matrix, dtype=np.int8)
    if len(edge_indices) > 0:
        adjacency[edge_indices[:, 0], edge_indices[:, 1]] = 1
        adjacency[edge_indices[:, 1], edge_indices[:, 0]] = 1

    score_cutoff = float(edge_abs_scores.min()) if len(edge_abs_scores) > 0 else 0.0
    return {
        'edge_indices': edge_indices.astype(np.int32),
        'edge_scores': edge_scores,
        'edge_abs_scores': edge_abs_scores,
        'edge_distances': edge_distances,
        'edge_true': edge_true,
        'adjacency': adjacency,
        'score_cutoff': score_cutoff,
        'n_candidates': int(len(abs_values)),
    }


def validate_edge_mapping(score_matrix, positions, selection, atol=1e-7):
    """Validate that selected edges map directly back to score-matrix cell indices."""
    score_matrix = np.asarray(score_matrix)
    positions = np.asarray(positions)

    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError('score_matrix must be square')
    if positions.shape[0] != score_matrix.shape[0]:
        raise ValueError('positions length must match score_matrix size')
    if not np.allclose(score_matrix, score_matrix.T, atol=atol, rtol=0.0):
        raise ValueError('score_matrix must be symmetric for undirected edge selection')

    edge_indices = np.asarray(selection['edge_indices'])
    edge_scores = np.asarray(selection['edge_scores'])
    if edge_indices.size == 0:
        return

    if edge_indices.ndim != 2 or edge_indices.shape[1] != 2:
        raise ValueError('edge_indices must have shape [n_edges, 2]')
    if edge_indices.min() < 0 or edge_indices.max() >= score_matrix.shape[0]:
        raise ValueError('edge indices fall outside the available cell index range')

    looked_up_scores = score_matrix[edge_indices[:, 0], edge_indices[:, 1]]
    if not np.allclose(looked_up_scores, edge_scores, atol=atol, rtol=0.0):
        raise ValueError('edge_scores do not match score_matrix lookup at the selected cell indices')


def evaluate_score_matrix(score_matrix, true_matrix):
    """Evaluate an undirected score matrix against undirected ground truth."""
    upper_i, upper_j = np.triu_indices(score_matrix.shape[0], k=1)
    scores = np.abs(score_matrix[upper_i, upper_j])
    labels = true_matrix[upper_i, upper_j].astype(np.int32)

    results = {
        'n_true_edges': int(labels.sum()),
        'n_pairs': int(len(labels)),
    }
    if len(np.unique(labels)) > 1:
        results['auc'] = float(roc_auc_score(labels, scores))
        results['ap'] = float(average_precision_score(labels, scores))
    else:
        results['auc'] = 0.0
        results['ap'] = 0.0
    return results


def compute_plot_order(cluster_assignments, n_neurons):
    """Order neurons by cluster when assignments are available."""
    if cluster_assignments is None:
        return np.arange(n_neurons)
    return np.argsort(np.asarray(cluster_assignments))


def plot_heatmap(score_matrix, cluster_assignments, title, output_path, colorbar_label):
    """Save a sorted heatmap for a symmetric score matrix."""
    n_neurons = score_matrix.shape[0]
    order = compute_plot_order(cluster_assignments, n_neurons)
    sorted_matrix = score_matrix[order][:, order]

    upper_i, upper_j = np.triu_indices(n_neurons, k=1)
    abs_vals = np.abs(score_matrix[upper_i, upper_j])
    vmax = float(np.percentile(abs_vals, 99)) if len(abs_vals) > 0 else 1.0
    vmax = max(vmax, 1e-3)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sorted_matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Neuron (cluster-sorted)')
    ax.set_ylabel('Neuron (cluster-sorted)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    if cluster_assignments is not None:
        sorted_clusters = np.asarray(cluster_assignments)[order]
        boundaries = np.where(np.diff(sorted_clusters) != 0)[0] + 0.5
        for boundary in boundaries:
            ax.axhline(boundary, color='black', linewidth=0.4, alpha=0.4)
            ax.axvline(boundary, color='black', linewidth=0.4, alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def draw_connectivity_map(ax, positions, selection, title, summary_lines):
    """Draw a connectivity map on an existing axis."""
    edge_indices = selection['edge_indices']
    edge_scores = selection['edge_scores']

    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=14, c='white', edgecolors='black', linewidths=0.45, zorder=3,
    )

    max_abs_score = float(np.max(np.abs(edge_scores))) if len(edge_scores) > 0 else 1.0
    max_abs_score = max(max_abs_score, 1e-6)
    draw_order = np.argsort(np.abs(edge_scores))
    for edge_idx in draw_order:
        node_i, node_j = edge_indices[edge_idx]
        score = float(edge_scores[edge_idx])
        color = 'crimson' if score >= 0 else 'royalblue'
        strength = abs(score) / max_abs_score
        linewidth = 0.8 + 2.4 * strength
        alpha = 0.35 + 0.55 * strength
        ax.plot(
            [positions[node_i, 0], positions[node_j, 0]],
            [positions[node_i, 1], positions[node_j, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle='round',
            zorder=2,
        )

    ax.text(
        0.02, 0.98, '\n'.join(summary_lines),
        transform=ax.transAxes,
        va='top', ha='left', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )
    ax.set_title(title)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_aspect('equal')


def plot_connectivity_map(positions, selection, title, summary_lines, output_path):
    """Save a single connectivity map figure."""
    fig, ax = plt.subplots(figsize=(11, 9))
    draw_connectivity_map(ax, positions, selection, title, summary_lines)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_maps(positions, map_specs, title, output_path):
    """Save a side-by-side comparison figure for multiple map types."""
    n_panels = len(map_specs)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.0 * n_panels, 6.2))
    if n_panels == 1:
        axes = [axes]

    for ax, spec in zip(axes, map_specs):
        draw_connectivity_map(
            ax,
            positions,
            spec['selection'],
            spec['title'],
            spec['summary_lines'],
        )

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_edge_csv(output_path, selection):
    """Save selected edges to CSV."""
    edge_true = selection['edge_true']
    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'cell_i_index', 'cell_j_index', 'score', 'abs_score',
            'distance', 'true_undirected_edge',
        ])
        for edge_idx, (node_i, node_j) in enumerate(selection['edge_indices']):
            writer.writerow([
                int(node_i),
                int(node_j),
                float(selection['edge_scores'][edge_idx]),
                float(selection['edge_abs_scores'][edge_idx]),
                float(selection['edge_distances'][edge_idx]),
                int(edge_true[edge_idx]) if edge_true is not None else '',
            ])


def resolve_learned_lif_connectivity_path(session_name, explicit_path=None):
    """Resolve the learned-LIF connectivity NPZ file for a session."""
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"learned-LIF connectivity file not found: {explicit_path}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'learned_lif_outputs')
    exact_path = os.path.join(output_dir, f'connectivity_{session_name}.npz')
    if os.path.exists(exact_path):
        return exact_path

    candidates = glob.glob(os.path.join(output_dir, f'connectivity_{session_name}*.npz'))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_learned_lif_connectivity(session_name, explicit_path=None):
    """Load a learned-LIF connectivity result and convert it to an undirected score map."""
    resolved_path = resolve_learned_lif_connectivity_path(session_name, explicit_path=explicit_path)
    if resolved_path is None:
        return None

    data = np.load(resolved_path, allow_pickle=True)
    if 'connectivity_matrix' not in data:
        raise KeyError(f"connectivity_matrix missing from {resolved_path}")

    directed_post_pre = data['connectivity_matrix'].astype(np.float32)
    directed_src_dst = directed_post_pre.T
    undirected_scores = directed_to_undirected_score_matrix(directed_src_dst)
    threshold = float(data['threshold']) if 'threshold' in data else None

    return {
        'path': resolved_path,
        'threshold': threshold,
        'directed_matrix': directed_src_dst,
        'undirected_matrix': undirected_scores,
    }


def precision_from_selection(selection):
    """Compute map precision against true undirected edges for a selected edge set."""
    edge_true = selection['edge_true']
    if edge_true is None or len(edge_true) == 0:
        return None
    return float(np.mean(edge_true))


def summarize_selection(selection, score_label, extra_lines=None):
    """Build summary lines for a selected edge set."""
    lines = [f'Edges shown: {len(selection["edge_indices"])}']
    lines.append(f'Cutoff {score_label}: {selection["score_cutoff"]:.4f}')
    precision = precision_from_selection(selection)
    if precision is not None:
        lines.append(f'Map precision vs truth: {precision:.4f}')
    if extra_lines:
        lines.extend(extra_lines)
    return lines


def select_session():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LIF data')
    sessions = sorted([path for path in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(path)])
    if not sessions:
        print('No sessions in LIF data/')
        sys.exit(1)

    print('\nSessions:')
    for idx, session_path in enumerate(sessions):
        print(f'  [{idx}] {os.path.basename(session_path)}')
    choice = input('Select (Enter=last): ').strip()
    return sessions[-1] if choice == '' else sessions[int(choice)]


def run_pipeline(session_dir, bin_size_ms=50.0, mode='binary', top_k=500,
                 threshold=None, use_combined=False, output_tag=None,
                 max_lag_ms=25.0, learned_lif_path=None):
    session_name = os.path.basename(session_dir)
    output_tag = output_tag.strip().replace(' ', '_') if output_tag else None
    output_name = session_name if not output_tag else f'{session_name}_{output_tag}'
    max_lag_bins = int(np.ceil(max(0.0, float(max_lag_ms)) / float(bin_size_ms))) if bin_size_ms > 0 else 0

    print('\n' + '=' * 70)
    print('CORRELATION CONNECTIVITY MAP')
    print(f'Session: {session_name}')
    if output_tag:
        print(f'Output tag: {output_tag}')
    print(f'Bin size: {bin_size_ms:.1f} ms | Mode: {mode} | Top-K edges: {top_k}')
    print(f'Max lag window: {max_lag_ms:.1f} ms ({max_lag_bins} bins)')
    if threshold is not None:
        print(f'Min |score| threshold: {threshold:.4f}')
    print('=' * 70)

    data = load_all_recordings(
        session_dir,
        bin_size_ms=bin_size_ms,
        mode=mode,
        use_combined=use_combined,
    )
    spike_matrix = data['spike_matrix']
    positions = data['neuron_positions']
    connections = data['connections']
    cluster_assignments = data['cluster_assignments']
    n_neurons = len(positions)

    print(f'  Loaded {data["n_recordings"]} recording block(s)')
    print(f'  Spike matrix: {spike_matrix.shape} from {data["source_note"]}')
    print(f'  Total duration: {data["total_duration_ms"] / 1000:.1f}s')

    print('  Computing lagged correlation scores...')
    directed_lagged_scores, best_lag_matrix, zero_lag_matrix = compute_lagged_correlation_matrix(
        spike_matrix,
        max_lag_bins=max_lag_bins,
    )
    lagged_score_matrix = directed_to_undirected_score_matrix(directed_lagged_scores)

    true_directed_weights = build_true_directed_weight_matrix(connections, n_neurons)
    true_undirected_matrix = build_true_undirected_matrix(true_directed_weights)
    true_undirected_scores = directed_to_undirected_score_matrix(true_directed_weights)

    lagged_eval_results = evaluate_score_matrix(lagged_score_matrix, true_undirected_matrix)

    print('  Selecting lagged-correlation edges...')
    lagged_selection = select_undirected_edges(
        lagged_score_matrix,
        positions,
        true_matrix=true_undirected_matrix,
        top_k=top_k,
        threshold=threshold,
    )
    validate_edge_mapping(lagged_score_matrix, positions, lagged_selection)
    print(
        f'  Lagged correlation: {len(lagged_selection["edge_indices"])} edges '
        f'(cutoff |score|={lagged_selection["score_cutoff"]:.4f}) '
        f'AUC={lagged_eval_results["auc"]:.4f} AP={lagged_eval_results["ap"]:.4f}'
    )

    nonzero_threshold = float(np.nextafter(np.float32(0.0), np.float32(1.0)))
    true_selection = select_undirected_edges(
        true_undirected_scores,
        positions,
        true_matrix=true_undirected_matrix,
        top_k=None,
        threshold=nonzero_threshold,
    )
    validate_edge_mapping(true_undirected_scores, positions, true_selection)
    print(f'  True map edges: {len(true_selection["edge_indices"])} undirected edges')

    learned_lif = load_learned_lif_connectivity(session_name, explicit_path=learned_lif_path)
    learned_lif_eval_results = None
    learned_lif_selection = None
    if learned_lif is not None:
        learned_lif_eval_results = evaluate_score_matrix(
            learned_lif['undirected_matrix'],
            true_undirected_matrix,
        )
        learned_lif_selection = select_undirected_edges(
            learned_lif['undirected_matrix'],
            positions,
            true_matrix=true_undirected_matrix,
            top_k=None if learned_lif['threshold'] is not None else top_k,
            threshold=learned_lif['threshold'],
        )
        validate_edge_mapping(learned_lif['undirected_matrix'], positions, learned_lif_selection)
        print(
            f'  Learned-LIF: {len(learned_lif_selection["edge_indices"])} edges '
            f'(cutoff |w|={learned_lif_selection["score_cutoff"]:.4f}) '
            f'AUC={learned_lif_eval_results["auc"]:.4f} AP={learned_lif_eval_results["ap"]:.4f}'
        )
    else:
        print('  Learned-LIF connectivity not found for this session; skipping learned-LIF map outputs')

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'correlation_outputs')
    os.makedirs(output_dir, exist_ok=True)

    heatmap_path = os.path.join(output_dir, f'lagged_correlation_heatmap_{output_name}.png')
    lagged_map_path = os.path.join(output_dir, f'lagged_correlation_map_{output_name}.png')
    true_map_path = os.path.join(output_dir, f'true_connectivity_map_{output_name}.png')
    learned_lif_map_path = os.path.join(output_dir, f'learned_lif_connectivity_map_{output_name}.png')
    comparison_path = os.path.join(output_dir, f'connectivity_map_comparison_{output_name}.png')
    npz_path = os.path.join(output_dir, f'correlation_connectivity_{output_name}.npz')
    csv_path = os.path.join(output_dir, f'correlation_edges_{output_name}.csv')

    plot_heatmap(
        lagged_score_matrix,
        cluster_assignments,
        title=f'Lagged Correlation Heatmap — {session_name}',
        output_path=heatmap_path,
        colorbar_label='Lagged correlation score',
    )

    lagged_summary_lines = summarize_selection(
        lagged_selection,
        score_label='|score|',
        extra_lines=[
            f'AUC: {lagged_eval_results["auc"]:.4f}',
            f'AP: {lagged_eval_results["ap"]:.4f}',
            f'Lag window: {max_lag_ms:.1f} ms',
            f'Source: {data["source_note"]}',
        ],
    )
    plot_connectivity_map(
        positions,
        lagged_selection,
        title=f'Lagged Correlation Connectivity Map — {session_name}',
        summary_lines=lagged_summary_lines,
        output_path=lagged_map_path,
    )

    true_summary_lines = summarize_selection(
        true_selection,
        score_label='|w|',
        extra_lines=[
            'All true undirected edges',
            f'True edge count: {len(true_selection["edge_indices"])}',
        ],
    )
    plot_connectivity_map(
        positions,
        true_selection,
        title=f'True Connectivity Map — {session_name}',
        summary_lines=true_summary_lines,
        output_path=true_map_path,
    )

    comparison_specs = [
        {
            'selection': true_selection,
            'title': 'True Connectivity',
            'summary_lines': true_summary_lines,
        },
        {
            'selection': lagged_selection,
            'title': 'Lagged Correlation',
            'summary_lines': lagged_summary_lines,
        },
    ]

    if learned_lif_selection is not None and learned_lif_eval_results is not None:
        learned_name = os.path.basename(learned_lif['path'])
        learned_lif_summary_lines = summarize_selection(
            learned_lif_selection,
            score_label='|w|',
            extra_lines=[
                f'AUC: {learned_lif_eval_results["auc"]:.4f}',
                f'AP: {learned_lif_eval_results["ap"]:.4f}',
                f'Threshold: {learned_lif["threshold"]:.4f}' if learned_lif['threshold'] is not None else 'Threshold: none',
                f'Source: {learned_name}',
            ],
        )
        plot_connectivity_map(
            positions,
            learned_lif_selection,
            title=f'Learned-LIF Connectivity Map — {session_name}',
            summary_lines=learned_lif_summary_lines,
            output_path=learned_lif_map_path,
        )
        comparison_specs.append(
            {
                'selection': learned_lif_selection,
                'title': 'Learned-LIF',
                'summary_lines': learned_lif_summary_lines,
            }
        )

    plot_comparison_maps(
        positions,
        comparison_specs,
        title=f'Connectivity Map Comparison — {session_name}',
        output_path=comparison_path,
    )

    save_edge_csv(csv_path, lagged_selection)

    np.savez_compressed(
        npz_path,
        lagged_correlation_matrix=lagged_score_matrix,
        lagged_directed_scores=directed_lagged_scores,
        best_lag_matrix=best_lag_matrix,
        zero_lag_correlation_matrix=zero_lag_matrix,
        adjacency_matrix=lagged_selection['adjacency'],
        edge_indices=lagged_selection['edge_indices'],
        edge_scores=lagged_selection['edge_scores'],
        edge_abs_scores=lagged_selection['edge_abs_scores'],
        edge_distances=lagged_selection['edge_distances'],
        edge_true=lagged_selection['edge_true'] if lagged_selection['edge_true'] is not None else np.array([], dtype=np.int32),
        cell_indices=np.arange(n_neurons, dtype=np.int32),
        neuron_positions=positions,
        cluster_assignments=cluster_assignments if cluster_assignments is not None else np.array([], dtype=np.int32),
        true_directed_weights=true_directed_weights,
        true_undirected_matrix=true_undirected_matrix,
        true_undirected_scores=true_undirected_scores,
        learned_lif_undirected_scores=(learned_lif['undirected_matrix'] if learned_lif is not None else np.array([], dtype=np.float32)),
        learned_lif_threshold=(float(learned_lif['threshold']) if learned_lif is not None and learned_lif['threshold'] is not None else np.nan),
        learned_lif_path=(learned_lif['path'] if learned_lif is not None else ''),
        bin_size_ms=float(bin_size_ms),
        max_lag_ms=float(max_lag_ms),
        max_lag_bins=int(max_lag_bins),
        top_k=int(top_k) if top_k is not None else -1,
        threshold=float(threshold) if threshold is not None else np.nan,
        score_cutoff=float(lagged_selection['score_cutoff']),
        total_duration_ms=float(data['total_duration_ms']),
        n_recordings=int(data['n_recordings']),
        session_name=session_name,
        output_name=output_name,
        source_note=data['source_note'],
        lagged_eval_auc=float(lagged_eval_results['auc']),
        lagged_eval_ap=float(lagged_eval_results['ap']),
        learned_lif_eval_auc=(float(learned_lif_eval_results['auc']) if learned_lif_eval_results is not None else np.nan),
        learned_lif_eval_ap=(float(learned_lif_eval_results['ap']) if learned_lif_eval_results is not None else np.nan),
    )

    print(f'  Heatmap saved: {heatmap_path}')
    print(f'  Lagged correlation map saved: {lagged_map_path}')
    print(f'  True connectivity map saved: {true_map_path}')
    if learned_lif_selection is not None:
        print(f'  Learned-LIF map saved: {learned_lif_map_path}')
    print(f'  Comparison figure saved: {comparison_path}')
    print(f'  Edge list saved: {csv_path}')
    print(f'  Data saved: {npz_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Correlation connectivity map')
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--bin-size-ms', type=float, default=50.0,
                        help='Time bin size for correlation features')
    parser.add_argument('--mode', type=str, default='binary',
                        choices=['binary', 'count'],
                        help='Whether to bin spikes as binary activity or counts')
    parser.add_argument('--top-k', type=int, default=500,
                        help='Number of strongest lagged-correlation edges to keep (0 keeps none)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Optional minimum absolute lagged-correlation cutoff before top-k')
    parser.add_argument('--max-lag-ms', type=float, default=25.0,
                        help='Maximum positive lag window used for directed lagged correlation')
    parser.add_argument('--use-combined-recording', action='store_true',
                        help='Use recording_combined.npz when available')
    parser.add_argument('--learned-lif-path', type=str, default=None,
                        help='Optional explicit path to learned-LIF connectivity NPZ')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional suffix for saved artifact names')
    args = parser.parse_args()

    session_dir = args.session if args.session else select_session()
    run_pipeline(
        session_dir,
        bin_size_ms=args.bin_size_ms,
        mode=args.mode,
        top_k=args.top_k,
        threshold=args.threshold,
        use_combined=args.use_combined_recording,
        output_tag=args.output_tag,
        max_lag_ms=args.max_lag_ms,
        learned_lif_path=args.learned_lif_path,
    )