"""
Predict connectivity on a new network using a pre-trained GNN model.
No re-training required - just loads the saved model and scalers.

Usage:5
    python predict_new_network.py <path_to_metadata_file>

Example:
    python predict_new_network.py "LIF data/20260107_132124/session_gnn_metadata.json"
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# Import from main script
from gnn_cross_network_prediction import (
    InductiveGNN, CrossNetworkConfig, CrossNetworkDataset,
    predict_new_network, SlidingWindowPredictor, ensure_output_dirs
)


def load_trained_model(model_path: str = 'gnn_outputs/models/trained_gnn_model.pt'):
    """Load pre-trained model and scalers."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run gnn_cross_network_prediction.py first to train and save the model."
        )

    # weights_only=False needed because checkpoint contains numpy arrays (scalers)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Recreate config
    config = checkpoint['config']

    # Recreate model
    model = InductiveGNN(
        node_dim=checkpoint['node_dim'],
        edge_dim=checkpoint['edge_dim'],
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Recreate dataset with fitted scalers
    dataset = CrossNetworkDataset(config)
    dataset.feature_extractor.node_scaler.mean_ = checkpoint['node_scaler_mean']
    dataset.feature_extractor.node_scaler.scale_ = checkpoint['node_scaler_scale']
    dataset.feature_extractor.edge_scaler.mean_ = checkpoint['edge_scaler_mean']
    dataset.feature_extractor.edge_scaler.scale_ = checkpoint['edge_scaler_scale']
    dataset.feature_extractor.fitted = True

    # Load training optimal threshold
    training_threshold = checkpoint.get('optimal_threshold', 0.5)

    print(f"Loaded model from: {model_path}")
    print(f"  Node features: {checkpoint['node_dim']}")
    print(f"  Edge features: {checkpoint['edge_dim']}")
    print(f"  Training optimal threshold: {training_threshold:.4f}")

    return model, config, dataset, training_threshold


def analyze_connections(connections: np.ndarray) -> dict:
    """
    Analyze connection structure to check for bidirectional edges.

    Returns:
        Dictionary with connection statistics
    """
    conn_set = set()
    bidirectional_count = 0

    for c in connections:
        i, j = int(c[0]), int(c[1])
        if (j, i) in conn_set:
            bidirectional_count += 1
        conn_set.add((i, j))

    # Count unique undirected pairs
    undirected_set = set()
    for c in connections:
        i, j = int(c[0]), int(c[1])
        undirected_set.add((min(i, j), max(i, j)))

    return {
        'total_entries': len(connections),
        'unique_directed': len(conn_set),
        'unique_undirected': len(undirected_set),
        'bidirectional_pairs': bidirectional_count,
        'unidirectional': len(conn_set) - 2 * bidirectional_count
    }


def predict_and_visualize(model, metadata_file: str, config, dataset, training_threshold: float = 0.5):
    """Run prediction and generate visualization using the training optimal threshold."""

    print("\n" + "="*70)
    print("PREDICTING ON NEW NETWORK")
    print("="*70)

    # Load network data first for analysis
    test_network = dataset.load_network(metadata_file)
    positions = test_network['neuron_positions']
    connections = test_network['connections']

    # Analyze connection structure
    conn_stats = analyze_connections(connections)
    print(f"\n  Connection Structure Analysis:")
    print(f"    Total connection entries: {conn_stats['total_entries']}")
    print(f"    Unique directed edges:    {conn_stats['unique_directed']}")
    print(f"    Unique undirected pairs:  {conn_stats['unique_undirected']}")
    print(f"    Bidirectional pairs:      {conn_stats['bidirectional_pairs']} (A↔B both exist)")
    print(f"    Unidirectional edges:     {conn_stats['unidirectional']} (A→B only)")

    # Run prediction
    prediction_results = predict_new_network(
        model,
        metadata_file,
        config,
        dataset
    )

    # Get raw predictions
    raw_edges = prediction_results['pred_edges']
    raw_probs = prediction_results['pred_probs']
    raw_labels = prediction_results['true_labels']

    # Deduplicate edges (keep highest probability for each unique edge)
    # This handles duplicates from overlapping sliding windows
    edge_dict = {}  # (src, tgt) -> (prob, label, index)
    for k in range(raw_edges.shape[1]):
        src, tgt = int(raw_edges[0, k]), int(raw_edges[1, k])
        prob = raw_probs[k]
        label = raw_labels[k]
        key = (src, tgt)
        if key not in edge_dict or prob > edge_dict[key][0]:
            edge_dict[key] = (prob, label)

    # Convert back to arrays
    n_unique = len(edge_dict)
    pred_edges = np.zeros((2, n_unique), dtype=np.int64)
    pred_probs = np.zeros(n_unique)
    true_labels = np.zeros(n_unique)

    for idx, ((src, tgt), (prob, label)) in enumerate(edge_dict.items()):
        pred_edges[0, idx] = src
        pred_edges[1, idx] = tgt
        pred_probs[idx] = prob
        true_labels[idx] = label

    n_duplicates = raw_edges.shape[1] - n_unique
    if n_duplicates > 0:
        print(f"\n  Deduplicated: removed {n_duplicates} duplicate edges from {raw_edges.shape[1]} total")

    # Additional prediction statistics (now using deduplicated data)
    n_evaluated = len(true_labels)
    n_positive_evaluated = int(np.sum(true_labels))
    n_negative_evaluated = n_evaluated - n_positive_evaluated

    print(f"\n  Evaluation Statistics (unique edges):")
    print(f"    Total edges evaluated:    {n_evaluated}")
    print(f"    Positive (connected):     {n_positive_evaluated}")
    print(f"    Negative (not connected): {n_negative_evaluated}")
    print(f"    Class imbalance ratio:    1:{n_negative_evaluated/max(1,n_positive_evaluated):.1f}")

    # Use the optimal threshold from training (not per-network)
    optimal_threshold = training_threshold
    precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
    # Find the index closest to the training threshold for plotting
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold)) if len(thresholds) > 0 else 0

    # Create figure with 3x2 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ---- Plot 1: Actual Connections ----
    ax1 = axes[0, 0]
    ax1.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=50, edgecolors='navy', zorder=3)

    actual_mask = true_labels == 1
    actual_edges = pred_edges[:, actual_mask]
    for k in range(actual_edges.shape[1]):
        i, j = actual_edges[0, k], actual_edges[1, k]
        ax1.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                'g-', alpha=0.3, linewidth=0.5)

    ax1.set_title(f'Actual Connections (n={int(np.sum(actual_mask))})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_aspect('equal')

    # ---- Plot 2: Predicted Connections (threshold=0.5) ----
    ax2 = axes[0, 1]
    ax2.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=50, edgecolors='navy', zorder=3)

    threshold = 0.5
    pred_mask = pred_probs > threshold
    pred_positive_edges = pred_edges[:, pred_mask]

    for k in range(pred_positive_edges.shape[1]):
        i, j = pred_positive_edges[0, k], pred_positive_edges[1, k]
        edge_idx = np.where(pred_mask)[0][k]
        is_correct = true_labels[edge_idx] == 1
        color = 'green' if is_correct else 'red'
        alpha = 0.5 if is_correct else 0.3
        ax2.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color=color, alpha=alpha, linewidth=0.5)

    tp_05 = np.sum((pred_probs > threshold) & (true_labels == 1))
    fp_05 = np.sum((pred_probs > threshold) & (true_labels == 0))
    ax2.set_title(f'Predicted (thresh=0.5)\nTP={tp_05} (green), FP={fp_05} (red)',
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_aspect('equal')

    # ---- Plot 3: Predicted Connections (optimal threshold) ----
    ax3 = axes[0, 2]
    ax3.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=50, edgecolors='navy', zorder=3)

    pred_mask_opt = pred_probs >= optimal_threshold
    pred_positive_edges_opt = pred_edges[:, pred_mask_opt]

    for k in range(pred_positive_edges_opt.shape[1]):
        i, j = pred_positive_edges_opt[0, k], pred_positive_edges_opt[1, k]
        edge_idx = np.where(pred_mask_opt)[0][k]
        is_correct = true_labels[edge_idx] == 1
        color = 'green' if is_correct else 'red'
        alpha = 0.5 if is_correct else 0.3
        ax3.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color=color, alpha=alpha, linewidth=0.5)

    tp_opt = np.sum((pred_probs >= optimal_threshold) & (true_labels == 1))
    fp_opt = np.sum((pred_probs >= optimal_threshold) & (true_labels == 0))
    fn_opt = np.sum((pred_probs < optimal_threshold) & (true_labels == 1))
    ax3.set_title(f'Predicted (optimal thresh={optimal_threshold:.3f})\nTP={tp_opt}, FP={fp_opt}, FN={fn_opt}',
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_aspect('equal')

    # ---- Plot 4: Score Distribution ----
    ax4 = axes[1, 0]
    pos_probs = pred_probs[true_labels == 1]
    neg_probs = pred_probs[true_labels == 0]

    ax4.hist(neg_probs, bins=50, alpha=0.6, label=f'No Connection (n={len(neg_probs)})', color='red')
    ax4.hist(pos_probs, bins=50, alpha=0.6, label=f'True Connection (n={len(pos_probs)})', color='green')
    ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax4.axvline(x=optimal_threshold, color='blue', linestyle='-', linewidth=2, label=f'Optimal={optimal_threshold:.3f}')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Count')
    ax4.set_title('Prediction Score Distribution', fontsize=12, fontweight='bold')
    ax4.legend()

    # ---- Plot 5: Precision-Recall Curve ----
    ax5 = axes[1, 1]
    ax5.plot(recall, precision, 'b-', linewidth=2)
    ax5.fill_between(recall, precision, alpha=0.2)

    ax5.scatter([recall[optimal_idx]], [precision[optimal_idx]], color='green', s=100, marker='*',
               label=f'Optimal (t={optimal_threshold:.2f})')

    # Mark threshold=0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) > 0 else 0
    ax5.scatter([recall[idx_05]], [precision[idx_05]], color='red', s=80,
               label=f't=0.5 (P={precision[idx_05]:.2f}, R={recall[idx_05]:.2f})')

    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title(f'Precision-Recall Curve (AP={prediction_results["ap"]:.3f})',
                 fontsize=12, fontweight='bold')
    ax5.legend(loc='lower left')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3)

    # ---- Plot 6: Comparison Summary ----
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Create summary text
    summary_text = f"""
    PREDICTION SUMMARY
    ══════════════════════════════

    Network: {os.path.basename(os.path.dirname(metadata_file))}
    Neurons: {len(positions)}

    Ground Truth:
      Total connections: {int(np.sum(actual_mask))}

    Threshold = 0.5:
      Predicted: {tp_05 + fp_05}
      True Positives: {tp_05}
      False Positives: {fp_05}
      Precision: {tp_05/(tp_05+fp_05+1e-10):.2%}
      Recall: {tp_05/max(1,int(np.sum(actual_mask))):.2%}

    Optimal Threshold = {optimal_threshold:.3f}:
      Predicted: {tp_opt + fp_opt}
      True Positives: {tp_opt}
      False Positives: {fp_opt}
      False Negatives: {fn_opt}
      Precision: {tp_opt/(tp_opt+fp_opt+1e-10):.2%}
      Recall: {tp_opt/max(1,int(np.sum(actual_mask))):.2%}
      F1 Score: {2*tp_opt/(2*tp_opt+fp_opt+fn_opt+1e-10):.2%}

    Metrics:
      AUC: {prediction_results['auc']:.4f}
      AP:  {prediction_results['ap']:.4f}
    """
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save with network name to figures directory
    output_dirs = ensure_output_dirs()
    network_name = os.path.basename(os.path.dirname(metadata_file))
    output_file = os.path.join(output_dirs['figures'], f'prediction_{network_name}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()

    # Print threshold analysis
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS")
    print("="*70)
    print(f"\nTraining optimal threshold: {optimal_threshold:.3f}")
    print(f"  Precision: {precision[optimal_idx]:.4f}")
    print(f"  Recall: {recall[optimal_idx]:.4f}")

    # ---- Export Predicted Connections ----
    export_predictions(
        pred_edges, pred_probs, true_labels,
        network_name, output_dirs,
        optimal_threshold
    )

    return prediction_results


def export_predictions(pred_edges: np.ndarray, pred_probs: np.ndarray,
                       true_labels: np.ndarray, network_name: str,
                       output_dirs: dict, optimal_threshold: float):
    """
    Export predicted connections to both NPZ and CSV formats.

    Note: Input data should already be deduplicated by predict_and_visualize().

    Saves:
    1. NPZ file with all predictions and metadata
    2. CSV file with predicted connections (above threshold)
    """
    n_unique = pred_edges.shape[1]

    # Sort by probability (highest first)
    sort_idx = np.argsort(pred_probs)[::-1]
    sorted_edges = pred_edges[:, sort_idx]
    sorted_probs = pred_probs[sort_idx]
    sorted_labels = true_labels[sort_idx]

    # Get predictions above optimal threshold
    above_threshold = sorted_probs >= optimal_threshold
    predicted_connections = sorted_edges[:, above_threshold]
    predicted_probs = sorted_probs[above_threshold]
    predicted_true = sorted_labels[above_threshold]

    # ---- Save NPZ file ----
    npz_path = os.path.join(output_dirs['results'], f'predictions_{network_name}.npz')
    np.savez(
        npz_path,
        # All predictions (sorted by probability)
        all_edges=sorted_edges,           # [2, n_edges] - source, target
        all_probs=sorted_probs,           # [n_edges] - probability scores
        all_true_labels=sorted_labels,    # [n_edges] - ground truth
        # Predictions above optimal threshold
        predicted_edges=predicted_connections,
        predicted_probs=predicted_probs,
        predicted_true_labels=predicted_true,
        # Metadata
        optimal_threshold=optimal_threshold,
        n_predicted=np.sum(above_threshold),
        n_true_positives=np.sum(predicted_true),
        n_false_positives=np.sum(above_threshold) - np.sum(predicted_true)
    )
    print(f"\n  Predictions saved to: {npz_path}")

    # ---- Save CSV file ----
    csv_path = os.path.join(output_dirs['results'], f'predictions_{network_name}.csv')
    with open(csv_path, 'w') as f:
        # Header
        f.write("# Predicted connections for network: {}\n".format(network_name))
        f.write("# Threshold: {:.4f} (optimal F1)\n".format(optimal_threshold))
        f.write("# Columns: source_neuron, target_neuron, probability, is_true_connection\n")
        f.write("source,target,probability,true_connection\n")

        # Write predictions above threshold
        for k in range(predicted_connections.shape[1]):
            src = predicted_connections[0, k]
            tgt = predicted_connections[1, k]
            prob = predicted_probs[k]
            true = int(predicted_true[k])
            f.write(f"{src},{tgt},{prob:.6f},{true}\n")

    print(f"  Predictions saved to: {csv_path}")

    # Summary
    n_pred = np.sum(above_threshold)
    n_tp = int(np.sum(predicted_true))
    n_fp = n_pred - n_tp
    print(f"\n  Export Summary (threshold={optimal_threshold:.3f}):")
    print(f"    Unique edges evaluated: {n_unique}")
    print(f"    Predicted connections:  {n_pred}")
    print(f"    True positives:         {n_tp}")
    print(f"    False positives:        {n_fp}")


def select_network_interactive() -> str:
    """Display available networks and let user choose interactively."""
    import glob
    import json

    metadata_files = sorted(glob.glob('LIF data/*/session_gnn_metadata.json'))

    if not metadata_files:
        print("No metadata files found in 'LIF data/*/session_gnn_metadata.json'")
        print("Make sure you run this script from the simulation directory.")
        sys.exit(1)

    print("\n" + "="*70)
    print("AVAILABLE NETWORKS")
    print("="*70)

    # Load and display info for each network
    for i, mf in enumerate(metadata_files):
        try:
            with open(mf) as f:
                meta = json.load(f)
            network_name = os.path.basename(os.path.dirname(mf))
            n_neurons = meta.get('num_neurons', '?')
            n_connections = meta.get('num_connections', '?')
            duration_s = meta.get('recording_duration', 0) / 1000
            print(f"  [{i}] {network_name} - {n_neurons} neurons, {n_connections} connections, {duration_s:.0f}s recording")
        except Exception as e:
            print(f"  [{i}] {mf} (error reading: {e})")

    print()
    print(f"  [Enter] Use newest: {os.path.basename(os.path.dirname(metadata_files[-1]))}")
    print(f"  [a] Predict ALL networks")
    print()

    # Get user input
    choice = input("Select network number (or Enter for newest, 'a' for all): ").strip().lower()

    if choice == '':
        return metadata_files[-1]
    elif choice == 'a':
        return 'all'
    else:
        try:
            idx = int(choice)
            if 0 <= idx < len(metadata_files):
                return metadata_files[idx]
            else:
                print(f"Invalid index. Using newest network.")
                return metadata_files[-1]
        except ValueError:
            print(f"Invalid input. Using newest network.")
            return metadata_files[-1]


if __name__ == "__main__":
    # Get metadata file from command line or interactive selection
    if len(sys.argv) > 1:
        metadata_file = sys.argv[1]
    else:
        metadata_file = select_network_interactive()

    # Load trained model
    model_path = os.path.join('gnn_outputs', 'models', 'trained_gnn_model.pt')
    model, config, dataset, training_threshold = load_trained_model(model_path)

    # Handle 'all' option or single network
    if metadata_file == 'all':
        import glob
        metadata_files = sorted(glob.glob('LIF data/*/session_gnn_metadata.json'))
        all_results = []

        for i, mf in enumerate(metadata_files):
            print(f"\n{'='*70}")
            print(f"NETWORK {i+1}/{len(metadata_files)}")
            print(f"{'='*70}")

            results = predict_and_visualize(model, mf, config, dataset, training_threshold)
            all_results.append({
                'file': mf,
                'name': os.path.basename(os.path.dirname(mf)),
                **results
            })

        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY - ALL NETWORKS")
        print("="*70)
        print(f"{'Network':<20} {'Neurons':<10} {'Connections':<12} {'AUC':<10} {'AP':<10}")
        print("-" * 62)
        for r in all_results:
            print(f"{r['name']:<20} {r['n_neurons']:<10} {r['n_connections']:<12} {r['auc']:<10.4f} {r['ap']:<10.4f}")
        print("-" * 62)
        avg_auc = np.mean([r['auc'] for r in all_results])
        avg_ap = np.mean([r['ap'] for r in all_results])
        print(f"{'AVERAGE':<20} {'':<10} {'':<12} {avg_auc:<10.4f} {avg_ap:<10.4f}")

    else:
        # Single network prediction
        results = predict_and_visualize(model, metadata_file, config, dataset, training_threshold)

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  AP:  {results['ap']:.4f}")
        print(f"  Neurons: {results['n_neurons']}")
        print(f"  True connections: {results['n_connections']}")
