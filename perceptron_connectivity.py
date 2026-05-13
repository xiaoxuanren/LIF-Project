"""
Perceptron-based connectivity inference from spike trains.

Based on: Ren, Bok, Vareberg, Hai (2023)
"Stimulation-mediated reverse engineering of silent neural networks"

For each postsynaptic neuron, learns input weights from all other neurons
using the perceptron learning rule on binary spike trains at 1 ms resolution.

Usage:
    python perceptron_connectivity.py [session_folder]
    python perceptron_connectivity.py   # interactive selection
"""

import numpy as np
import os
import sys
import glob
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, confusion_matrix)


# ============================================================================
# CORE: Perceptron weight learning
# ============================================================================

def spike_times_to_binary(spike_times, duration_ms, dt=1.0):
    """Convert spike times to binary matrix at given resolution."""
    n_neurons = len(spike_times)
    n_bins = int(duration_ms / dt)
    binary = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for i in range(n_neurons):
        for t in spike_times[i]:
            bin_idx = int(t / dt)
            if 0 <= bin_idx < n_bins:
                binary[i, bin_idx] = 1
    return binary


def learn_weights_perceptron(spike_matrix, target_neuron_idx, threshold=20.0,
                             lr=0.01, n_iterations=100, delay_ms=1):
    """
    Learn input weights to a single postsynaptic neuron using perceptron rule.

    Args:
        spike_matrix: Binary spike matrix [n_neurons, T] at 1 ms resolution
        target_neuron_idx: Index of the postsynaptic neuron
        threshold: Firing threshold (mV-like units)
        lr: Learning rate
        n_iterations: Number of passes through the data
        delay_ms: Synaptic delay in bins (1 bin = 1 ms)

    Returns:
        weights: Learned weights [n_neurons] (weight for target neuron is 0)
    """
    n_neurons, T = spike_matrix.shape
    weights = np.zeros(n_neurons, dtype=np.float64)

    # Target neuron's spike train
    y_actual = spike_matrix[target_neuron_idx]

    for iteration in range(n_iterations):
        # Iterate through time steps (skip first delay_ms bins)
        for t in range(delay_ms, T):
            # Presynaptic input at t - delay
            x = spike_matrix[:, t - delay_ms]

            # Predicted membrane potential
            v = np.dot(weights, x)

            # Predicted spike
            y_pred = 1.0 if v > threshold else 0.0

            # Actual spike at time t
            y_true = y_actual[t]

            # Perceptron update
            if y_pred != y_true:
                error = y_true - y_pred
                weights += lr * error * x

        # Zero out self-connection
        weights[target_neuron_idx] = 0.0

    return weights


def learn_weights_perceptron_vectorized(spike_matrix, target_neuron_idx,
                                        threshold=20.0, lr=0.01,
                                        n_iterations=100, delay_ms=1,
                                        trial_duration_ms=1000):
    """
    Vectorized perceptron learning using trial-based approach (matching paper).

    Splits recording into trials, processes each trial as a batch.
    """
    n_neurons, T = spike_matrix.shape
    weights = np.zeros(n_neurons, dtype=np.float64)

    y_actual = spike_matrix[target_neuron_idx]

    # Split into trials
    trial_bins = int(trial_duration_ms)
    n_trials = T // trial_bins

    for iteration in range(n_iterations):
        # Shuffle trial order each iteration
        trial_order = np.random.permutation(n_trials)

        for trial_idx in trial_order:
            t_start = trial_idx * trial_bins
            t_end = t_start + trial_bins

            # Process each time step in this trial
            for t in range(t_start + delay_ms, t_end):
                x = spike_matrix[:, t - delay_ms]
                v = np.dot(weights, x)
                y_pred = 1.0 if v > threshold else 0.0
                y_true = y_actual[t]

                if y_pred != y_true:
                    error = y_true - y_pred
                    weights += lr * error * x

        weights[target_neuron_idx] = 0.0

    return weights


def learn_all_weights(spike_matrix, n_iterations=50, lr=0.01, threshold=20.0,
                      delay_ms=1, verbose=True):
    """
    Learn connectivity for all neurons.

    Returns:
        weight_matrix: [n_neurons, n_neurons] where W[i,j] = weight from j to i
    """
    n_neurons = spike_matrix.shape[0]
    weight_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float64)

    t_start = time.time()

    for i in range(n_neurons):
        weights = learn_weights_perceptron(
            spike_matrix, i,
            threshold=threshold, lr=lr,
            n_iterations=n_iterations, delay_ms=delay_ms
        )
        weight_matrix[i, :] = weights

        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n_neurons - i - 1) / rate
            print(f"  Neuron {i+1}/{n_neurons}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return weight_matrix


# ============================================================================
# EVALUATION
# ============================================================================

def build_ground_truth_matrix(connections, n_neurons):
    """
    Build ground truth weight matrix from connections array.

    Returns:
        weight_matrix: [n_neurons, n_neurons] where W[post, pre] = weight
        binary_matrix: [n_neurons, n_neurons] where 1 = connected
    """
    weight_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    binary_matrix = np.zeros((n_neurons, n_neurons), dtype=np.int32)

    for c in connections:
        pre, post = int(c[0]), int(c[1])
        weight = float(c[2])
        weight_matrix[post, pre] = weight
        binary_matrix[post, pre] = 1

    return weight_matrix, binary_matrix


def evaluate_predictions(learned_weights, true_weights, true_binary,
                         weight_threshold=None):
    """
    Evaluate learned weights against ground truth.

    Returns dict with metrics.
    """
    n_neurons = learned_weights.shape[0]

    # Flatten, excluding diagonal (self-connections)
    mask = ~np.eye(n_neurons, dtype=bool)
    w_learned = learned_weights[mask]
    w_true = true_weights[mask]
    b_true = true_binary[mask]

    # Use absolute learned weight as connectivity score
    scores = np.abs(w_learned)

    # Metrics
    results = {}

    # Correlation between learned and true weights
    valid = b_true == 1  # only for true connections
    if np.sum(valid) > 1:
        results['weight_correlation'] = np.corrcoef(w_learned[valid], w_true[valid])[0, 1]
        results['weight_rmse'] = np.sqrt(np.mean((w_learned[valid] - w_true[valid])**2))
    else:
        results['weight_correlation'] = 0.0
        results['weight_rmse'] = float('inf')

    # Binary classification metrics
    if len(np.unique(b_true)) > 1:
        results['auc'] = roc_auc_score(b_true, scores)
        results['ap'] = average_precision_score(b_true, scores)
    else:
        results['auc'] = 0.0
        results['ap'] = 0.0

    # Find optimal threshold on absolute weight
    if weight_threshold is None:
        precision, recall, thresholds = precision_recall_curve(b_true, scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        weight_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    results['optimal_threshold'] = weight_threshold

    # Metrics at optimal threshold
    predicted = (scores >= weight_threshold).astype(int)
    tp = np.sum((predicted == 1) & (b_true == 1))
    fp = np.sum((predicted == 1) & (b_true == 0))
    fn = np.sum((predicted == 0) & (b_true == 1))
    tn = np.sum((predicted == 0) & (b_true == 0))

    results['tp'] = int(tp)
    results['fp'] = int(fp)
    results['fn'] = int(fn)
    results['tn'] = int(tn)
    results['precision'] = tp / (tp + fp + 1e-10)
    results['recall'] = tp / (tp + fn + 1e-10)
    results['f1'] = 2 * tp / (2 * tp + fp + fn + 1e-10)
    results['n_true_connections'] = int(np.sum(b_true))
    results['n_predicted'] = int(np.sum(predicted))

    # Sign accuracy (for true connections, is the sign correct?)
    if np.sum(valid) > 0:
        sign_correct = np.sign(w_learned[valid]) == np.sign(w_true[valid])
        results['sign_accuracy'] = np.mean(sign_correct)
    else:
        results['sign_accuracy'] = 0.0

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(learned_weights, true_weights, true_binary, connections,
                 neuron_positions, results, session_name, output_dir):
    """Generate 6-panel visualization."""

    n_neurons = learned_weights.shape[0]
    mask = ~np.eye(n_neurons, dtype=bool)
    w_learned = learned_weights[mask]
    w_true = true_weights[mask]
    b_true = true_binary[mask]
    scores = np.abs(w_learned)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Perceptron Connectivity Inference — {session_name}',
                 fontsize=14, fontweight='bold')

    # ---- Plot 1: Learned vs True Weights (true connections only) ----
    ax = axes[0, 0]
    valid = b_true == 1
    if np.sum(valid) > 0:
        ax.scatter(w_true[valid], w_learned[valid], s=3, alpha=0.3, c='steelblue')
        lim = max(abs(w_true[valid]).max(), abs(w_learned[valid]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='Perfect')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel('True Weight')
    ax.set_ylabel('Learned Weight')
    ax.set_title(f'Weight Recovery (r={results["weight_correlation"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Score Distribution ----
    ax = axes[0, 1]
    pos_scores = scores[b_true == 1]
    neg_scores = scores[b_true == 0]
    ax.hist(neg_scores, bins=50, alpha=0.6, label=f'No conn (n={len(neg_scores)})',
            color='red', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, label=f'Connected (n={len(pos_scores)})',
            color='green', density=True)
    ax.axvline(results['optimal_threshold'], color='blue', linestyle='-',
               linewidth=2, label=f'Threshold={results["optimal_threshold"]:.4f}')
    ax.set_xlabel('|Learned Weight|')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()

    # ---- Plot 3: Precision-Recall Curve ----
    ax = axes[0, 2]
    if results['auc'] > 0:
        precision, recall, _ = precision_recall_curve(b_true, scores)
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve (AUC={results["auc"]:.3f}, AP={results["ap"]:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ---- Plot 4: Actual Connections ----
    ax = axes[1, 0]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=30, edgecolors='navy', zorder=3)
    for c in connections:
        i, j = int(c[0]), int(c[1])
        ax.plot([neuron_positions[i, 0], neuron_positions[j, 0]],
                [neuron_positions[i, 1], neuron_positions[j, 1]],
                'g-', alpha=0.15, linewidth=0.3)
    ax.set_title(f'True Connections (n={len(connections)})')
    ax.set_aspect('equal')

    # ---- Plot 5: Predicted Connections ----
    ax = axes[1, 1]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=30, edgecolors='navy', zorder=3)
    thresh = results['optimal_threshold']
    for post in range(n_neurons):
        for pre in range(n_neurons):
            if pre == post:
                continue
            if abs(learned_weights[post, pre]) >= thresh:
                is_true = true_binary[post, pre] == 1
                color = 'green' if is_true else 'red'
                alpha = 0.3 if is_true else 0.15
                ax.plot([neuron_positions[pre, 0], neuron_positions[post, 0]],
                        [neuron_positions[pre, 1], neuron_positions[post, 1]],
                        color=color, alpha=alpha, linewidth=0.3)
    ax.set_title(f'Predicted (TP={results["tp"]}, FP={results["fp"]})')
    ax.set_aspect('equal')

    # ---- Plot 6: Summary ----
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
    PERCEPTRON CONNECTIVITY INFERENCE
    {'='*40}

    Network: {session_name}
    Neurons: {n_neurons}
    True connections: {results['n_true_connections']}

    Weight Recovery:
      Correlation (r): {results['weight_correlation']:.4f}
      RMSE: {results['weight_rmse']:.4f}
      Sign accuracy: {results['sign_accuracy']:.1%}

    Binary Classification:
      AUC:       {results['auc']:.4f}
      AP:        {results['ap']:.4f}

    At Optimal Threshold ({results['optimal_threshold']:.4f}):
      Precision: {results['precision']:.4f}
      Recall:    {results['recall']:.4f}
      F1:        {results['f1']:.4f}
      TP: {results['tp']}  FP: {results['fp']}
      FN: {results['fn']}  TN: {results['tn']}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'perceptron_{session_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def run_inference(session_dir, recording_idx=0, n_iterations=50, lr=0.01,
                  threshold=20.0, delay_ms=1):
    """Run perceptron connectivity inference on a single session/recording."""

    session_name = os.path.basename(session_dir)
    print(f"\n{'='*70}")
    print(f"PERCEPTRON CONNECTIVITY INFERENCE")
    print(f"Session: {session_name}, Recording: {recording_idx:03d}")
    print(f"Parameters: lr={lr}, iterations={n_iterations}, "
          f"threshold={threshold}, delay={delay_ms}ms")
    print(f"{'='*70}")

    # Load data
    rec_path = os.path.join(session_dir, f'recording{recording_idx:03d}.npz')
    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f"No network file in {session_dir}")
    net_path = net_files[0]

    rec_data = np.load(rec_path, allow_pickle=True)
    net_data = np.load(net_path, allow_pickle=True)

    spike_times = rec_data['spike_times']
    duration = float(rec_data['duration'])
    connections = net_data['connections']
    neuron_positions = net_data['neuron_positions']
    n_neurons = len(spike_times)

    print(f"\n  Neurons: {n_neurons}")
    print(f"  Connections: {len(connections)}")
    print(f"  Duration: {duration/1000:.0f}s")
    print(f"  Total spikes: {sum(len(s) for s in spike_times)}")

    # Convert to binary at 1 ms
    print(f"\n  Converting spike times to 1ms binary matrix...")
    spike_matrix = spike_times_to_binary(spike_times, duration, dt=1.0)
    print(f"  Spike matrix shape: {spike_matrix.shape}")

    # Build ground truth
    true_weights, true_binary = build_ground_truth_matrix(connections, n_neurons)

    # Learn weights
    print(f"\n  Learning weights ({n_iterations} iterations)...")
    t0 = time.time()
    learned_weights = learn_all_weights(
        spike_matrix, n_iterations=n_iterations, lr=lr,
        threshold=threshold, delay_ms=delay_ms, verbose=True
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Evaluate
    print(f"\n  Evaluating...")
    results = evaluate_predictions(learned_weights, true_weights, true_binary)

    print(f"\n  {'='*50}")
    print(f"  RESULTS")
    print(f"  {'='*50}")
    print(f"  Weight correlation:  {results['weight_correlation']:.4f}")
    print(f"  Weight RMSE:         {results['weight_rmse']:.4f}")
    print(f"  Sign accuracy:       {results['sign_accuracy']:.1%}")
    print(f"  AUC:                 {results['auc']:.4f}")
    print(f"  AP:                  {results['ap']:.4f}")
    print(f"  F1 (optimal thresh): {results['f1']:.4f}")
    print(f"  Precision:           {results['precision']:.4f}")
    print(f"  Recall:              {results['recall']:.4f}")

    # Visualize
    output_dir = os.path.join(os.path.dirname(session_dir), '..', 'perceptron_outputs')
    output_dir = os.path.normpath(output_dir)
    plot_results(learned_weights, true_weights, true_binary, connections,
                 neuron_positions, results, session_name, output_dir)

    # Save learned weights
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f'perceptron_weights_{session_name}.npz')
    np.savez_compressed(
        npz_path,
        learned_weights=learned_weights,
        true_weights=true_weights,
        true_binary=true_binary,
        results=results,
        session_name=session_name,
        n_iterations=n_iterations,
        lr=lr,
        threshold=threshold,
        delay_ms=delay_ms
    )
    print(f"  Weights saved: {npz_path}")

    return results, learned_weights


def select_session():
    """Interactive session selection."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIF data")
    sessions = sorted(glob.glob(os.path.join(data_dir, "*")))
    sessions = [s for s in sessions if os.path.isdir(s)]

    if not sessions:
        print("No sessions found in LIF data/")
        sys.exit(1)

    print("\nAvailable sessions:")
    for i, s in enumerate(sessions):
        name = os.path.basename(s)
        net_files = glob.glob(os.path.join(s, 'network_*.npz'))
        rec_files = glob.glob(os.path.join(s, 'recording[0-9][0-9][0-9].npz'))
        print(f"  [{i}] {name}  ({len(rec_files)} recordings)")

    print(f"\n  [Enter] Use first: {os.path.basename(sessions[0])}")
    choice = input("Select: ").strip()

    if choice == '':
        return sessions[0]
    return sessions[int(choice)]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        session_dir = sys.argv[1]
    else:
        session_dir = select_session()

    results, weights = run_inference(
        session_dir,
        recording_idx=0,
        n_iterations=50,
        lr=0.01,
        threshold=20.0,
        delay_ms=1
    )
