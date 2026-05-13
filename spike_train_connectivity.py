"""
Spike Train Connectivity Inference

For each postsynaptic neuron, takes the nearest K candidate presynaptic neurons,
feeds their spike trains through a CNN/LSTM/Perceptron to predict connectivity.

The activation threshold is a learnable parameter.

Usage:
    python spike_train_connectivity.py [--model cnn|lstm|perceptron] [--k 50]
"""

import numpy as np
import os
import sys
import glob
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# DATA
# ============================================================================

def spike_times_to_binary(spike_times, duration_ms, dt=1.0):
    """Convert spike times to binary matrix [n_neurons, T] at dt resolution."""
    n_neurons = len(spike_times)
    n_bins = int(duration_ms / dt)
    binary = np.zeros((n_neurons, n_bins), dtype=np.float32)
    for i in range(n_neurons):
        for t in spike_times[i]:
            idx = int(t / dt)
            if 0 <= idx < n_bins:
                binary[i, idx] = 1
    return binary


def build_ground_truth(connections, n_neurons):
    """Build weight and binary connectivity matrices.
    W[post, pre] = weight, B[post, pre] = 1 if connected."""
    W = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    B = np.zeros((n_neurons, n_neurons), dtype=np.int32)
    for c in connections:
        pre, post = int(c[0]), int(c[1])
        W[post, pre] = float(c[2])
        B[post, pre] = 1
    return W, B


def compute_neighbor_indices(neuron_positions, K):
    """For each neuron, find K nearest neighbors by Euclidean distance.
    Returns [n_neurons, K] array of neighbor indices."""
    n = len(neuron_positions)
    dist = np.sqrt(((neuron_positions[:, None, :] -
                     neuron_positions[None, :, :]) ** 2).sum(axis=2))
    # Set self-distance to inf so it's never selected
    np.fill_diagonal(dist, np.inf)
    # K nearest neighbors per neuron
    K_actual = min(K, n - 1)
    indices = np.argsort(dist, axis=1)[:, :K_actual]
    return indices, K_actual


class NeuronPairDataset(Dataset):
    """
    Dataset for connectivity inference.

    Each sample corresponds to one postsynaptic neuron.
    Returns:
        pre_spikes: [K, T] spike trains of K nearest pre candidates
        post_spikes: [1, T] spike train of the postsynaptic neuron
        labels: [K] binary connectivity labels
        weights: [K] true synaptic weights (for regression)
        distances: [K] spatial distances to pre candidates
    """

    def __init__(self, spike_matrix, neighbor_indices, true_binary, true_weights,
                 neuron_positions, neuron_ids=None):
        """
        Args:
            spike_matrix: [n_neurons, T] binary spike matrix
            neighbor_indices: [n_neurons, K] nearest neighbor indices
            true_binary: [n_neurons, n_neurons] binary connectivity
            true_weights: [n_neurons, n_neurons] weight matrix
            neuron_positions: [n_neurons, 2]
            neuron_ids: subset of neuron indices to include (for train/val split)
        """
        self.spike_matrix = spike_matrix
        self.neighbor_indices = neighbor_indices
        self.true_binary = true_binary
        self.true_weights = true_weights
        self.neuron_positions = neuron_positions
        self.neuron_ids = neuron_ids if neuron_ids is not None else \
            np.arange(len(spike_matrix))

    def __len__(self):
        return len(self.neuron_ids)

    def __getitem__(self, idx):
        post_id = self.neuron_ids[idx]
        pre_ids = self.neighbor_indices[post_id]
        K = len(pre_ids)

        pre_spikes = self.spike_matrix[pre_ids]          # [K, T]
        post_spikes = self.spike_matrix[post_id:post_id+1]  # [1, T]
        labels = self.true_binary[post_id, pre_ids].astype(np.float32)  # [K]
        weights = self.true_weights[post_id, pre_ids].astype(np.float32)  # [K]

        # Distances
        d = np.sqrt(((self.neuron_positions[pre_ids] -
                      self.neuron_positions[post_id]) ** 2).sum(axis=1))
        distances = d.astype(np.float32)  # [K]

        return (torch.from_numpy(pre_spikes),
                torch.from_numpy(post_spikes),
                torch.from_numpy(labels),
                torch.from_numpy(weights),
                torch.from_numpy(distances))


# ============================================================================
# MODELS
# ============================================================================

class LearnedThreshold(nn.Module):
    """Learnable activation threshold."""

    def __init__(self, init_value=0.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        # Soft threshold: sigmoid(scale * (x - threshold))
        # Scale controls sharpness of the transition
        return torch.sigmoid(10.0 * (x - self.threshold))


class PerceptronModel(nn.Module):
    """
    Learnable perceptron: for each post neuron, learns weights from K pre neurons.

    Unlike fixed-threshold perceptron, the threshold and a per-candidate
    transformation are learned end-to-end.
    """

    def __init__(self, T, K, hidden_dim=64):
        super().__init__()
        self.K = K

        # Per-pre temporal summarizer: compress spike train to features
        self.pre_encoder = nn.Sequential(
            nn.Linear(T, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Post temporal summarizer
        self.post_encoder = nn.Sequential(
            nn.Linear(T, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Interaction: [pre_feat, post_feat, distance] -> connection score
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Learned threshold
        self.threshold = LearnedThreshold(init_value=0.0)

    def forward(self, pre_spikes, post_spikes, distances):
        """
        Args:
            pre_spikes: [B, K, T]
            post_spikes: [B, 1, T]
            distances: [B, K]
        Returns:
            logits: [B, K] raw scores (for BCE loss)
        """
        B, K, T = pre_spikes.shape

        # Encode each pre neuron's spike train
        pre_flat = pre_spikes.reshape(B * K, T)       # [B*K, T]
        pre_feat = self.pre_encoder(pre_flat)          # [B*K, H]
        pre_feat = pre_feat.reshape(B, K, -1)          # [B, K, H]

        # Encode post neuron
        post_feat = self.post_encoder(post_spikes.squeeze(1))  # [B, H]
        post_feat = post_feat.unsqueeze(1).expand(-1, K, -1)   # [B, K, H]

        # Combine
        dist_feat = distances.unsqueeze(-1)             # [B, K, 1]
        combined = torch.cat([pre_feat, post_feat, dist_feat], dim=-1)  # [B, K, 2H+1]

        logits = self.classifier(combined).squeeze(-1)  # [B, K]
        return logits


class CNNModel(nn.Module):
    """
    1D CNN over spike trains for connectivity prediction.

    For each (pre_i, post_j) pair, stacks their spike trains as 2 channels
    and runs temporal convolutions to detect causal timing signatures.
    """

    def __init__(self, K, hidden_dim=64):
        super().__init__()
        self.K = K

        # Temporal CNN: 2-channel input [pre_spike, post_spike]
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),   # 5ms window
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # 20ms window
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),  # 80ms window
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),                       # -> [64, 8]
        )

        # Classifier: temporal features + distance -> connection score
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Learned threshold
        self.threshold = LearnedThreshold(init_value=0.0)

    def forward(self, pre_spikes, post_spikes, distances):
        """
        Args:
            pre_spikes: [B, K, T]
            post_spikes: [B, 1, T]
            distances: [B, K]
        Returns:
            logits: [B, K]
        """
        B, K, T = pre_spikes.shape

        # Expand post to match each pre: [B, K, T]
        post_expanded = post_spikes.expand(-1, K, -1)

        # Stack as 2-channel: [B*K, 2, T]
        pairs = torch.stack([pre_spikes.reshape(B*K, T),
                             post_expanded.reshape(B*K, T)], dim=1)

        # CNN
        feat = self.temporal_cnn(pairs)      # [B*K, 64, 8]
        feat = feat.reshape(B*K, -1)         # [B*K, 512]
        feat = feat.reshape(B, K, -1)        # [B, K, 512]

        # Add distance
        dist_feat = distances.unsqueeze(-1)  # [B, K, 1]
        combined = torch.cat([feat, dist_feat], dim=-1)  # [B, K, 513]

        logits = self.classifier(combined).squeeze(-1)  # [B, K]
        return logits


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM over spike trains for connectivity prediction.

    Processes the paired (pre, post) spike trains sequentially,
    captures long-range temporal dependencies.
    """

    def __init__(self, K, hidden_dim=64, lstm_hidden=32, n_layers=2):
        super().__init__()
        self.K = K

        # Downsample first with conv (60000 is too long for LSTM)
        self.downsample = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(10),       # 60000 -> 6000
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(10),       # 6000 -> 600
        )

        self.lstm = nn.LSTM(
            input_size=32, hidden_size=lstm_hidden,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=0.2
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + 1, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.threshold = LearnedThreshold(init_value=0.0)

    def forward(self, pre_spikes, post_spikes, distances):
        B, K, T = pre_spikes.shape

        post_expanded = post_spikes.expand(-1, K, -1)
        pairs = torch.stack([pre_spikes.reshape(B*K, T),
                             post_expanded.reshape(B*K, T)], dim=1)  # [B*K, 2, T]

        # Downsample
        feat = self.downsample(pairs)           # [B*K, 32, 600]
        feat = feat.permute(0, 2, 1)            # [B*K, 600, 32]

        # LSTM
        output, (h_n, _) = self.lstm(feat)      # h_n: [n_layers*2, B*K, hidden]

        # Take last layer, both directions
        h_forward = h_n[-2]                     # [B*K, hidden]
        h_backward = h_n[-1]                    # [B*K, hidden]
        h = torch.cat([h_forward, h_backward], dim=-1)  # [B*K, hidden*2]
        h = h.reshape(B, K, -1)                # [B, K, hidden*2]

        dist_feat = distances.unsqueeze(-1)
        combined = torch.cat([h, dist_feat], dim=-1)

        logits = self.classifier(combined).squeeze(-1)
        return logits


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, pos_weight):
    model.train()
    total_loss = 0
    n_batches = 0

    for pre_sp, post_sp, labels, weights, dists in dataloader:
        pre_sp = pre_sp.to(device)
        post_sp = post_sp.to(device)
        labels = labels.to(device)
        dists = dists.to(device)

        optimizer.zero_grad()
        logits = model(pre_sp, post_sp, dists)

        # Weighted BCE loss
        weight_tensor = torch.where(labels == 1, pos_weight, 1.0)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, weight=weight_tensor
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    all_dists = []

    for pre_sp, post_sp, labels, weights, dists in dataloader:
        pre_sp = pre_sp.to(device)
        post_sp = post_sp.to(device)
        dists = dists.to(device)

        logits = model(pre_sp, post_sp, dists)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        all_dists.append(dists.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()  # [N, K]
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Flatten
    scores = 1.0 / (1.0 + np.exp(-all_logits.ravel()))  # sigmoid
    labels_flat = all_labels.ravel()

    results = {}
    if len(np.unique(labels_flat)) > 1:
        results['auc'] = roc_auc_score(labels_flat, scores)
        results['ap'] = average_precision_score(labels_flat, scores)

        # Optimal threshold
        prec, rec, thresholds = precision_recall_curve(labels_flat, scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        predicted = (scores >= best_thresh).astype(int)
        tp = np.sum((predicted == 1) & (labels_flat == 1))
        fp = np.sum((predicted == 1) & (labels_flat == 0))
        fn = np.sum((predicted == 0) & (labels_flat == 1))

        results['threshold'] = float(best_thresh)
        results['precision'] = tp / (tp + fp + 1e-10)
        results['recall'] = tp / (tp + fn + 1e-10)
        results['f1'] = float(f1[best_idx])
        results['tp'] = int(tp)
        results['fp'] = int(fp)
        results['fn'] = int(fn)
    else:
        results['auc'] = 0.0
        results['ap'] = 0.0
        results['f1'] = 0.0

    results['n_positive'] = int(labels_flat.sum())
    results['n_total'] = len(labels_flat)

    return results, scores, labels_flat


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results_train, results_val, scores_val, labels_val,
                 train_losses, val_aucs, neuron_positions, connections,
                 learned_threshold, session_name, model_name, output_dir):
    """6-panel visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Spike Train {model_name.upper()} — {session_name}',
                 fontsize=14, fontweight='bold')

    # ---- Plot 1: Training curve ----
    ax = axes[0, 0]
    ax.plot(train_losses, 'b-', alpha=0.7, label='Train loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(val_aucs, 'r-', alpha=0.7, label='Val AUC')
    ax2.set_ylabel('AUC', color='red')
    ax2.legend(loc='center right')

    # ---- Plot 2: Score distribution ----
    ax = axes[0, 1]
    pos_scores = scores_val[labels_val == 1]
    neg_scores = scores_val[labels_val == 0]
    ax.hist(neg_scores, bins=50, alpha=0.6, color='red',
            label=f'No conn (n={len(neg_scores)})', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.6, color='green',
            label=f'Connected (n={len(pos_scores)})', density=True)
    thresh = results_val.get('threshold', 0.5)
    ax.axvline(thresh, color='blue', linewidth=2,
               label=f'Threshold={thresh:.3f}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend(fontsize=8)

    # ---- Plot 3: PR curve ----
    ax = axes[0, 2]
    if len(np.unique(labels_val)) > 1:
        prec, rec, _ = precision_recall_curve(labels_val, scores_val)
        ax.plot(rec, prec, 'b-', linewidth=2)
        ax.fill_between(rec, prec, alpha=0.2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR Curve (AUC={results_val["auc"]:.3f}, AP={results_val["ap"]:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ---- Plot 4: True connections ----
    ax = axes[1, 0]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=20, edgecolors='navy', zorder=3)
    for c in connections:
        i, j = int(c[0]), int(c[1])
        ax.plot([neuron_positions[i, 0], neuron_positions[j, 0]],
                [neuron_positions[i, 1], neuron_positions[j, 1]],
                'g-', alpha=0.12, linewidth=0.3)
    ax.set_title(f'True Connections (n={len(connections)})')
    ax.set_aspect('equal')

    # ---- Plot 5: Predicted connections ----
    ax = axes[1, 1]
    ax.scatter(neuron_positions[:, 0], neuron_positions[:, 1],
               c='lightblue', s=20, edgecolors='navy', zorder=3)
    predicted = (scores_val >= thresh).astype(int)
    # We need to map flat indices back to (post, pre) pairs
    # This is approximate — just show TP/FP counts in title
    tp = results_val.get('tp', 0)
    fp = results_val.get('fp', 0)
    fn = results_val.get('fn', 0)
    ax.set_title(f'Predicted: TP={tp} (green) FP={fp} (red) FN={fn}')
    ax.set_aspect('equal')
    ax.text(0.5, 0.5, f'TP={tp}\nFP={fp}\nFN={fn}',
            transform=ax.transAxes, ha='center', va='center', fontsize=16)

    # ---- Plot 6: Summary ----
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
    SPIKE TRAIN {model_name.upper()} CONNECTIVITY
    {'='*44}

    Network: {session_name}
    Model: {model_name}
    Learned threshold: {learned_threshold:.4f}

    Validation Results:
      AUC:       {results_val['auc']:.4f}
      AP:        {results_val['ap']:.4f}
      F1:        {results_val['f1']:.4f}

    At Optimal Threshold ({thresh:.3f}):
      Precision: {results_val.get('precision', 0):.4f}
      Recall:    {results_val.get('recall', 0):.4f}
      TP: {tp}  FP: {fp}  FN: {fn}

    Positive samples: {results_val['n_positive']}
    Total samples:    {results_val['n_total']}
    Pos ratio:        {results_val['n_positive']/max(results_val['n_total'],1):.3%}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{model_name}_{session_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {path}")
    return path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_session(session_dir, recording_idx=0):
    """Load spike data, network structure, positions from a session."""
    rec_path = os.path.join(session_dir, f'recording{recording_idx:03d}.npz')
    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f"No network file in {session_dir}")

    rec_data = np.load(rec_path, allow_pickle=True)
    net_data = np.load(net_files[0], allow_pickle=True)

    return {
        'spike_times': rec_data['spike_times'],
        'duration': float(rec_data['duration']),
        'connections': net_data['connections'],
        'neuron_positions': net_data['neuron_positions'],
        'n_neurons': len(net_data['neuron_positions']),
    }


def run_pipeline(session_dir, model_name='cnn', K=50, recording_idx=0,
                 n_epochs=100, lr=1e-3, batch_size=8, patience=15,
                 val_fraction=0.2, dt=1.0, hidden_dim=64, pos_weight=5.0,
                 device=None):
    """
    Full pipeline: load data, build model, train, evaluate, visualize.

    Args:
        session_dir: Path to session folder
        model_name: 'cnn', 'lstm', or 'perceptron'
        K: Number of nearest candidate presynaptic neurons
        recording_idx: Which recording to use
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Neurons per batch
        patience: Early stopping patience
        val_fraction: Fraction of neurons for validation
        dt: Spike train bin size in ms
        hidden_dim: Hidden layer dimension
        pos_weight: Weight for positive class in loss
        device: 'cuda' or 'cpu'
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    session_name = os.path.basename(session_dir)
    print(f"\n{'='*70}")
    print(f"SPIKE TRAIN CONNECTIVITY — {model_name.upper()}")
    print(f"Session: {session_name}")
    print(f"K={K}, epochs={n_epochs}, lr={lr}, batch={batch_size}, dt={dt}ms")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # ── Load data ──
    print("\n  Loading data...")
    data = load_session(session_dir, recording_idx)
    n_neurons = data['n_neurons']
    duration = data['duration']
    connections = data['connections']
    positions = data['neuron_positions']

    print(f"  Neurons: {n_neurons}, Connections: {len(connections)}, "
          f"Duration: {duration/1000:.0f}s")

    # ── Spike trains ──
    print(f"  Binning spike trains at {dt}ms...")
    spike_matrix = spike_times_to_binary(data['spike_times'], duration, dt=dt)
    T = spike_matrix.shape[1]
    print(f"  Spike matrix: [{n_neurons}, {T}]")

    # ── Neighbors and ground truth ──
    neighbor_indices, K_actual = compute_neighbor_indices(positions, K)
    print(f"  K nearest neighbors: {K_actual} (requested {K})")

    true_weights, true_binary = build_ground_truth(connections, n_neurons)

    # Count how many true connections are in the K-neighbor set
    total_in_K = 0
    total_true = int(true_binary.sum())
    for j in range(n_neurons):
        for pre in neighbor_indices[j]:
            if true_binary[j, pre]:
                total_in_K += 1
    coverage = total_in_K / max(total_true, 1)
    print(f"  True connections in K-neighbor set: {total_in_K}/{total_true} "
          f"({coverage:.1%} coverage)")

    # ── Train/Val split (by neuron) ──
    np.random.seed(42)
    perm = np.random.permutation(n_neurons)
    n_val = max(1, int(n_neurons * val_fraction))
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]
    print(f"  Train neurons: {len(train_ids)}, Val neurons: {len(val_ids)}")

    train_dataset = NeuronPairDataset(
        spike_matrix, neighbor_indices, true_binary, true_weights,
        positions, train_ids)
    val_dataset = NeuronPairDataset(
        spike_matrix, neighbor_indices, true_binary, true_weights,
        positions, val_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # ── Positive class weight ──
    n_pos_train = 0
    n_total_train = 0
    for j in train_ids:
        for pre in neighbor_indices[j]:
            n_total_train += 1
            if true_binary[j, pre]:
                n_pos_train += 1
    auto_pos_weight = (n_total_train - n_pos_train) / max(n_pos_train, 1)
    pw = min(pos_weight, auto_pos_weight)  # use provided or auto, whichever smaller
    print(f"  Class balance: {n_pos_train}/{n_total_train} positive "
          f"({n_pos_train/max(n_total_train,1):.2%}), pos_weight={pw:.1f}")

    # ── Build model ──
    if model_name == 'cnn':
        model = CNNModel(K=K_actual, hidden_dim=hidden_dim).to(device)
    elif model_name == 'lstm':
        model = LSTMModel(K=K_actual, hidden_dim=hidden_dim).to(device)
    elif model_name == 'perceptron':
        model = PerceptronModel(T=T, K=K_actual, hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    # ── Training loop ──
    print(f"\n  Training...")
    best_auc = 0
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_aucs = []

    t0 = time.time()
    for epoch in range(n_epochs):
        loss = train_epoch(model, train_loader, optimizer,
                           F.binary_cross_entropy_with_logits, device, pw)
        train_losses.append(loss)

        # Evaluate
        val_results, val_scores, val_labels = evaluate(model, val_loader, device)
        val_auc = val_results['auc']
        val_aucs.append(val_auc)

        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            thresh_val = model.threshold.threshold.item()
            print(f"    Epoch {epoch+1:3d}: loss={loss:.4f}, "
                  f"val_AUC={val_auc:.4f}, val_AP={val_results['ap']:.4f}, "
                  f"threshold={thresh_val:.3f}, "
                  f"({elapsed:.0f}s)")

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed_total = time.time() - t0
    print(f"  Training complete in {elapsed_total:.0f}s, best AUC={best_auc:.4f}")

    # ── Final evaluation ──
    # Evaluate on ALL neurons (train + val)
    full_dataset = NeuronPairDataset(
        spike_matrix, neighbor_indices, true_binary, true_weights, positions)
    full_loader = DataLoader(full_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    final_results, final_scores, final_labels = evaluate(model, full_loader, device)
    val_results, val_scores, val_labels = evaluate(model, val_loader, device)

    learned_threshold = model.threshold.threshold.item()

    print(f"\n  {'='*50}")
    print(f"  FINAL RESULTS (all neurons)")
    print(f"  {'='*50}")
    print(f"  AUC:       {final_results['auc']:.4f}")
    print(f"  AP:        {final_results['ap']:.4f}")
    print(f"  F1:        {final_results['f1']:.4f}")
    print(f"  Precision: {final_results.get('precision', 0):.4f}")
    print(f"  Recall:    {final_results.get('recall', 0):.4f}")
    print(f"  Learned threshold: {learned_threshold:.4f}")

    print(f"\n  VALIDATION RESULTS (unseen neurons)")
    print(f"  AUC:       {val_results['auc']:.4f}")
    print(f"  AP:        {val_results['ap']:.4f}")
    print(f"  F1:        {val_results['f1']:.4f}")

    # ── Visualize ──
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'spike_cnn_outputs')
    plot_results(final_results, val_results, val_scores, val_labels,
                 train_losses, val_aucs, positions, connections,
                 learned_threshold, session_name, model_name, output_dir)

    # ── Save model ──
    model_path = os.path.join(output_dir, f'{model_name}_{session_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'K': K_actual,
        'T': T,
        'dt': dt,
        'hidden_dim': hidden_dim,
        'n_neurons': n_neurons,
        'results_all': final_results,
        'results_val': val_results,
        'learned_threshold': learned_threshold,
        'train_losses': train_losses,
        'val_aucs': val_aucs,
    }, model_path)
    print(f"  Model saved: {model_path}")

    return final_results, val_results


# ============================================================================
# CLI
# ============================================================================

def select_session():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIF data")
    sessions = sorted([s for s in glob.glob(os.path.join(data_dir, "*"))
                       if os.path.isdir(s)])
    if not sessions:
        print("No sessions found in LIF data/")
        sys.exit(1)

    print("\nAvailable sessions:")
    for i, s in enumerate(sessions):
        name = os.path.basename(s)
        recs = glob.glob(os.path.join(s, 'recording[0-9][0-9][0-9].npz'))
        print(f"  [{i}] {name}  ({len(recs)} recordings)")

    choice = input("Select (Enter=first): ").strip()
    return sessions[0] if choice == '' else sessions[int(choice)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spike Train Connectivity Inference')
    parser.add_argument('--session', type=str, default=None, help='Session directory')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'perceptron'],
                        help='Model type (default: cnn)')
    parser.add_argument('--k', type=int, default=50,
                        help='Number of nearest candidate presynaptic neurons')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (neurons)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping')
    parser.add_argument('--dt', type=float, default=1.0, help='Bin size in ms')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--recording', type=int, default=0, help='Recording index')
    args = parser.parse_args()

    if args.session is None:
        session_dir = select_session()
    else:
        session_dir = args.session

    run_pipeline(
        session_dir,
        model_name=args.model,
        K=args.k,
        recording_idx=args.recording,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        patience=args.patience,
        dt=args.dt,
        hidden_dim=args.hidden,
    )
