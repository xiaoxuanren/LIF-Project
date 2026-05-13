"""
GNN Cross-Network Connectivity Prediction
==========================================

Train on multiple networks, predict on completely new networks.
This tests true generalization - can the model learn universal
patterns of how connectivity relates to activity?

Key Challenge: Different networks have different:
- Number of neurons
- Number of clusters
- Spatial arrangements
- Connection patterns

Solution: Use INDUCTIVE learning with features that generalize:
- Relative features (not absolute positions)
- Statistical features (firing rates, correlations)
- The GNN learns patterns, not specific connections
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import json
import glob
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_output_dirs(base_dir: str = "gnn_outputs") -> Dict[str, str]:
    """
    Create output directory structure for saving models, results, and figures.

    Args:
        base_dir: Base directory name for outputs (relative to current working directory)

    Returns:
        Dictionary with paths to each subdirectory
    """
    dirs = {
        'base': base_dir,
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results'),
        'figures': os.path.join(base_dir, 'figures')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def compute_transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
    """
    Compute transfer entropy from source to target.
    TE(X->Y) measures how much knowing X's past reduces uncertainty about Y's future.
    Higher TE indicates potential causal influence.
    
    Args:
        source: Source signal (binned spikes)
        target: Target signal (binned spikes)
        lag: Time lag for computing TE
        
    Returns:
        Transfer entropy value (non-negative)
    """
    if len(source) < lag + 2 or len(target) < lag + 2:
        return 0.0
    
    # Binarize signals (spike present or not)
    src = (source > 0).astype(int)
    tgt = (target > 0).astype(int)
    
    # Future of target, past of target, past of source
    y_future = tgt[lag:]
    y_past = tgt[:-lag]
    x_past = src[:-lag]
    
    # Compute joint and marginal probabilities
    n = len(y_future)
    eps = 1e-10
    
    # Count occurrences
    counts = {}
    for yf, yp, xp in zip(y_future, y_past, x_past):
        key = (yf, yp, xp)
        counts[key] = counts.get(key, 0) + 1
    
    # Marginal counts
    count_yp_xp = {}  # P(y_past, x_past)
    count_yp = {}     # P(y_past)
    count_yf_yp = {}  # P(y_future, y_past)
    
    for yf, yp, xp in zip(y_future, y_past, x_past):
        count_yp_xp[(yp, xp)] = count_yp_xp.get((yp, xp), 0) + 1
        count_yp[yp] = count_yp.get(yp, 0) + 1
        count_yf_yp[(yf, yp)] = count_yf_yp.get((yf, yp), 0) + 1
    
    # Compute transfer entropy
    te = 0.0
    for (yf, yp, xp), count in counts.items():
        p_joint = count / n
        p_yp_xp = count_yp_xp.get((yp, xp), eps) / n
        p_yp = count_yp.get(yp, eps) / n
        p_yf_yp = count_yf_yp.get((yf, yp), eps) / n
        
        p_yf_given_yp_xp = p_joint / (p_yp_xp + eps)
        p_yf_given_yp = p_yf_yp / (p_yp + eps)
        
        if p_yf_given_yp_xp > eps and p_yf_given_yp > eps:
            te += p_joint * np.log2(p_yf_given_yp_xp / (p_yf_given_yp + eps) + eps)
    
    return max(0, te)  # TE should be non-negative


# ============================================================================
# PART 1: CONFIGURATION
# ============================================================================

@dataclass
class CrossNetworkConfig:
    """Configuration for cross-network prediction"""
    # Feature extraction
    bin_size_ms: float = 5.0  # Finer resolution to capture synaptic timing (was 20ms)
    correlation_lags: Optional[List[int]] = None
    
    # Spatial parameters (for real data without cluster info)
    connection_radius: float = 2.0  # Max distance to consider connections
    neighbor_radius: float = 1.0    # Radius for computing local neighborhood features

    # Sliding window parameters (for scalable prediction)
    use_sliding_window: bool = True   # Use sliding window for prediction
    window_size: float = 7.0          # Size of sliding window
    window_overlap: float = 0.5       # Overlap ratio between windows (0-1)
    edges_per_batch: int = 2000       # Max edges to predict per batch

    # Sampling
    neg_ratio: float = 3.0
    max_edges_per_graph: int = 5000  # Limit for memory
    filter_by_radius: bool = True    # Only consider edges within connection_radius
    
    # Hard negative mining
    hard_negative_ratio: float = 0.5  # 50% of negatives are "hard" (balanced approach)
    hard_negative_radius: float = 1.0  # Radius for hard negatives (closer = harder)

    # Model
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    conv_type: str = 'SAGE'  # SAGE works better for inductive learning

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 300
    patience: int = 30
    batch_size: int = 1  # Graphs per batch (1 for different sized graphs)
    pos_weight: float = 5.0

    def __post_init__(self):
        if self.correlation_lags is None:
            # Finer lags: at 5ms bins, covers -20ms to +20ms
            # This captures both fast synaptic effects and slower correlations
            self.correlation_lags = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


# ============================================================================
# PART 2: FEATURE EXTRACTION (Generalizable Features)
# ============================================================================

class GeneralizableFeatureExtractor:
    """
    Extract features that generalize across different networks

    Key principle: Use RELATIVE and STATISTICAL features, not absolute ones
    """

    def __init__(self, config: CrossNetworkConfig):
        self.config = config
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.fitted = False

    def compute_binned_spikes(self, spike_times: List, duration_ms: float) -> np.ndarray:
        """Convert spike times to binned spike trains"""
        n_neurons = len(spike_times)
        n_bins = int(duration_ms / self.config.bin_size_ms)

        binned = np.zeros((n_neurons, n_bins))
        for i, spikes in enumerate(spike_times):
            for spike in spikes:
                bin_idx = int(spike / self.config.bin_size_ms)
                if bin_idx < n_bins:
                    binned[i, bin_idx] = 1
        return binned

    def extract_node_features(self, spike_times: List, duration_ms: float,
                             neuron_positions: np.ndarray,
                             fit_scaler: bool = False) -> np.ndarray:
        """
        Extract generalizable node features (no cluster info needed)

        Features (all relative/statistical):
        - Firing rate (Hz) - normalized
        - Spike count - normalized
        - Mean ISI - normalized
        - CV of ISI
        - Burstiness (fraction of short ISIs)
        - Local density (neurons within neighbor_radius)
        - Avg distance to neighbors
        - Neighbor firing rate mean
        - Neighbor firing rate std
        """
        n_neurons = len(spike_times)
        duration_s = duration_ms / 1000.0
        radius = self.config.neighbor_radius

        # Precompute all firing rates
        all_firing_rates = np.array([
            len(list(s)) / duration_s for s in spike_times
        ])

        # Precompute distance matrix
        dist_matrix = np.sqrt(
            np.sum((neuron_positions[:, np.newaxis, :] - 
                    neuron_positions[np.newaxis, :, :]) ** 2, axis=2)
        )

        features = []

        for i, spikes in enumerate(spike_times):
            spikes = np.array(list(spikes)) if not isinstance(spikes, np.ndarray) else spikes
            n_spikes = len(spikes)

            # Basic firing stats
            firing_rate = n_spikes / duration_s

            if n_spikes > 1:
                isis = np.diff(spikes)
                mean_isi = np.mean(isis)
                cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else 0
                # Burstiness: fraction of ISIs < 10ms
                burstiness = np.mean(isis < 10) if len(isis) > 0 else 0
            else:
                mean_isi = 0
                cv_isi = 0
                burstiness = 0

            # Spatial neighborhood features (no cluster needed)
            distances = dist_matrix[i]
            neighbor_mask = (distances < radius) & (distances > 0)  # Exclude self
            n_neighbors = np.sum(neighbor_mask)

            if n_neighbors > 0:
                neighbor_distances = distances[neighbor_mask]
                avg_neighbor_dist = np.mean(neighbor_distances)
                neighbor_fr = all_firing_rates[neighbor_mask]
                neighbor_fr_mean = np.mean(neighbor_fr)
                neighbor_fr_std = np.std(neighbor_fr)
            else:
                avg_neighbor_dist = radius  # Max distance if no neighbors
                neighbor_fr_mean = 0
                neighbor_fr_std = 0

            features.append([
                firing_rate,
                n_spikes,
                mean_isi,
                cv_isi,
                burstiness,
                n_neighbors,
                avg_neighbor_dist,
                neighbor_fr_mean,
                neighbor_fr_std
            ])

        features = np.array(features)

        # Normalize features
        if fit_scaler:
            self.node_scaler.fit(features)
            self.fitted = True

        if self.fitted:
            features = self.node_scaler.transform(features)

        return features.astype(np.float32)

    def extract_edge_features(self, spike_times: List, duration_ms: float,
                             neuron_positions: np.ndarray,
                             edge_index: np.ndarray,
                             fit_scaler: bool = False) -> np.ndarray:
        """
        Extract generalizable edge features (no cluster info needed)

        Features (all relative):
        - Euclidean distance
        - Normalized distance (dist / connection_radius)
        - Within radius (binary)
        - Correlation at multiple lags
        - Difference in firing rates
        - Product of firing rates
        """
        binned = self.compute_binned_spikes(spike_times, duration_ms)
        n_edges = edge_index.shape[1]
        duration_s = duration_ms / 1000.0
        conn_radius = self.config.connection_radius

        # Precompute firing rates
        firing_rates = np.array([len(list(s)) / duration_s for s in spike_times])

        features = []

        for k in range(n_edges):
            i, j = edge_index[0, k], edge_index[1, k]

            # Distance features
            dist = np.sqrt(np.sum((neuron_positions[i] - neuron_positions[j])**2))
            normalized_dist = dist / conn_radius  # Normalized by connection radius
            within_radius = float(dist <= conn_radius)  # Binary: within connection radius

            # Correlations at multiple lags
            correlations = []
            x, y = binned[i], binned[j]
            correlation_lags = self.config.correlation_lags if self.config.correlation_lags is not None else [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            for lag in correlation_lags:
                if lag == 0:
                    if np.std(x) > 0 and np.std(y) > 0:
                        corr = np.corrcoef(x, y)[0, 1]
                    else:
                        corr = 0
                elif lag > 0 and len(x) > lag:
                    if np.std(x[:-lag]) > 0 and np.std(y[lag:]) > 0:
                        corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                    else:
                        corr = 0
                elif lag < 0 and len(x) > -lag:
                    if np.std(x[-lag:]) > 0 and np.std(y[:lag]) > 0:
                        corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
                    else:
                        corr = 0
                else:
                    corr = 0
                correlations.append(corr if not np.isnan(corr) else 0)

            # Transfer entropy features (directional causality)
            te_i_to_j = compute_transfer_entropy(x, y, lag=1)
            te_j_to_i = compute_transfer_entropy(y, x, lag=1)
            te_asymmetry = te_i_to_j - te_j_to_i
            te_max = max(te_i_to_j, te_j_to_i)

            # Firing rate features
            fr_diff = abs(firing_rates[i] - firing_rates[j])
            fr_product = firing_rates[i] * firing_rates[j]

            features.append([dist, normalized_dist, within_radius] + correlations + 
                          [te_i_to_j, te_j_to_i, te_asymmetry, te_max, fr_diff, fr_product])

        features = np.array(features)
        
        # Store expected feature dimension for validation
        self._edge_feature_dim = features.shape[1] if len(features) > 0 else None

        # Normalize (except within_radius which is binary)
        if fit_scaler and len(features) > 0:
            # Don't normalize binary feature (index 2)
            cols_to_normalize = [0, 1] + list(range(3, features.shape[1]))
            self.edge_scaler.fit(features[:, cols_to_normalize])

        if self.fitted and len(features) > 0:
            cols_to_normalize = [0, 1] + list(range(3, features.shape[1]))
            features[:, cols_to_normalize] = self.edge_scaler.transform(features[:, cols_to_normalize])

        return features.astype(np.float32)


# ============================================================================
# PART 3: DATASET PREPARATION
# ============================================================================

class CrossNetworkDataset:
    """Prepare multiple networks for cross-network training"""

    def __init__(self, config: CrossNetworkConfig):
        self.config = config
        self.feature_extractor = GeneralizableFeatureExtractor(config)

    def load_network(self, metadata_file: str) -> Dict:
        """Load a single network's data (cluster_assignments optional)"""
        with open(metadata_file) as f:
            metadata = json.load(f)

        network_data = np.load(metadata['network_file'], allow_pickle=True)
        rec_file = metadata['recordings'][0]['file']
        rec_data = np.load(rec_file, allow_pickle=True)

        return {
            'metadata_file': metadata_file,
            'connections': network_data['connections'],
            'neuron_positions': network_data['neuron_positions'],
            'spike_times': rec_data['spike_times'],
            'duration_ms': metadata['recording_duration'],
            'n_neurons': len(network_data['neuron_positions'])
        }

    def create_graph_data(self, network_data: Dict,
                         fit_scaler: bool = False,
                         include_all_edges: bool = False) -> Dict:
        """
        Create graph data with positive and negative edges

        Args:
            network_data: Network data dictionary
            fit_scaler: Whether to fit the feature scalers
            include_all_edges: If True, include all possible edges (for prediction)
        """
        n_neurons = network_data['n_neurons']
        connections = network_data['connections']

        # Extract node features
        node_features = self.feature_extractor.extract_node_features(
            network_data['spike_times'],
            network_data['duration_ms'],
            network_data['neuron_positions'],
            fit_scaler=fit_scaler
        )

        # Get positive edges
        pos_edges = np.array([[int(c[0]), int(c[1])] for c in connections]).T
        pos_set = set(map(tuple, pos_edges.T))
        
        # Precompute distance matrix for radius filtering
        neuron_positions = network_data['neuron_positions']
        dist_matrix = np.sqrt(
            np.sum((neuron_positions[:, np.newaxis, :] - 
                    neuron_positions[np.newaxis, :, :]) ** 2, axis=2)
        )

        if include_all_edges:
            # For prediction: create edges (optionally filtered by radius)
            all_edges = []
            for i in range(n_neurons):
                for j in range(n_neurons):
                    if i != j:
                        # Optionally filter by connection radius
                        if self.config.filter_by_radius:
                            if dist_matrix[i, j] <= self.config.connection_radius:
                                all_edges.append([i, j])
                        else:
                            all_edges.append([i, j])
            all_edges = np.array(all_edges).T

            # Labels: 1 if edge exists, 0 otherwise
            labels = np.array([1 if (e[0], e[1]) in pos_set else 0
                              for e in all_edges.T])

            # Subsample if too many edges
            if all_edges.shape[1] > self.config.max_edges_per_graph:
                # Keep all positive, sample negatives
                pos_mask = labels == 1
                neg_mask = labels == 0

                n_pos = np.sum(pos_mask)
                n_neg_sample = min(np.sum(neg_mask),
                                  self.config.max_edges_per_graph - n_pos)

                neg_indices = np.where(neg_mask)[0]
                sampled_neg = np.random.choice(neg_indices, n_neg_sample, replace=False)

                keep_indices = np.concatenate([np.where(pos_mask)[0], sampled_neg])
                all_edges = all_edges[:, keep_indices]
                labels = labels[keep_indices]

            pred_edges = all_edges
            pred_labels = labels

        else:
            # For training: sample negative edges with HARD NEGATIVE MINING
            # Hard negatives = nearby neurons that are NOT connected (harder to classify)
            n_pos = pos_edges.shape[1]
            n_neg = int(n_pos * self.config.neg_ratio)
            
            # Split into hard and easy negatives
            n_hard = int(n_neg * self.config.hard_negative_ratio)  # 70% hard
            n_easy = n_neg - n_hard  # 30% random
            
            hard_radius = self.config.hard_negative_radius
            conn_radius = self.config.connection_radius
            
            neg_edges = []
            hard_neg_edges = []
            easy_neg_edges = []
            
            # First, collect all candidate hard negatives (nearby but unconnected)
            hard_candidates = []
            for i in range(n_neurons):
                for j in range(n_neurons):
                    if i != j and (i, j) not in pos_set:
                        d = dist_matrix[i, j]
                        if d <= hard_radius:  # Very close = hard negative
                            hard_candidates.append([i, j])
            
            # Sample hard negatives
            if len(hard_candidates) > 0:
                hard_candidates = np.array(hard_candidates)
                n_hard_sample = min(n_hard, len(hard_candidates))
                hard_indices = np.random.choice(len(hard_candidates), n_hard_sample, replace=False)
                hard_neg_edges = hard_candidates[hard_indices].tolist()
                # Add to pos_set to avoid duplicates
                for e in hard_neg_edges:
                    pos_set.add((e[0], e[1]))
            
            # Sample easy/medium negatives (within connection radius but not hard)
            max_attempts = n_easy * 100
            attempts = 0
            while len(easy_neg_edges) < n_easy and attempts < max_attempts:
                i = np.random.randint(0, n_neurons)
                j = np.random.randint(0, n_neurons)
                attempts += 1
                if i != j and (i, j) not in pos_set:
                    d = dist_matrix[i, j]
                    # Easy negatives: within connection radius but not too close
                    if self.config.filter_by_radius:
                        if hard_radius < d <= conn_radius:
                            easy_neg_edges.append([i, j])
                            pos_set.add((i, j))
                    else:
                        if d > hard_radius:  # Not hard
                            easy_neg_edges.append([i, j])
                            pos_set.add((i, j))
            
            # Combine hard and easy negatives
            neg_edges = hard_neg_edges + easy_neg_edges
            
            if len(neg_edges) == 0:
                raise ValueError("Could not sample any negative edges. Try increasing connection_radius.")
            
            print(f"    Sampled negatives: {len(hard_neg_edges)} hard + {len(easy_neg_edges)} easy = {len(neg_edges)} total")
            neg_edges = np.array(neg_edges).T

            pred_edges = np.concatenate([pos_edges, neg_edges], axis=1)
            pred_labels = np.concatenate([
                np.ones(n_pos),
                np.zeros(len(neg_edges.T))
            ])

        # Extract edge features
        edge_features = self.feature_extractor.extract_edge_features(
            network_data['spike_times'],
            network_data['duration_ms'],
            network_data['neuron_positions'],
            pred_edges,
            fit_scaler=fit_scaler
        )

        # Create message passing edges (use positive edges only)
        # Make bidirectional
        mp_edges = np.concatenate([pos_edges, pos_edges[::-1]], axis=1)

        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(mp_edges, dtype=torch.long),
            'pred_edges': torch.tensor(pred_edges, dtype=torch.long),
            'pred_labels': torch.tensor(pred_labels, dtype=torch.float32),
            'edge_features': torch.tensor(edge_features, dtype=torch.float32),
            'n_neurons': n_neurons,
            'n_connections': len(connections)
        }

    def prepare_train_test_split(self, metadata_files: List[str],
                                 test_indices: List[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split networks into train and test sets

        Args:
            metadata_files: List of all metadata files
            test_indices: Indices of networks to use for testing

        Returns:
            train_graphs, test_graphs
        """
        train_files = [f for i, f in enumerate(metadata_files) if i not in test_indices]
        test_files = [f for i, f in enumerate(metadata_files) if i in test_indices]

        print(f"Train networks: {len(train_files)}")
        print(f"Test networks: {len(test_files)}")

        # Load and prepare training graphs
        train_graphs = []
        for i, f in enumerate(train_files):
            print(f"  Loading train network {i+1}/{len(train_files)}")
            network = self.load_network(f)
            graph = self.create_graph_data(network, fit_scaler=(i==0))
            train_graphs.append(graph)

        # Load and prepare test graphs (use all edges for evaluation)
        test_graphs = []
        for i, f in enumerate(test_files):
            print(f"  Loading test network {i+1}/{len(test_files)}")
            network = self.load_network(f)
            graph = self.create_graph_data(network, fit_scaler=False, include_all_edges=True)
            test_graphs.append(graph)

        return train_graphs, test_graphs

    def create_embedding_graph(self, network_data: Dict) -> Dict:
        """
        Create graph data for computing node embeddings only.
        Uses radius-based edges for message passing.
        
        Args:
            network_data: Network data dictionary
            
        Returns:
            Dictionary with node features and message passing edges
        """
        n_neurons = network_data['n_neurons']
        neuron_positions = network_data['neuron_positions']
        
        # Extract node features (don't fit scaler - use existing)
        node_features = self.feature_extractor.extract_node_features(
            network_data['spike_times'],
            network_data['duration_ms'],
            network_data['neuron_positions'],
            fit_scaler=False
        )
        
        # Create message passing edges based on spatial proximity
        # Use connection_radius for determining neighbors
        dist_matrix = np.sqrt(
            np.sum((neuron_positions[:, np.newaxis, :] - 
                    neuron_positions[np.newaxis, :, :]) ** 2, axis=2)
        )
        
        # Create edges for all pairs within radius (bidirectional)
        mp_edges = []
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and dist_matrix[i, j] <= self.config.connection_radius:
                    mp_edges.append([i, j])
        
        if len(mp_edges) == 0:
            # Fallback: connect each neuron to its k nearest neighbors
            k = min(10, n_neurons - 1)
            for i in range(n_neurons):
                nearest = np.argsort(dist_matrix[i])[1:k+1]  # Exclude self
                for j in nearest:
                    mp_edges.append([i, j])
                    mp_edges.append([j, i])
        
        mp_edges = np.array(mp_edges).T if mp_edges else np.zeros((2, 0), dtype=int)
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(mp_edges, dtype=torch.long),
            'neuron_positions': neuron_positions,
            'n_neurons': n_neurons
        }


# ============================================================================
# PART 4: INDUCTIVE GNN MODEL
# ============================================================================

class EdgePredictor(nn.Module):
    """
    Predict edges using node embeddings + edge features
    
    Simple MLP that combines node embeddings with edge features.
    Distance information is already included in edge_features.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        # Process edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combine node embeddings + edge features
        self.predictor = nn.Sequential(
            nn.Linear(node_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Predict edge probability

        Args:
            z_i: Source node embeddings [n_edges, dim]
            z_j: Target node embeddings [n_edges, dim]
            edge_features: Edge features [n_edges, edge_dim]
        """
        # Encode all edge features
        edge_enc = self.edge_encoder(edge_features)
        
        # Combine node embeddings + edge encoding
        combined = torch.cat([z_i, z_j, edge_enc], dim=-1)
        
        # Predict edge probability (logits)
        logits = self.predictor(combined).squeeze(-1)
        
        return logits


class InductiveGNN(nn.Module):
    """
    GNN that can generalize to unseen graphs

    Uses GraphSAGE which is naturally inductive (samples neighbors)
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()

        # Node feature projection
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        # GNN layers (SAGE is inductive)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

        # Edge predictor
        self.edge_predictor = EdgePredictor(hidden_dim, edge_dim, hidden_dim, dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings"""
        x = self.node_encoder(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                pred_edges: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [n_nodes, node_dim]
            edge_index: Message passing edges [2, n_mp_edges]
            pred_edges: Edges to predict [2, n_pred_edges]
            edge_features: Features for pred_edges [n_pred_edges, edge_dim]
        """
        # Get node embeddings
        z = self.encode(x, edge_index)

        # Predict edges
        z_i = z[pred_edges[0]]
        z_j = z[pred_edges[1]]

        return self.edge_predictor(z_i, z_j, edge_features)


# ============================================================================
# PART 5: TRAINING
# ============================================================================

class CrossNetworkTrainer:
    """Train GNN across multiple networks"""

    def __init__(self, model: InductiveGNN, config: CrossNetworkConfig,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config.pos_weight]).to(device)
        )

    def train_epoch(self, train_graphs: List[Dict]) -> float:
        """Train on all training graphs"""
        self.model.train()
        total_loss = 0

        # Shuffle graphs
        indices = np.random.permutation(len(train_graphs))

        for idx in indices:
            graph = train_graphs[idx]

            # Move to device
            x = graph['node_features'].to(self.device)
            edge_index = graph['edge_index'].to(self.device)
            pred_edges = graph['pred_edges'].to(self.device)
            edge_features = graph['edge_features'].to(self.device)
            labels = graph['pred_labels'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(x, edge_index, pred_edges, edge_features)
            loss = self.criterion(pred, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_graphs)

    @torch.no_grad()
    def evaluate(self, graphs: List[Dict]) -> Dict:
        """Evaluate on graphs"""
        self.model.eval()

        all_preds = []
        all_labels = []

        for graph in graphs:
            x = graph['node_features'].to(self.device)
            edge_index = graph['edge_index'].to(self.device)
            pred_edges = graph['pred_edges'].to(self.device)
            edge_features = graph['edge_features'].to(self.device)
            labels = graph['pred_labels']

            pred = self.model(x, edge_index, pred_edges, edge_features)
            pred_prob = torch.sigmoid(pred).cpu().numpy()

            all_preds.append(pred_prob)
            all_labels.append(labels.numpy())

        # Concatenate all predictions
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)

        # Precision at different recall levels
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        p_at_50 = precision[np.argmin(np.abs(recall - 0.5))]

        return {
            'auc': auc,
            'ap': ap,
            'precision_at_50_recall': p_at_50,
            'n_samples': len(all_labels),
            'n_positive': np.sum(all_labels)
        }

    def train(self, train_graphs: List[Dict], val_graphs: Optional[List[Dict]] = None) -> Dict:
        """Full training loop"""
        import time
        
        best_val_auc = 0
        best_epoch = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_auc': []}
        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Use subset of train for validation if no val provided
        if val_graphs is None:
            val_graphs = train_graphs[:1]  # Use first graph for validation

        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Train
            loss = self.train_epoch(train_graphs)
            history['train_loss'].append(loss)

            # Validate
            val_metrics = self.evaluate(val_graphs)
            history['val_auc'].append(val_metrics['auc'])

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_per_epoch = elapsed / (epoch + 1)
                remaining = avg_per_epoch * (self.config.num_epochs - epoch - 1)
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: Loss={loss:.4f}, Val AUC={val_metrics['auc']:.4f}, "
                      f"Best={best_val_auc:.4f}, Patience={patience_counter}/{self.config.patience}, "
                      f"Time: {elapsed:.0f}s, ETA: {remaining:.0f}s")

            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Restore best model
        self.model.load_state_dict(best_state)

        return {
            'history': history,
            'best_epoch': best_epoch,
            'best_val_auc': best_val_auc
        }


# ============================================================================
# PART 6: MAIN PIPELINE
# ============================================================================

def run_cross_network_pipeline(metadata_files: List[str],
                               test_indices: List[int],
                               config: Optional[CrossNetworkConfig] = None) -> Dict:
    """
    Train on some networks, test on others

    Args:
        metadata_files: List of all simulation metadata files
        test_indices: Indices of networks to hold out for testing
        config: Pipeline configuration
    """
    if config is None:
        config = CrossNetworkConfig()

    print("="*70)
    print("CROSS-NETWORK GNN CONNECTIVITY PREDICTION")
    print("="*70)

    # Prepare data
    print("\n[1/4] Preparing datasets...")
    dataset = CrossNetworkDataset(config)
    train_graphs, test_graphs = dataset.prepare_train_test_split(
        metadata_files, test_indices
    )

    # Get dimensions from first training graph
    node_dim = train_graphs[0]['node_features'].shape[1]
    edge_dim = train_graphs[0]['edge_features'].shape[1]

    print(f"\n  Node feature dim: {node_dim}")
    print(f"  Edge feature dim: {edge_dim}")

    # Initialize model
    print("\n[2/4] Initializing model...")
    model = InductiveGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params}")

    # Train
    print("\n[3/4] Training...")
    trainer = CrossNetworkTrainer(model, config)
    train_results = trainer.train(train_graphs)

    # Test on held-out networks
    print("\n[4/4] Evaluating on test networks...")
    test_metrics = trainer.evaluate(test_graphs)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTraining (on {len(train_graphs)} networks):")
    print(f"  Best validation AUC: {train_results['best_val_auc']:.4f}")
    print(f"  Best epoch: {train_results['best_epoch']}")

    print(f"\nTest (on {len(test_graphs)} UNSEEN networks):")
    print(f"  AUC:                    {test_metrics['auc']:.4f}")
    print(f"  Average Precision:      {test_metrics['ap']:.4f}")
    print(f"  Precision @ 50% Recall: {test_metrics['precision_at_50_recall']:.4f}")
    print(f"  Total samples:          {test_metrics['n_samples']}")
    print(f"  Positive samples:       {test_metrics['n_positive']}")

    return {
        'model': model,
        'train_results': train_results,
        'test_metrics': test_metrics,
        'config': config,
        'dataset': dataset  # Include dataset for visualization (has fitted scaler)
    }


class SlidingWindowPredictor:
    """
    Two-Stage Sliding Window Prediction
    
    Stage 1: Compute node embeddings using FULL graph (consistent embeddings)
    Stage 2: Predict edges in sliding windows using pre-computed embeddings
    
    Benefits:
    - Memory efficient: processes edges in batches
    - Consistent embeddings: each neuron has ONE embedding
    - Scalable: works for very large networks
    - No boundary artifacts
    """
    
    def __init__(self, model: InductiveGNN, config: CrossNetworkConfig,
                 device = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.feature_extractor = GeneralizableFeatureExtractor(config)
    
    def compute_windows(self, neuron_positions: np.ndarray) -> List[Dict]:
        """
        Compute sliding windows across the spatial extent of the network.
        
        Returns list of windows, each containing:
        - bounds: (x_min, x_max, y_min, y_max)
        - center: (x_center, y_center)
        """
        # Get spatial extent
        x_min, y_min = neuron_positions.min(axis=0)
        x_max, y_max = neuron_positions.max(axis=0)
        
        window_size = self.config.window_size
        step = window_size * (1 - self.config.window_overlap)
        
        windows = []
        x = x_min
        while x < x_max:
            y = y_min
            while y < y_max:
                windows.append({
                    'bounds': (x, x + window_size, y, y + window_size),
                    'center': (x + window_size/2, y + window_size/2)
                })
                y += step
            x += step
        
        return windows
    
    def get_edges_for_window(self, window: Dict, neuron_positions: np.ndarray,
                             pos_set: set) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get candidate edges for a window.
        
        Strategy: An edge (i,j) belongs to the window containing its MIDPOINT.
        This ensures each edge is predicted exactly once.
        
        Returns:
            edges: [2, n_edges] array of edge indices
            labels: [n_edges] array of ground truth (1=connected, 0=not)
        """
        x_min, x_max, y_min, y_max = window['bounds']
        n_neurons = len(neuron_positions)
        
        edges = []
        labels = []
        
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i == j:
                    continue
                
                # Compute midpoint of edge
                mid_x = (neuron_positions[i, 0] + neuron_positions[j, 0]) / 2
                mid_y = (neuron_positions[i, 1] + neuron_positions[j, 1]) / 2
                
                # Check if midpoint is in this window
                if x_min <= mid_x < x_max and y_min <= mid_y < y_max:
                    # Optionally filter by distance
                    if self.config.filter_by_radius:
                        dist = np.sqrt(np.sum((neuron_positions[i] - neuron_positions[j])**2))
                        if dist > self.config.connection_radius:
                            continue
                    
                    edges.append([i, j])
                    labels.append(1 if (i, j) in pos_set else 0)
        
        if len(edges) == 0:
            return np.zeros((2, 0), dtype=int), np.zeros(0)
        
        return np.array(edges).T, np.array(labels)
    
    def compute_edge_features_batch(self, edges: np.ndarray, 
                                    spike_times: List, duration_ms: float,
                                    neuron_positions: np.ndarray,
                                    binned_spikes: np.ndarray,
                                    firing_rates: np.ndarray) -> np.ndarray:
        """
        Compute edge features for a batch of edges.
        Uses pre-computed binned spikes and firing rates for efficiency.
        Includes transfer entropy for causal connectivity inference.
        """
        n_edges = edges.shape[1]
        conn_radius = self.config.connection_radius
        
        features = []
        
        for k in range(n_edges):
            i, j = edges[0, k], edges[1, k]
            
            # Distance features
            dist = np.sqrt(np.sum((neuron_positions[i] - neuron_positions[j])**2))
            normalized_dist = dist / conn_radius
            within_radius = float(dist <= conn_radius)
            
            # Correlations at multiple lags
            correlations = []
            x, y = binned_spikes[i], binned_spikes[j]
            correlation_lags = self.config.correlation_lags if self.config.correlation_lags is not None else [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            for lag in correlation_lags:
                if lag == 0:
                    if np.std(x) > 0 and np.std(y) > 0:
                        corr = np.corrcoef(x, y)[0, 1]
                    else:
                        corr = 0
                elif lag > 0 and len(x) > lag:
                    if np.std(x[:-lag]) > 0 and np.std(y[lag:]) > 0:
                        corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                    else:
                        corr = 0
                elif lag < 0 and len(x) > -lag:
                    if np.std(x[-lag:]) > 0 and np.std(y[:lag]) > 0:
                        corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
                    else:
                        corr = 0
                else:
                    corr = 0
                correlations.append(corr if not np.isnan(corr) else 0)
            
            # Transfer entropy features (directional causality)
            # TE from i->j (does i's past predict j's future?)
            te_i_to_j = compute_transfer_entropy(x, y, lag=1)
            # TE from j->i (does j's past predict i's future?)
            te_j_to_i = compute_transfer_entropy(y, x, lag=1)
            # Asymmetry: positive means i->j is stronger
            te_asymmetry = te_i_to_j - te_j_to_i
            # Net transfer entropy (for undirected prediction)
            te_max = max(te_i_to_j, te_j_to_i)
            
            # Firing rate features
            fr_diff = abs(firing_rates[i] - firing_rates[j])
            fr_product = firing_rates[i] * firing_rates[j]
            
            features.append([dist, normalized_dist, within_radius] + correlations + 
                          [te_i_to_j, te_j_to_i, te_asymmetry, te_max, fr_diff, fr_product])
        
        features = np.array(features, dtype=np.float32)
        
        # Validate feature dimension matches training
        expected_dim = getattr(self.feature_extractor, '_edge_feature_dim', None)
        if expected_dim is not None and features.shape[1] != expected_dim:
            raise ValueError(
                f"Edge feature dimension mismatch: got {features.shape[1]}, expected {expected_dim}. "
                f"This may indicate inconsistent configuration between training and prediction."
            )
        
        return features
    
    @torch.no_grad()
    def predict(self, network_data: Dict, dataset: CrossNetworkDataset) -> Dict:
        """
        Two-stage prediction with sliding windows.
        
        Stage 1: Compute node embeddings on full graph
        Stage 2: Predict edges in batches using sliding windows
        """
        self.model.eval()
        
        neuron_positions = network_data['neuron_positions']
        n_neurons = network_data['n_neurons']
        connections = network_data['connections']
        spike_times = network_data['spike_times']
        duration_ms = network_data['duration_ms']
        
        # Ground truth edge set
        pos_set = set((int(c[0]), int(c[1])) for c in connections)
        
        print(f"  Stage 1: Computing node embeddings...")
        
        # Create graph for embedding computation
        embed_graph = dataset.create_embedding_graph(network_data)
        
        # Compute node embeddings (ONCE for entire graph)
        x = embed_graph['node_features'].to(self.device)
        edge_index = embed_graph['edge_index'].to(self.device)
        node_embeddings = self.model.encode(x, edge_index)  # [N, hidden_dim]
        
        print(f"    Node embeddings shape: {node_embeddings.shape}")
        
        # Pre-compute binned spikes and firing rates
        duration_s = duration_ms / 1000.0
        binned_spikes = self.feature_extractor.compute_binned_spikes(spike_times, duration_ms)
        firing_rates = np.array([len(list(s)) / duration_s for s in spike_times])
        
        print(f"  Stage 2: Predicting edges with sliding windows...")
        
        # Compute windows
        windows = self.compute_windows(neuron_positions)
        print(f"    Number of windows: {len(windows)}")
        
        # Collect all predictions
        all_edges = []
        all_probs = []
        all_labels = []
        
        for w_idx, window in enumerate(windows):
            # Get edges for this window
            edges, labels = self.get_edges_for_window(window, neuron_positions, pos_set)
            
            if edges.shape[1] == 0:
                continue
            
            # Process in batches if too many edges
            batch_size = self.config.edges_per_batch
            n_batches = (edges.shape[1] + batch_size - 1) // batch_size
            
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, edges.shape[1])
                
                batch_edges = edges[:, start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                
                # Compute edge features for this batch
                edge_features = self.compute_edge_features_batch(
                    batch_edges, spike_times, duration_ms,
                    neuron_positions, binned_spikes, firing_rates
                )
                
                # Normalize edge features (use fitted scaler from feature extractor)
                if self.feature_extractor.fitted and len(edge_features) > 0:
                    cols_to_normalize = [0, 1] + list(range(3, edge_features.shape[1]))
                    edge_features[:, cols_to_normalize] = \
                        self.feature_extractor.edge_scaler.transform(edge_features[:, cols_to_normalize])
                
                # Get node embeddings for source and target
                edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32).to(self.device)
                z_i = node_embeddings[batch_edges[0]]
                z_j = node_embeddings[batch_edges[1]]
                
                # Predict
                logits = self.model.edge_predictor(z_i, z_j, edge_features_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                all_edges.append(batch_edges)
                all_probs.append(probs)
                all_labels.append(batch_labels)
        
        # Concatenate all predictions
        if len(all_edges) == 0:
            raise ValueError("No edges to predict. Check window size and connection radius.")
        
        all_edges = np.concatenate(all_edges, axis=1)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        print(f"    Total edges predicted: {len(all_probs)}")
        print(f"    Positive edges: {np.sum(all_labels)}")
        
        return {
            'pred_edges': all_edges,
            'pred_probs': all_probs,
            'true_labels': all_labels
        }


def predict_new_network(model: InductiveGNN, metadata_file: str,
                       config: Optional[CrossNetworkConfig] = None,
                       dataset: Optional[CrossNetworkDataset] = None) -> Dict:
    """
    Predict connectivity for a completely new network.
    
    Uses Two-Stage Sliding Window approach if config.use_sliding_window=True:
    - Stage 1: Compute node embeddings on full graph (consistent embeddings)
    - Stage 2: Predict edges in sliding windows (memory efficient)

    Args:
        model: Trained GNN model
        metadata_file: Path to new network's metadata
        config: Configuration (use same as training)
        dataset: Optional pre-initialized dataset (for reusing fitted scalers)

    Returns:
        Predictions and ground truth
    """
    if config is None:
        config = CrossNetworkConfig()

    print(f"\nPredicting connectivity for: {metadata_file}")
    
    if dataset is None:
        dataset = CrossNetworkDataset(config)
    
    network = dataset.load_network(metadata_file)
    
    device = next(model.parameters()).device
    
    if config.use_sliding_window:
        # Two-Stage Sliding Window Prediction
        print("  Using Two-Stage Sliding Window Prediction")
        predictor = SlidingWindowPredictor(model, config, device)
        predictor.feature_extractor = dataset.feature_extractor  # Share fitted scaler
        
        results = predictor.predict(network, dataset)
        
        pred_edges = results['pred_edges']
        probs = results['pred_probs']
        labels = results['true_labels']
    else:
        # Original method: predict all edges at once
        print("  Using full graph prediction")
        graph = dataset.create_graph_data(network, fit_scaler=False, include_all_edges=True)
        
        model.eval()
        with torch.no_grad():
            x = graph['node_features'].to(device)
            edge_index = graph['edge_index'].to(device)
            pred_edges_tensor = graph['pred_edges'].to(device)
            edge_features = graph['edge_features'].to(device)

            logits = model(x, edge_index, pred_edges_tensor, edge_features)
            probs = torch.sigmoid(logits).cpu().numpy()

        pred_edges = graph['pred_edges'].numpy()
        labels = graph['pred_labels'].numpy()

    # Compute metrics
    if len(np.unique(labels)) > 1:  # Need both classes for AUC
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
    else:
        auc = 0.0
        ap = 0.0
        print("  Warning: Only one class present, cannot compute AUC/AP")

    n_connections = len(network.get('connections', [])) if 'connections' in network else int(np.sum(labels))
    
    print(f"  Neurons: {network['n_neurons']}")
    print(f"  True connections: {n_connections}")
    print(f"  AUC: {auc:.4f}")
    print(f"  AP: {ap:.4f}")

    return {
        'pred_edges': pred_edges,
        'pred_probs': probs,
        'true_labels': labels,
        'auc': auc,
        'ap': ap,
        'n_neurons': network['n_neurons'],
        'n_connections': n_connections
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Create output directories
    output_dirs = ensure_output_dirs()
    print(f"Output directories created: {output_dirs['base']}/")

    # Find all simulations
    metadata_files = sorted(glob.glob('LIF data/*/session_gnn_metadata.json'))

    if len(metadata_files) < 2:
        print("Need at least 2 simulation sessions for cross-network training!")
        print(f"Found: {len(metadata_files)}")
        print("\nRun the LIF simulation notebook multiple times with different")
        print("random seeds or parameters to generate multiple networks.")
    else:
        print(f"Found {len(metadata_files)} simulation sessions")

        # Use random selection for test networks (better diversity)
        # This ensures test networks are distributed across all available data
        # np.random.seed(42)  # For reproducibility
        n_test = 1  # Train on 4, test on 1
        all_indices = list(range(len(metadata_files)))
        test_indices = sorted(np.random.choice(all_indices, n_test, replace=False).tolist())
        
        print(f"Training networks: {len(metadata_files) - n_test}")
        print(f"Test networks: {n_test}")
        print(f"Test indices: {test_indices}")

        config = CrossNetworkConfig(
            hidden_dim=64,
            num_layers=3,
            dropout=0.3,
            num_epochs=300,
            patience=30,
            neg_ratio=3.0,
            pos_weight=5.0,
            connection_radius=8.0,
            neighbor_radius=4.0,
            filter_by_radius=True,
            # Sliding window parameters
            use_sliding_window=True,
            window_size=7.0,
            window_overlap=0.5,
            edges_per_batch=2000
        )

        results = run_cross_network_pipeline(
            metadata_files,
            test_indices,
            config
        )

        # Compute optimal threshold from training evaluation BEFORE saving
        # This threshold will be saved and reused for predictions on new networks
        _pred_results_for_threshold = predict_new_network(
            results['model'],
            metadata_files[test_indices[0]],
            config,
            results['dataset']
        )
        _tr_probs = _pred_results_for_threshold['pred_probs']
        _tr_labels = _pred_results_for_threshold['true_labels']
        _tr_precision, _tr_recall, _tr_thresholds = precision_recall_curve(_tr_labels, _tr_probs)
        _tr_f1 = 2 * (_tr_precision[:-1] * _tr_recall[:-1]) / (_tr_precision[:-1] + _tr_recall[:-1] + 1e-10)
        _tr_opt_idx = np.argmax(_tr_f1)
        training_optimal_threshold = float(_tr_thresholds[_tr_opt_idx]) if len(_tr_thresholds) > 0 else 0.5
        print(f"\nTraining optimal threshold (max F1): {training_optimal_threshold:.4f}")

        # Save model and scalers for later prediction on new data
        print("\n[Saving model and scalers...]")
        model_path = os.path.join(output_dirs['models'], 'trained_gnn_model.pt')
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'node_scaler_mean': results['dataset'].feature_extractor.node_scaler.mean_,
            'node_scaler_scale': results['dataset'].feature_extractor.node_scaler.scale_,
            'edge_scaler_mean': results['dataset'].feature_extractor.edge_scaler.mean_,
            'edge_scaler_scale': results['dataset'].feature_extractor.edge_scaler.scale_,
            'config': config,
            'node_dim': results['model'].node_encoder.in_features,
            'edge_dim': results['model'].edge_predictor.edge_encoder[0].in_features,
            'optimal_threshold': training_optimal_threshold,
        }, model_path)
        print(f"Model saved to: {model_path}")

        # ================================================================
        # VISUALIZATION: Predicted vs Actual Connections
        # ================================================================
        import matplotlib.pyplot as plt
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION...")
        print("="*70)
        
        # Get predictions for the test network
        test_file = metadata_files[test_indices[0]]
        
        # Use the dataset from training (already has fitted scaler)
        dataset = results['dataset']
        
        # Now predict on test network using the properly fitted scaler
        prediction_results = predict_new_network(
            results['model'], 
            test_file, 
            config,
            dataset
        )
        
        # Load network data for positions
        test_network = dataset.load_network(test_file)
        positions = test_network['neuron_positions']
        n_neurons = test_network['n_neurons']
        
        pred_edges = prediction_results['pred_edges']
        pred_probs = prediction_results['pred_probs']
        true_labels = prediction_results['true_labels']
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # ---- Plot 1: Actual Connections ----
        ax1 = axes[0, 0]
        ax1.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=50, edgecolors='navy', zorder=3)
        
        # Draw actual connections
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
        
        # Color by correctness
        for k in range(pred_positive_edges.shape[1]):
            i, j = pred_positive_edges[0, k], pred_positive_edges[1, k]
            # Check if this is a true positive
            edge_idx = np.where(pred_mask)[0][k]
            is_correct = true_labels[edge_idx] == 1
            color = 'green' if is_correct else 'red'
            alpha = 0.5 if is_correct else 0.3
            ax2.plot([positions[i, 0], positions[j, 0]], 
                    [positions[i, 1], positions[j, 1]], 
                    color=color, alpha=alpha, linewidth=0.5)
        
        tp = np.sum((pred_probs > threshold) & (true_labels == 1))
        fp = np.sum((pred_probs > threshold) & (true_labels == 0))
        ax2.set_title(f'Predicted Connections (thresh=0.5)\nTP={tp} (green), FP={fp} (red)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_aspect('equal')
        
        # ---- Plot 3: Prediction Probability Distribution ----
        ax3 = axes[1, 0]
        
        # Separate by class
        pos_probs = pred_probs[true_labels == 1]
        neg_probs = pred_probs[true_labels == 0]
        
        ax3.hist(neg_probs, bins=50, alpha=0.6, label=f'No Connection (n={len(neg_probs)})', color='red')
        ax3.hist(pos_probs, bins=50, alpha=0.6, label=f'True Connection (n={len(pos_probs)})', color='green')
        ax3.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
        ax3.set_xlabel('Predicted Probability')
        ax3.set_ylabel('Count')
        ax3.set_title('Prediction Score Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        
        # ---- Plot 4: Precision-Recall Curve ----
        ax4 = axes[1, 1]
        
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
        ax4.plot(recall, precision, 'b-', linewidth=2)
        ax4.fill_between(recall, precision, alpha=0.2)
        
        # Mark threshold=0.5
        idx_05 = np.argmin(np.abs(thresholds - 0.5)) if len(thresholds) > 0 else 0
        ax4.scatter([recall[idx_05]], [precision[idx_05]], color='red', s=100, zorder=5, 
                   label=f'Threshold=0.5\nP={precision[idx_05]:.2f}, R={recall[idx_05]:.2f}')
        
        # Find optimal threshold (maximize F1)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        ax4.scatter([recall[optimal_idx]], [precision[optimal_idx]], color='green', s=100, zorder=5, marker='*',
                   label=f'Optimal (t={optimal_threshold:.2f})\nP={precision[optimal_idx]:.2f}, R={recall[optimal_idx]:.2f}')
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title(f'Precision-Recall Curve (AP={prediction_results["ap"]:.3f})', 
                     fontsize=12, fontweight='bold')
        ax4.legend(loc='lower left')
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figure_path = os.path.join(output_dirs['figures'], 'prediction_results.png')
        plt.savefig(figure_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {figure_path}")
        plt.close()  # Close instead of show to avoid blocking
        
        # ================================================================
        # THRESHOLD ANALYSIS
        # ================================================================
        print("\n" + "="*70)
        print("THRESHOLD ANALYSIS")
        print("="*70)
        
        print(f"\nOptimal threshold (max F1): {optimal_threshold:.3f}")
        print(f"  F1 Score: {f1_scores[optimal_idx]:.4f}")
        print(f"  Precision: {precision[optimal_idx]:.4f}")
        print(f"  Recall: {recall[optimal_idx]:.4f}")
        
        print("\nMetrics at different thresholds:")
        print("-" * 60)
        print(f"{'Threshold':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<10}")
        print("-" * 60)
        
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, optimal_threshold]:
            tp_t = np.sum((pred_probs >= thresh) & (true_labels == 1))
            fp_t = np.sum((pred_probs >= thresh) & (true_labels == 0))
            fn_t = np.sum((pred_probs < thresh) & (true_labels == 1))
            prec_t = tp_t / (tp_t + fp_t + 1e-10)
            rec_t = tp_t / (tp_t + fn_t + 1e-10)
            f1_t = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-10)
            marker = " <-- optimal" if abs(thresh - optimal_threshold) < 0.01 else ""
            print(f"{thresh:<12.2f} {tp_t:<8} {fp_t:<8} {fn_t:<8} {prec_t:<12.4f} {rec_t:<12.4f} {f1_t:<10.4f}{marker}")
        
        print("-" * 60)
        print(f"\nVisualization saved to: {figure_path}")