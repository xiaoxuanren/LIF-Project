"""Connectivity scoring and thresholding helpers for learned-LIF inference."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def flatten_candidate_scores(conn_matrix, neighbor_indices, neuron_ids=None,
                             absolute=True):
    """Flatten candidate-edge scores from a learned connectivity matrix."""
    if neuron_ids is None:
        neuron_ids = np.arange(conn_matrix.shape[0])

    all_scores = []
    for neuron_id in np.asarray(neuron_ids, dtype=np.int32):
        row_scores = np.asarray(conn_matrix[neuron_id, neighbor_indices[neuron_id]])
        all_scores.append(np.abs(row_scores) if absolute else row_scores)

    if not all_scores:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(all_scores).astype(np.float32, copy=False)


def compute_binary_classification_metrics(labels, predicted):
    """Compute confusion counts plus precision, recall, and F1."""
    labels = np.asarray(labels).astype(np.int32, copy=False)
    predicted = np.asarray(predicted).astype(np.int32, copy=False)

    tp = int(np.sum((predicted == 1) & (labels == 1)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))
    tn = int(np.sum((predicted == 0) & (labels == 0)))
    precision = float(tp / (tp + fp + 1e-10))
    recall = float(tp / (tp + fn + 1e-10))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-10))

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def select_connectivity_threshold(labels, scores, mode='oracle_f1',
                                  surrogate_score_sets=None,
                                  surrogate_fdr=0.005,
                                  default_threshold=0.5):
    """Choose a connectivity cutoff from labels or surrogate null scores."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    mode = str(mode).strip().lower()

    if scores.size == 0:
        return {
            'threshold': float(default_threshold),
            'connectivity_threshold_mode': mode,
            'estimated_fdr': 0.0,
            'expected_null_selected': 0.0,
            'selected_edges': 0,
        }

    if mode == 'oracle_f1':
        prec, rec, thresholds = precision_recall_curve(labels, scores)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-10)
        best_idx = int(np.argmax(f1))
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(default_threshold)
        return {
            'threshold': best_thresh,
            'connectivity_threshold_mode': 'oracle_f1',
            'estimated_fdr': None,
            'expected_null_selected': None,
            'selected_edges': int(np.sum(scores >= best_thresh)),
        }

    if mode != 'surrogate_fdr':
        raise ValueError(
            f'Unsupported connectivity_threshold_mode={mode!r}; '
            "use 'oracle_f1' or 'surrogate_fdr'"
        )

    if surrogate_score_sets is None:
        raise ValueError('surrogate_score_sets are required when using surrogate_fdr thresholding')

    surrogate_score_sets = np.asarray(surrogate_score_sets, dtype=np.float64)
    if surrogate_score_sets.ndim == 1:
        surrogate_score_sets = surrogate_score_sets[None, :]
    if surrogate_score_sets.ndim != 2 or surrogate_score_sets.size == 0:
        raise ValueError('surrogate_score_sets must be a non-empty 1D or 2D array')

    chosen_threshold = float('inf')
    chosen_selected = 0
    chosen_expected_null = 0.0
    chosen_estimated_fdr = 0.0

    for threshold in np.unique(scores)[::-1]:
        observed_selected = int(np.sum(scores >= threshold))
        if observed_selected <= 0:
            continue
        expected_null_selected = float(np.mean(np.sum(surrogate_score_sets >= threshold, axis=1)))
        estimated_fdr = float(expected_null_selected / observed_selected)
        if estimated_fdr <= float(surrogate_fdr):
            chosen_threshold = float(threshold)
            chosen_selected = observed_selected
            chosen_expected_null = expected_null_selected
            chosen_estimated_fdr = estimated_fdr

    return {
        'threshold': chosen_threshold,
        'connectivity_threshold_mode': 'surrogate_fdr',
        'estimated_fdr': chosen_estimated_fdr,
        'expected_null_selected': chosen_expected_null,
        'selected_edges': chosen_selected,
        'surrogate_fdr_target': float(surrogate_fdr),
        'surrogate_n_models': int(surrogate_score_sets.shape[0]),
        'surrogate_edges_per_model': int(surrogate_score_sets.shape[1]),
    }


@torch.no_grad()
def evaluate_connectivity(model, neighbor_indices, true_binary, neuron_ids=None,
                          connectivity_threshold_mode='oracle_f1',
                          surrogate_score_sets=None,
                          surrogate_fdr=0.005):
    """Evaluate connectivity prediction from the learned weight matrix."""
    model.eval()
    conn_matrix = model.get_connectivity_matrix(neighbor_indices)
    n_neurons = model.n_neurons

    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)

    all_scores = []
    all_labels = []

    for neuron_id in neuron_ids:
        pre_ids = neighbor_indices[neuron_id]
        scores = np.abs(conn_matrix[neuron_id, pre_ids])
        labels = true_binary[neuron_id, pre_ids].astype(np.float32)
        all_scores.append(scores)
        all_labels.append(labels)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    results = {}
    if len(np.unique(labels)) > 1:
        results['auc'] = roc_auc_score(labels, scores)
        results['ap'] = average_precision_score(labels, scores)

        threshold_info = select_connectivity_threshold(
            labels,
            scores,
            mode=connectivity_threshold_mode,
            surrogate_score_sets=surrogate_score_sets,
            surrogate_fdr=surrogate_fdr,
            default_threshold=0.5,
        )
        predicted = np.zeros_like(labels, dtype=np.int32)
        if np.isfinite(threshold_info['threshold']):
            predicted = (scores >= threshold_info['threshold']).astype(np.int32)

        results.update(threshold_info)
        results.update(compute_binary_classification_metrics(labels, predicted))
    else:
        results.update({
            'auc': 0,
            'ap': 0,
            'f1': 0,
            'threshold': 0.5,
            'precision': 0,
            'recall': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': int(np.sum(labels == 0)),
            'connectivity_threshold_mode': str(connectivity_threshold_mode).strip().lower(),
            'estimated_fdr': None,
            'expected_null_selected': None,
            'selected_edges': 0,
        })

    all_w_learned = []
    for neuron_id in neuron_ids:
        pre_ids = neighbor_indices[neuron_id]
        mask = true_binary[neuron_id, pre_ids] == 1
        if mask.any():
            all_w_learned.append(conn_matrix[neuron_id, pre_ids[mask]])
    all_w_learned_full = np.concatenate(all_w_learned) if all_w_learned else np.array([])
    if len(all_w_learned_full) > 0:
        results['mean_connected_weight'] = float(np.mean(np.abs(all_w_learned_full)))
    else:
        results['mean_connected_weight'] = 0.0

    results['n_positive'] = int(labels.sum())
    results['n_total'] = len(labels)

    return results, scores, labels, conn_matrix