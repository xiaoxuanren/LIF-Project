"""
Analyze predicted connection distributions from learned-LIF outputs.

Generates summary statistics and plots for:
  - thresholded predicted connections
  - true positives / false positives
  - false negative true-connection distribution

Usage:
    python analyze_predicted_connectivity_distribution.py --session "LIF data/20260425_110211"
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_network_data(session_dir):
    net_files = glob.glob(os.path.join(session_dir, 'network_*.npz'))
    if not net_files:
        raise FileNotFoundError(f'No network file found in {session_dir}')
    return np.load(net_files[0], allow_pickle=True)


def resolve_connectivity_path(session_name, explicit_path=None):
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f'Connectivity file not found: {explicit_path}')

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'learned_lif_outputs')
    exact_path = os.path.join(output_dir, f'connectivity_{session_name}.npz')
    if os.path.exists(exact_path):
        return exact_path

    candidates = glob.glob(os.path.join(output_dir, f'connectivity_{session_name}*.npz'))
    if not candidates:
        raise FileNotFoundError(f'No learned-LIF connectivity output found for session {session_name}')
    return max(candidates, key=os.path.getmtime)


def build_ground_truth(connections, n_neurons):
    true_weights = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    true_binary = np.zeros((n_neurons, n_neurons), dtype=bool)
    for conn in connections:
        pre_id = int(conn[0])
        post_id = int(conn[1])
        weight = float(conn[2])
        true_weights[post_id, pre_id] = weight
        true_binary[post_id, pre_id] = True
    return true_weights, true_binary


def describe_values(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            'count': 0,
            'mean_abs': 0.0,
            'median_abs': 0.0,
            'p90_abs': 0.0,
            'frac_abs_ge_0_3': 0.0,
            'frac_abs_ge_0_5': 0.0,
        }

    abs_values = np.abs(values)
    return {
        'count': int(values.size),
        'mean_abs': float(abs_values.mean()),
        'median_abs': float(np.median(abs_values)),
        'p90_abs': float(np.percentile(abs_values, 90)),
        'frac_abs_ge_0_3': float(np.mean(abs_values >= 0.3)),
        'frac_abs_ge_0_5': float(np.mean(abs_values >= 0.5)),
    }


def sign_breakdown(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            'positive_count': 0,
            'negative_count': 0,
            'positive_fraction': 0.0,
            'negative_fraction': 0.0,
        }

    positive = values > 0
    negative = values < 0
    return {
        'positive_count': int(positive.sum()),
        'negative_count': int(negative.sum()),
        'positive_fraction': float(positive.mean()),
        'negative_fraction': float(negative.mean()),
    }


def build_analysis(session_dir, connectivity_path=None, threshold_override=None):
    session_name = os.path.basename(session_dir)
    net_data = load_network_data(session_dir)
    resolved_path = resolve_connectivity_path(session_name, explicit_path=connectivity_path)
    pred_data = np.load(resolved_path, allow_pickle=True)

    connectivity_matrix = pred_data['connectivity_matrix'].astype(np.float32)  # [post, pre]
    n_neurons = connectivity_matrix.shape[0]
    threshold = float(threshold_override) if threshold_override is not None else float(pred_data['threshold'])

    true_weights, true_binary = build_ground_truth(net_data['connections'], n_neurons)
    mask = ~np.eye(n_neurons, dtype=bool)
    predicted_binary = (np.abs(connectivity_matrix) >= threshold) & mask

    true_positive = predicted_binary & true_binary
    false_positive = predicted_binary & (~true_binary) & mask
    false_negative = true_binary & (~predicted_binary) & mask

    predicted_weights = connectivity_matrix[predicted_binary]
    tp_predicted_weights = connectivity_matrix[true_positive]
    fp_predicted_weights = connectivity_matrix[false_positive]
    fn_true_weights = true_weights[false_negative]
    tp_true_weights = true_weights[true_positive]

    summary = {
        'session_name': session_name,
        'connectivity_path': resolved_path,
        'threshold': threshold,
        'n_neurons': int(n_neurons),
        'predicted_edges': int(predicted_binary.sum()),
        'true_positive_edges': int(true_positive.sum()),
        'false_positive_edges': int(false_positive.sum()),
        'false_negative_edges': int(false_negative.sum()),
        'predicted_sign': sign_breakdown(predicted_weights),
        'predicted_strength': describe_values(predicted_weights),
        'true_positive_predicted_strength': describe_values(tp_predicted_weights),
        'false_positive_predicted_strength': describe_values(fp_predicted_weights),
        'false_negative_true_sign': sign_breakdown(fn_true_weights),
        'false_negative_true_strength': describe_values(fn_true_weights),
        'true_positive_true_strength': describe_values(tp_true_weights),
    }

    return {
        'session_name': session_name,
        'connectivity_path': resolved_path,
        'threshold': threshold,
        'predicted_weights': predicted_weights,
        'tp_predicted_weights': tp_predicted_weights,
        'fp_predicted_weights': fp_predicted_weights,
        'fn_true_weights': fn_true_weights,
        'tp_true_weights': tp_true_weights,
        'summary': summary,
    }


def plot_analysis(analysis, output_path):
    predicted_weights = analysis['predicted_weights']
    tp_predicted_weights = analysis['tp_predicted_weights']
    fp_predicted_weights = analysis['fp_predicted_weights']
    fn_true_weights = analysis['fn_true_weights']
    tp_true_weights = analysis['tp_true_weights']
    summary = analysis['summary']

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(f"Predicted Connection Distribution — {analysis['session_name']}", fontsize=14, fontweight='bold')

    # 1. Predicted sign counts
    ax = axes[0, 0]
    positive_count = int(np.sum(predicted_weights > 0))
    negative_count = int(np.sum(predicted_weights < 0))
    ax.bar(['Excitatory (+)', 'Inhibitory (-)'], [positive_count, negative_count], color=['crimson', 'royalblue'])
    ax.set_title('Predicted Edge Sign Count')
    ax.set_ylabel('Edge count')

    # 2. Predicted signed weights
    ax = axes[0, 1]
    ax.hist(predicted_weights[predicted_weights > 0], bins=50, alpha=0.7, color='crimson', label='Predicted +')
    ax.hist(predicted_weights[predicted_weights < 0], bins=50, alpha=0.7, color='royalblue', label='Predicted -')
    ax.axvline(0.0, color='black', linewidth=1)
    ax.set_title('Predicted Signed Weight Distribution')
    ax.set_xlabel('Predicted weight')
    ax.set_ylabel('Count')
    ax.legend()

    # 3. Predicted absolute strengths for TP vs FP
    ax = axes[0, 2]
    ax.hist(np.abs(tp_predicted_weights), bins=50, alpha=0.65, color='seagreen', label='True positives')
    ax.hist(np.abs(fp_predicted_weights), bins=50, alpha=0.65, color='darkorange', label='False positives')
    ax.axvline(summary['threshold'], color='black', linestyle='--', linewidth=1.5, label=f"Threshold={summary['threshold']:.3f}")
    ax.set_title('Predicted |Weight| for TP vs FP')
    ax.set_xlabel('|Predicted weight|')
    ax.set_ylabel('Count')
    ax.legend()

    # 4. False negative sign counts
    ax = axes[1, 0]
    fn_exc_count = int(np.sum(fn_true_weights > 0))
    fn_inh_count = int(np.sum(fn_true_weights < 0))
    ax.bar(['FN excitatory', 'FN inhibitory'], [fn_exc_count, fn_inh_count], color=['crimson', 'royalblue'])
    ax.set_title('False Negative True Edge Sign Count')
    ax.set_ylabel('Edge count')

    # 5. True |weight| for TP vs FN
    ax = axes[1, 1]
    ax.hist(np.abs(tp_true_weights), bins=50, alpha=0.65, color='seagreen', label='Recovered true edges (TP)')
    ax.hist(np.abs(fn_true_weights), bins=50, alpha=0.65, color='purple', label='Missed true edges (FN)')
    ax.set_title('True |Weight| for TP vs FN')
    ax.set_xlabel('|True weight|')
    ax.set_ylabel('Count')
    ax.legend()

    # 6. False negative signed true weights
    ax = axes[1, 2]
    ax.hist(fn_true_weights[fn_true_weights > 0], bins=50, alpha=0.7, color='crimson', label='FN excitatory')
    ax.hist(fn_true_weights[fn_true_weights < 0], bins=50, alpha=0.7, color='royalblue', label='FN inhibitory')
    ax.axvline(0.0, color='black', linewidth=1)
    ax.set_title('False Negative Signed True Weight Distribution')
    ax.set_xlabel('True weight')
    ax.set_ylabel('Count')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


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


def main():
    parser = argparse.ArgumentParser(description='Analyze predicted connectivity distribution')
    parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--connectivity-path', type=str, default=None,
                        help='Optional explicit learned-LIF connectivity NPZ path')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Optional threshold override for predicted edges')
    parser.add_argument('--output-tag', type=str, default=None,
                        help='Optional suffix for saved artifact names')
    args = parser.parse_args()

    session_dir = args.session if args.session else select_session()
    analysis = build_analysis(
        session_dir,
        connectivity_path=args.connectivity_path,
        threshold_override=args.threshold,
    )

    session_name = analysis['session_name']
    output_tag = args.output_tag.strip().replace(' ', '_') if args.output_tag else None
    output_name = session_name if not output_tag else f'{session_name}_{output_tag}'

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'learned_lif_outputs')
    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, f'predicted_connection_distribution_{output_name}.png')
    json_path = os.path.join(output_dir, f'predicted_connection_distribution_{output_name}.json')

    plot_analysis(analysis, figure_path)
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(analysis['summary'], handle, indent=2)

    print(f"Session: {session_name}")
    print(f"Connectivity file: {analysis['connectivity_path']}")
    print(f"Predicted edges: {analysis['summary']['predicted_edges']}")
    print(f"TP / FP / FN: {analysis['summary']['true_positive_edges']} / {analysis['summary']['false_positive_edges']} / {analysis['summary']['false_negative_edges']}")
    print(
        'Predicted sign fractions: '
        f"+ {analysis['summary']['predicted_sign']['positive_fraction']:.3f}, "
        f"- {analysis['summary']['predicted_sign']['negative_fraction']:.3f}"
    )
    print(
        'Predicted |weight|: '
        f"mean={analysis['summary']['predicted_strength']['mean_abs']:.3f}, "
        f"median={analysis['summary']['predicted_strength']['median_abs']:.3f}, "
        f"p90={analysis['summary']['predicted_strength']['p90_abs']:.3f}"
    )
    print(
        'False negative true-edge sign fractions: '
        f"+ {analysis['summary']['false_negative_true_sign']['positive_fraction']:.3f}, "
        f"- {analysis['summary']['false_negative_true_sign']['negative_fraction']:.3f}"
    )
    print(
        'False negative |true weight|: '
        f"mean={analysis['summary']['false_negative_true_strength']['mean_abs']:.3f}, "
        f"median={analysis['summary']['false_negative_true_strength']['median_abs']:.3f}, "
        f"p90={analysis['summary']['false_negative_true_strength']['p90_abs']:.3f}"
    )
    print(f'Figure saved: {figure_path}')
    print(f'Stats saved: {json_path}')


if __name__ == '__main__':
    main()