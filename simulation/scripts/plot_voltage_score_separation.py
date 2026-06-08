import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lif_inference.learned_lif_connectivity import (
    PerNeuronLIF,
    build_ground_truth,
    estimate_surrogate_connectivity_score_sets as estimate_spike_surrogate_connectivity_score_sets,
    evaluate_connectivity as evaluate_spike_connectivity,
    load_session,
    select_connectivity_threshold,
    spike_times_to_binary,
)
from lif_inference.voltage_augmented_learned_lif_connectivity import (
    VoltageAugmentedPerNeuronLIF,
    estimate_surrogate_connectivity_score_sets as estimate_voltage_surrogate_connectivity_score_sets,
    evaluate_connectivity as evaluate_voltage_connectivity,
    load_single_recording_with_voltage,
    plot_score_separation_histogram,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def safe_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        if all(str(key).startswith('module.') for key in state_dict.keys()):
            stripped = {str(key)[7:]: value for key, value in state_dict.items()}
            model.load_state_dict(stripped)
            return
        raise


def infer_family(checkpoint_path, checkpoint):
    if 'voltage_cleaning' in checkpoint or 'recording_summaries' in checkpoint:
        return 'voltage'
    if 'voltage_augmented' in os.path.basename(checkpoint_path).lower():
        return 'voltage'
    return 'spike'


def clip_excluded_bins(excluded_bins, limit):
    if excluded_bins is None:
        return np.array([], dtype=np.int32)
    arr = np.asarray(excluded_bins, dtype=np.int32)
    return arr[(arr >= 0) & (arr < int(limit))]


def build_output_path(checkpoint_path, output_path, overlay_surrogate):
    if output_path:
        return output_path
    checkpoint = Path(checkpoint_path)
    suffix = '_score_separation_surrogate.png' if overlay_surrogate else '_score_separation.png'
    return str(checkpoint.with_name(f'{checkpoint.stem}{suffix}'))


def resolve_arg(value, fallback):
    return fallback if value is None else value


def family_defaults(family):
    if family == 'spike':
        return {
            'dt': 1.0,
            'batch_size': 64,
            'lr': 1e-3,
            'pos_weight': 5.0,
            'l1_lambda': 0.01,
            'pre_context': 50,
            'post_context': 10,
            'warmup': 30,
            'neg_ratio': 1.0,
            'neg_min_distance': 100,
            'val_fraction': 0.2,
        }
    return {
        'dt': 1.0,
        'batch_size': 128,
        'lr': 1e-3,
        'pos_weight': 5.0,
        'l1_lambda': 0.01,
        'pre_context': 50,
        'post_context': 10,
        'warmup': 30,
        'neg_ratio': 1.0,
        'neg_min_distance': 100,
        'val_fraction': 0.2,
        'voltage_lambda': 1.0,
        'mask_pre_ms': 1.0,
        'mask_post_ms': 5.0,
        'peak_threshold_mv': 15.0,
    }


def saved_threshold_info(checkpoint, evaluation_results):
    saved_results = checkpoint.get('results_all', {}) if isinstance(checkpoint, dict) else {}
    threshold = saved_results.get('threshold', evaluation_results.get('threshold'))
    label = checkpoint.get('connectivity_threshold_mode')
    if not label:
        label = checkpoint.get('connectivity_thresholding', {}).get('mode')
    if not label:
        label = saved_results.get('connectivity_threshold_mode')
    if not label:
        label = 'saved'
    return threshold, label


def prepare_spike_context(args, checkpoint):
    defaults = family_defaults('spike')
    dt = float(resolve_arg(args.dt, checkpoint.get('dt', defaults['dt'])))
    model = PerNeuronLIF(
        n_neurons=int(checkpoint['n_neurons']),
        K=int(checkpoint['K']),
        max_delay=int(checkpoint['max_delay']),
        threshold_mode=str(checkpoint.get('threshold_mode', 'adaptive')),
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
    if state_dict is None:
        raise KeyError('Checkpoint is missing model_state_dict/state_dict')
    safe_load_state_dict(model, state_dict)
    model.eval()

    loaded = load_session(args.session, recording_idx=args.recording_idx, dt=dt)
    spike_matrix = spike_times_to_binary(loaded['spike_times'], loaded['duration'], dt=dt)
    limit = int(resolve_arg(args.subsample_t, checkpoint.get('T', spike_matrix.shape[1])))
    limit = min(limit, int(spike_matrix.shape[1]))
    spike_matrix = spike_matrix[:, :limit]
    neighbor_indices = [np.asarray(indices, dtype=np.int64) for indices in checkpoint['neighbor_indices']]
    excluded_bins = clip_excluded_bins(checkpoint.get('excluded_bins'), limit)
    _, true_binary = build_ground_truth(loaded['connections'], int(checkpoint['n_neurons']))
    results, abs_scores, labels, _ = evaluate_spike_connectivity(
        model,
        neighbor_indices,
        true_binary,
        connectivity_threshold_mode='oracle_f1',
    )

    surrogate_kwargs = {
        'spike_matrix': spike_matrix,
        'neighbor_indices': neighbor_indices,
        'n_neurons': int(checkpoint['n_neurons']),
        'K_actual': int(checkpoint['K']),
        'max_delay': int(checkpoint['max_delay']),
        'threshold_mode': str(checkpoint.get('threshold_mode', 'adaptive')),
        'lr': float(resolve_arg(args.lr, defaults['lr'])),
        'batch_size': int(resolve_arg(args.batch_size, defaults['batch_size'])),
        'pos_weight': float(resolve_arg(args.pos_weight, defaults['pos_weight'])),
        'l1_lambda': float(resolve_arg(args.l1_lambda, defaults['l1_lambda'])),
        'pre_context': int(resolve_arg(args.pre_context, defaults['pre_context'])),
        'post_context': int(resolve_arg(args.post_context, defaults['post_context'])),
        'warmup': int(resolve_arg(args.warmup, defaults['warmup'])),
        'neg_ratio': float(resolve_arg(args.neg_ratio, defaults['neg_ratio'])),
        'neg_min_distance': int(resolve_arg(args.neg_min_distance, defaults['neg_min_distance'])),
        'boundaries': [0, limit],
        'excluded_bins': excluded_bins,
        'val_fraction': float(resolve_arg(args.val_fraction, defaults['val_fraction'])),
        'device': args.device,
        'n_surrogates': int(args.n_surrogates),
        'surrogate_epochs': int(args.surrogate_epochs),
        'surrogate_patience': int(args.surrogate_patience),
        'surrogate_min_shift_fraction': float(args.surrogate_min_shift_fraction),
        'surrogate_seed': int(args.surrogate_seed),
    }

    return {
        'family': 'spike',
        'session_name': checkpoint.get('session_name') or os.path.basename(os.path.normpath(args.session)),
        'model': model,
        'results': results,
        'abs_scores': abs_scores,
        'labels': labels,
        'surrogate_fn': estimate_spike_surrogate_connectivity_score_sets,
        'surrogate_kwargs': surrogate_kwargs,
        'title': 'Spike-Only Score Separation',
    }


def prepare_voltage_context(args, checkpoint):
    defaults = family_defaults('voltage')
    cleaning = checkpoint.get('voltage_cleaning', {}) if isinstance(checkpoint, dict) else {}
    dt = float(resolve_arg(args.dt, checkpoint.get('dt', defaults['dt'])))
    mask_pre_ms = float(resolve_arg(args.mask_pre_ms, cleaning.get('mask_pre_ms', defaults['mask_pre_ms'])))
    mask_post_ms = float(resolve_arg(args.mask_post_ms, cleaning.get('mask_post_ms', defaults['mask_post_ms'])))
    peak_threshold_mv = float(resolve_arg(
        args.peak_threshold_mv,
        cleaning.get('peak_threshold_mv', defaults['peak_threshold_mv']),
    ))

    model = VoltageAugmentedPerNeuronLIF(
        n_neurons=int(checkpoint['n_neurons']),
        K=int(checkpoint['K']),
        max_delay=int(checkpoint['max_delay']),
        threshold_mode=str(checkpoint.get('threshold_mode', 'adaptive')),
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
    if state_dict is None:
        raise KeyError('Checkpoint is missing model_state_dict/state_dict')
    safe_load_state_dict(model, state_dict)
    model.eval()

    loaded = load_single_recording_with_voltage(
        args.session,
        recording_idx=args.recording_idx,
        dt=dt,
        mask_pre_ms=mask_pre_ms,
        mask_post_ms=mask_post_ms,
        peak_threshold_mv=peak_threshold_mv,
    )
    limit = int(resolve_arg(args.subsample_t, checkpoint.get('T', loaded['spike_matrix'].shape[1])))
    limit = min(limit, int(loaded['spike_matrix'].shape[1]))
    spike_matrix = loaded['spike_matrix'][:, :limit]
    voltage_matrix = loaded['voltage_matrix'][:, :limit]
    voltage_mask = loaded['voltage_mask'][:, :limit]
    neighbor_indices = [np.asarray(indices, dtype=np.int64) for indices in checkpoint['neighbor_indices']]
    excluded_bins = clip_excluded_bins(checkpoint.get('excluded_bins'), limit)
    true_weights, true_binary = build_ground_truth(loaded['connections'], int(checkpoint['n_neurons']))
    results, abs_scores, labels, _, _, _ = evaluate_voltage_connectivity(
        model,
        neighbor_indices,
        true_binary,
        true_weights,
        connectivity_threshold_mode='oracle_f1',
    )

    surrogate_kwargs = {
        'spike_matrix': spike_matrix,
        'voltage_matrix': voltage_matrix,
        'voltage_mask': voltage_mask,
        'neighbor_indices': neighbor_indices,
        'n_neurons': int(checkpoint['n_neurons']),
        'K_actual': int(checkpoint['K']),
        'max_delay': int(checkpoint['max_delay']),
        'threshold_mode': str(checkpoint.get('threshold_mode', 'adaptive')),
        'lr': float(resolve_arg(args.lr, defaults['lr'])),
        'batch_size': int(resolve_arg(args.batch_size, defaults['batch_size'])),
        'warmup': int(resolve_arg(args.warmup, defaults['warmup'])),
        'pos_weight': float(resolve_arg(args.pos_weight, defaults['pos_weight'])),
        'l1_lambda': float(resolve_arg(args.l1_lambda, defaults['l1_lambda'])),
        'voltage_lambda': float(resolve_arg(
            args.voltage_lambda,
            cleaning.get('voltage_lambda', defaults['voltage_lambda']),
        )),
        'pre_context': int(resolve_arg(args.pre_context, defaults['pre_context'])),
        'post_context': int(resolve_arg(args.post_context, defaults['post_context'])),
        'neg_ratio': float(resolve_arg(args.neg_ratio, defaults['neg_ratio'])),
        'neg_min_distance': int(resolve_arg(args.neg_min_distance, defaults['neg_min_distance'])),
        'boundaries': [0, limit],
        'excluded_bins': excluded_bins,
        'val_fraction': float(resolve_arg(args.val_fraction, defaults['val_fraction'])),
        'device': args.device,
        'n_surrogates': int(args.n_surrogates),
        'surrogate_epochs': int(args.surrogate_epochs),
        'surrogate_patience': int(args.surrogate_patience),
        'surrogate_min_shift_fraction': float(args.surrogate_min_shift_fraction),
        'surrogate_seed': int(args.surrogate_seed),
    }

    return {
        'family': 'voltage',
        'session_name': checkpoint.get('session_name') or os.path.basename(os.path.normpath(args.session)),
        'model': model,
        'results': results,
        'abs_scores': abs_scores,
        'labels': labels,
        'surrogate_fn': estimate_voltage_surrogate_connectivity_score_sets,
        'surrogate_kwargs': surrogate_kwargs,
        'title': 'Voltage-Augmented Score Separation',
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot log10 edge vs non-edge learned-weight separation for a saved spike-only or voltage checkpoint.'
    )
    parser.add_argument(
        '--session',
        required=True,
        help='Session directory containing the saved recording(s) and network_*.npz ground truth.',
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Saved learned-LIF checkpoint (.pt). Supports spike-only and voltage-augmented checkpoints.',
    )
    parser.add_argument(
        '--family',
        choices=['auto', 'spike', 'voltage'],
        default='auto',
        help='Checkpoint family. Defaults to auto-detection from checkpoint contents.',
    )
    parser.add_argument(
        '--recording-idx',
        type=int,
        default=0,
        help='Recording index used when loading one saved recording for surrogate calibration.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Optional output PNG path. Defaults next to the checkpoint.',
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=80,
        help='Number of histogram bins to use.',
    )
    parser.add_argument(
        '--log-floor',
        type=float,
        default=1e-10,
        help='Minimum absolute weight used before applying log10.',
    )
    parser.add_argument(
        '--overlay-surrogate',
        action='store_true',
        help='Estimate circular-shift surrogate null scores and overlay them with a surrogate-FDR cutoff.',
    )
    parser.add_argument('--subsample-t', type=int, default=None,
                        help='Optional time-axis truncation. Defaults to the checkpoint T when available.')
    parser.add_argument('--dt', type=float, default=None,
                        help='Override the saved bin size used when loading the recording.')
    parser.add_argument('--device', default='cpu',
                        help='Device used for surrogate calibration. Defaults to cpu.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Surrogate-fit learning rate. Defaults to the family run_pipeline default.')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Surrogate-fit batch size. Defaults to the family run_pipeline default.')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='Surrogate-fit positive-class weight. Defaults to the family run_pipeline default.')
    parser.add_argument('--l1-lambda', type=float, default=None,
                        help='Surrogate-fit L1 regularization weight. Defaults to the family run_pipeline default.')
    parser.add_argument('--pre-context', type=int, default=None,
                        help='Event window history length in bins. Defaults to the family run_pipeline default.')
    parser.add_argument('--post-context', type=int, default=None,
                        help='Event window look-ahead length in bins. Defaults to the family run_pipeline default.')
    parser.add_argument('--warmup', type=int, default=None,
                        help='Warmup bins ignored at the start of each event window. Defaults to the family run_pipeline default.')
    parser.add_argument('--neg-ratio', type=float, default=None,
                        help='Negative-to-positive event sampling ratio for surrogate fits.')
    parser.add_argument('--neg-min-distance', type=int, default=None,
                        help='Minimum negative-sample distance in bins for surrogate fits.')
    parser.add_argument('--val-fraction', type=float, default=None,
                        help='Validation fraction used during surrogate calibration.')
    parser.add_argument('--surrogate-fdr', type=float, default=0.005,
                        help='Target FDR used to derive the surrogate cutoff overlay.')
    parser.add_argument('--n-surrogates', type=int, default=4,
                        help='Number of surrogate null models used for the overlay.')
    parser.add_argument('--surrogate-epochs', type=int, default=2,
                        help='Epoch cap for each surrogate null model.')
    parser.add_argument('--surrogate-patience', type=int, default=1,
                        help='Early-stopping patience for surrogate null models.')
    parser.add_argument('--surrogate-min-shift-fraction', type=float, default=0.10,
                        help='Minimum circular-shift fraction used to generate surrogate recordings.')
    parser.add_argument('--surrogate-seed', type=int, default=1234,
                        help='Random seed used for surrogate calibration.')
    parser.add_argument('--voltage-lambda', type=float, default=None,
                        help='Voltage reconstruction loss weight used for voltage-family surrogate fits.')
    parser.add_argument('--mask-pre-ms', type=float, default=None,
                        help='Voltage masking window before each spike when reloading voltage traces.')
    parser.add_argument('--mask-post-ms', type=float, default=None,
                        help='Voltage masking window after each spike when reloading voltage traces.')
    parser.add_argument('--peak-threshold-mv', type=float, default=None,
                        help='Voltage peak clipping threshold used when reloading voltage traces.')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    family = infer_family(args.checkpoint, checkpoint) if args.family == 'auto' else args.family
    if family == 'spike':
        context = prepare_spike_context(args, checkpoint)
    else:
        context = prepare_voltage_context(args, checkpoint)

    threshold, threshold_label = saved_threshold_info(checkpoint, context['results'])
    output_path = build_output_path(args.checkpoint, args.output, args.overlay_surrogate)

    surrogate_score_sets = None
    surrogate_info = None
    if args.overlay_surrogate:
        surrogate_score_sets = context['surrogate_fn'](**context['surrogate_kwargs'])
        surrogate_info = select_connectivity_threshold(
            context['labels'],
            context['abs_scores'],
            mode='surrogate_fdr',
            surrogate_score_sets=surrogate_score_sets,
            surrogate_fdr=args.surrogate_fdr,
            default_threshold=threshold if threshold is not None else 0.5,
        )

    plot_score_separation_histogram(
        context['abs_scores'],
        context['labels'],
        output_path,
        threshold=threshold,
        threshold_label=threshold_label,
        title=f'{context["title"]} - {context["session_name"]}',
        bins=args.bins,
        log_floor=args.log_floor,
        surrogate_scores=None if surrogate_score_sets is None else surrogate_score_sets.reshape(-1),
        surrogate_label='Surrogate null',
        surrogate_threshold=None if surrogate_info is None else surrogate_info.get('threshold'),
        surrogate_threshold_label=(
            None if surrogate_info is None else f'surrogate_fdr={args.surrogate_fdr:g}'
        ),
    )

    print(f'Saved score-separation figure: {output_path}')
    print(f'Family={family}')
    print(f'Edges={int(context["labels"].sum())}, Non-edges={int((context["labels"] == 0).sum())}')
    print(f'AUC={float(context["results"]["auc"]):.6f}, AP={float(context["results"]["ap"]):.6f}')
    print(f'Annotated cutoff source={threshold_label}, cutoff={float(threshold):.6g}')
    if surrogate_info is not None:
        print(
            'Surrogate overlay: '
            f'models={int(surrogate_info.get("surrogate_n_models", 0))}, '
            f'edges_per_model={int(surrogate_info.get("surrogate_edges_per_model", 0))}'
        )
        print(
            'Surrogate cutoff: '
            f'{float(surrogate_info.get("threshold", float("nan"))):.6g} '
            f'(estimated_fdr={float(surrogate_info.get("estimated_fdr", float("nan"))):.6g}, '
            f'target={float(args.surrogate_fdr):.6g})'
        )


if __name__ == '__main__':
    main()