"""Voltage-augmented surrogate threshold calibration helpers."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .connectivity_metrics import flatten_candidate_scores
from .shared_data import (
    build_interval_jitter_surrogates,
    build_segmentwise_circular_shift_surrogates,
)


def estimate_surrogate_connectivity_score_sets(
        model_class, dataset_builder, train_epoch_fn, evaluate_fn,
        spike_matrix, voltage_matrix, voltage_mask,
        neighbor_indices, n_neurons, K_actual, max_delay,
        threshold_mode, lr, batch_size, warmup,
        pos_weight, l1_lambda, voltage_lambda,
        pre_context=50, post_context=10,
        neg_ratio=1.0, neg_min_distance=100,
        boundaries=None, excluded_bins=None, val_fraction=0.2,
        device='cpu', n_surrogates=4, surrogate_epochs=2,
        surrogate_patience=1, surrogate_min_shift_fraction=0.10,
        surrogate_seed=1234, model_kwargs=None,
        surrogate_null='circular_shift', jitter_bins=25):
    """Fit lightweight null models on surrogate spike+voltage data.

    ``surrogate_null='circular_shift'`` (default) rigidly rotates spikes, voltage,
    and mask together per neuron. ``'interval_jitter'`` resamples only the spike
    trains within ``jitter_bins``-wide windows and keeps the real voltage and
    mask: this preserves the population common-input envelope (the source of the
    voltage model's false positives) while still destroying monosynaptic timing,
    so the null reproduces the real FP floor instead of sitting far below it.
    """
    surrogate_null = str(surrogate_null).strip().lower()
    if surrogate_null not in {'circular_shift', 'interval_jitter'}:
        raise ValueError(
            f"Unsupported surrogate_null={surrogate_null!r}; "
            "use 'circular_shift' or 'interval_jitter'"
        )
    rng = np.random.default_rng(surrogate_seed)
    all_neuron_ids = np.arange(n_neurons)
    score_sets = []
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    for surrogate_idx in range(int(n_surrogates)):
        if surrogate_null == 'interval_jitter':
            # Jitter spikes only; keep the real voltage/mask so the common-input
            # envelope that drives false positives is preserved in the null.
            surrogate_spike_matrix, = build_interval_jitter_surrogates(
                [spike_matrix],
                boundaries=boundaries,
                rng=rng,
                jitter_bins=jitter_bins,
            )
            surrogate_voltage_matrix = voltage_matrix
            surrogate_voltage_mask = voltage_mask
        else:
            surrogate_spike_matrix, surrogate_voltage_matrix, surrogate_voltage_mask = (
                build_segmentwise_circular_shift_surrogates(
                    [spike_matrix, voltage_matrix, voltage_mask],
                    boundaries=boundaries,
                    rng=rng,
                    min_shift_fraction=surrogate_min_shift_fraction,
                )
            )
        dataset_seed = surrogate_seed + 1000 * (surrogate_idx + 1)
        train_ds, val_ds, _ = dataset_builder(
            surrogate_spike_matrix,
            surrogate_voltage_matrix,
            surrogate_voltage_mask,
            neighbor_indices,
            all_neuron_ids,
            pre_context=pre_context,
            post_context=post_context,
            warmup=warmup,
            neg_ratio=neg_ratio,
            neg_min_distance=neg_min_distance,
            boundaries=boundaries,
            excluded_bins=excluded_bins,
            val_fraction=val_fraction,
            rng_seed=dataset_seed,
        )
        if len(train_ds) == 0:
            raise RuntimeError('Surrogate threshold calibration produced zero training windows')

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        torch.manual_seed(surrogate_seed + surrogate_idx)
        surrogate_model = model_class(
            n_neurons=n_neurons,
            K=K_actual,
            max_delay=max_delay,
            threshold_mode=threshold_mode,
            **model_kwargs,
        ).to(device)
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=lr, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0
        max_epochs = max(int(surrogate_epochs), 1)
        max_patience = max(int(surrogate_patience), 1)

        for _ in range(max_epochs):
            train_epoch_fn(
                surrogate_model, train_loader, optimizer, device, warmup,
                pos_weight, l1_lambda, voltage_lambda,
            )
            val_stats = evaluate_fn(
                surrogate_model, val_loader, device, warmup,
                pos_weight, l1_lambda, voltage_lambda,
            )

            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                best_state = {key: value.detach().clone() for key, value in surrogate_model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= max_patience:
                break

        if best_state is not None:
            surrogate_model.load_state_dict(best_state)

        surrogate_conn_matrix = surrogate_model.get_connectivity_matrix(neighbor_indices)
        score_sets.append(
            flatten_candidate_scores(
                surrogate_conn_matrix,
                neighbor_indices,
                neuron_ids=all_neuron_ids,
                absolute=True,
            )
        )

    return np.stack(score_sets, axis=0)