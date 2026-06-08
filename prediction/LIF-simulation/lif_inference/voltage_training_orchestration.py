"""High-level voltage event-window training orchestration helpers."""

import time

import torch


def train_voltage_model_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        train_epoch_fn,
        evaluate_windows_fn,
        evaluate_connectivity_fn,
        neighbor_indices,
        true_binary,
        true_weights,
        device,
        warmup,
        pos_weight,
        l1_lambda,
        voltage_lambda,
        n_epochs,
        patience,
        log_every=5,
        log_fn=print,
        progress_fn=None,
        completion_fn=None):
    """Train the voltage-augmented model with held-out window early stopping."""
    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    train_history = []
    val_history = []
    conn_aucs = []
    start_time = time.time()

    max_patience = None if patience is None else max(int(patience), 1)
    log_interval = max(int(log_every), 1)

    for epoch in range(int(n_epochs)):
        train_stats = train_epoch_fn(
            model, train_loader, optimizer, device, warmup,
            pos_weight, l1_lambda, voltage_lambda,
        )
        train_history.append(train_stats)

        val_stats = evaluate_windows_fn(
            model, val_loader, device, warmup,
            pos_weight, l1_lambda, voltage_lambda,
        )
        val_history.append(val_stats)

        conn_results, _, _, _, _, _ = evaluate_connectivity_fn(
            model, neighbor_indices, true_binary, true_weights,
        )
        conn_aucs.append(conn_results['auc'])

        if scheduler is not None:
            scheduler.step(val_stats['loss'])

        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        elapsed = time.time() - start_time
        if progress_fn is not None:
            progress_fn(
                epoch=epoch + 1,
                n_epochs=int(n_epochs),
                train_stats=train_stats,
                val_stats=val_stats,
                conn_results=conn_results,
                model=model,
                elapsed_seconds=elapsed,
            )
        elif log_fn is not None and (((epoch + 1) % log_interval == 0) or epoch == 0):
            alpha = torch.sigmoid(model.alpha_logit).item()
            thresh_inc = model.threshold_increment.mean().item()
            thresh_decay = model.threshold_decay.item()
            log_fn(
                f'    Epoch {epoch + 1:3d}: '
                f'train={train_stats["loss"]:.4f} '
                f'(spike={train_stats["spike_loss"]:.4f} voltage={train_stats["voltage_loss"]:.4f} l1={train_stats["l1_loss"]:.4f}) '
                f'val={val_stats["loss"]:.4f} conn_AUC={conn_results["auc"]:.4f} '
                f'alpha={alpha:.3f} theta_mode={model.threshold_mode} theta0={model.threshold.item():.3f} eta={thresh_inc:.3f} rho={thresh_decay:.3f} ({elapsed:.0f}s)'
            )

        if max_patience is not None and epochs_no_improve >= max_patience:
            if log_fn is not None:
                log_fn(f'    Early stopping at epoch {epoch + 1}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_seconds = time.time() - start_time
    if completion_fn is not None:
        completion_fn(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            elapsed_seconds=elapsed_seconds,
            model=model,
        )
    elif log_fn is not None:
        log_fn(f'  Done in {elapsed_seconds:.0f}s, best val loss={best_val_loss:.4f}')

    val_window_results = evaluate_windows_fn(
        model, val_loader, device, warmup,
        pos_weight, l1_lambda, voltage_lambda,
    )

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_history': train_history,
        'val_history': val_history,
        'conn_aucs': conn_aucs,
        'val_window_results': val_window_results,
        'elapsed_seconds': elapsed_seconds,
    }