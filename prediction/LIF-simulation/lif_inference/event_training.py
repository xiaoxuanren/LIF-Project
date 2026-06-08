"""Spike-only event-window loss and training helpers."""

import torch
import torch.nn.functional as F


def compute_event_loss(spike_probs, post_spikes, weights, warmup,
                       pos_weight=5.0, l1_lambda=0.01):
    """Compute event-window loss on the non-warmup region only."""
    spike_region = spike_probs[:, warmup:]
    post_region = post_spikes[:, warmup:]

    weight_mask = torch.where(post_region == 1, pos_weight, 1.0)
    spike_loss = F.binary_cross_entropy(
        spike_region.clamp(1e-7, 1 - 1e-7), post_region, weight=weight_mask
    )
    l1_loss = l1_lambda * weights.abs().mean()
    return spike_loss + l1_loss, spike_loss.item(), l1_loss.item()


def train_epoch_events(model, dataloader, optimizer, device, pos_weight,
                       l1_lambda, warmup):
    """Run one event-window training epoch for the spike-only model."""
    model.train()
    total_loss = 0
    total_spike = 0
    total_l1 = 0
    n_batches = 0

    for pre_spikes, post_spikes, neuron_ids, is_positive in dataloader:
        pre_spikes = pre_spikes.to(device)
        post_spikes = post_spikes.to(device)
        neuron_ids = neuron_ids.to(device)

        optimizer.zero_grad()
        window_len = pre_spikes.shape[2]
        spike_probs, voltages, weights = model(
            pre_spikes, post_spikes, neuron_ids, tbptt_len=window_len
        )

        loss, spike_loss, l1_loss = compute_event_loss(
            spike_probs, post_spikes, weights, warmup, pos_weight, l1_lambda
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_spike += spike_loss
        total_l1 += l1_loss
        n_batches += 1

    return (
        total_loss / max(n_batches, 1),
        total_spike / max(n_batches, 1),
        total_l1 / max(n_batches, 1),
    )


@torch.no_grad()
def evaluate_event_windows(model, dataloader, device, pos_weight,
                           l1_lambda, warmup):
    """Evaluate spike prediction on held-out event windows."""
    model.eval()
    total_loss = 0
    total_spike = 0
    total_l1 = 0
    n_batches = 0

    for pre_spikes, post_spikes, neuron_ids, is_positive in dataloader:
        pre_spikes = pre_spikes.to(device)
        post_spikes = post_spikes.to(device)
        neuron_ids = neuron_ids.to(device)

        window_len = pre_spikes.shape[2]
        spike_probs, voltages, weights = model(
            pre_spikes, post_spikes, neuron_ids, tbptt_len=window_len
        )
        loss, spike_loss, l1_loss = compute_event_loss(
            spike_probs, post_spikes, weights, warmup, pos_weight, l1_lambda
        )

        total_loss += loss.item()
        total_spike += spike_loss
        total_l1 += l1_loss
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'spike_loss': total_spike / max(n_batches, 1),
        'l1_loss': total_l1 / max(n_batches, 1),
        'n_batches': n_batches,
        'n_windows': len(dataloader.dataset),
    }