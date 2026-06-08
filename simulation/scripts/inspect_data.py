import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lif_simulation.session_io import load_single_recording
from lif_simulation.voltage_storage import resolve_recording_voltage


DEFAULT_DATA_PATH = Path('LIF data/h_current_ablation_60s_raw_voltage/20260519_123510/recording000.npz')
WINDOW_MS = 5000.0


def main():
    """Inspect one saved recording and plot representative raw voltage traces.

    Args:
        None.

    Returns:
        None. The function loads a saved recording, prints a short voltage summary,
        and writes an example figure beside the recording file.
    """
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f'Recording file not found: {data_path}')

    data = load_single_recording(str(data_path))
    voltage_bundle = resolve_recording_voltage(data, recording_path=str(data_path), load_into_memory=False)
    if voltage_bundle is None:
        raise KeyError(f'No voltage data found in {data_path}')

    voltage_traces = voltage_bundle['traces']
    voltage_times = voltage_bundle['times']
    spike_times = data['spike_times']
    voltage_step_ms = float(voltage_bundle['sample_rate']) if voltage_bundle['sample_rate'] is not None else np.nan

    # Inspect three representative neurons: most active, typical active, and silent when available.
    counts = np.array([len(spikes) for spikes in spike_times], dtype=int)

    highest_idx = int(np.argmax(counts))
    nonzero_indices = np.where(counts > 0)[0]
    if len(nonzero_indices) > 0:
        median_val = np.median(counts[nonzero_indices])
        median_idx = int(nonzero_indices[np.argmin(np.abs(counts[nonzero_indices] - median_val))])
    else:
        median_idx = 0

    silent_indices = np.where(counts == 0)[0]
    silent_idx = int(silent_indices[0]) if len(silent_indices) > 0 else None

    selected_ids = [highest_idx, median_idx]
    if silent_idx is not None and silent_idx not in selected_ids:
        selected_ids.append(silent_idx)

    end_idx = int(np.searchsorted(voltage_times, WINDOW_MS, side='right'))
    plot_times = voltage_times[:end_idx]

    plt.figure(figsize=(10, 8))
    for plot_idx, neuron_idx in enumerate(selected_ids):
        trace_window = np.asarray(voltage_traces[neuron_idx, :end_idx], dtype=np.float32)
        neuron_spikes = np.asarray(spike_times[neuron_idx], dtype=float)
        spikes_in_window = neuron_spikes[neuron_spikes <= WINDOW_MS]

        plt.subplot(len(selected_ids), 1, plot_idx + 1)
        plt.plot(plot_times, trace_window, color='navy', linewidth=0.8)
        plt.axhline(-50.0, color='firebrick', linestyle='--', linewidth=0.6, alpha=0.5)
        if len(spikes_in_window) > 0:
            # Raw stored voltage does not contain synthetic spike glyphs, so mark spikes above the trace.
            spike_marker_y = max(float(trace_window.max()) + 1.5, -48.0)
            plt.scatter(
                spikes_in_window,
                np.full(len(spikes_in_window), spike_marker_y),
                color='red',
                marker='v',
                s=10,
                alpha=0.7,
            )

        plt.title(
            f'Neuron {neuron_idx} (Total spikes: {counts[neuron_idx]}, spikes in first 5 s: {len(spikes_in_window)})'
        )
        plt.ylabel('Voltage (mV)')

        approaches_threshold = bool(np.any(trace_window >= -50.5))

        print(f'Neuron {neuron_idx}:')
        print(f'  Total Spike Count: {counts[neuron_idx]}')
        print(f'  Spikes in First 5 s: {len(spikes_in_window)}')
        print(f'  Min/Max (first 5 s): {trace_window.min():.2f} / {trace_window.max():.2f}')
        print(f'  Approaches Threshold in Window: {approaches_threshold}')

    output_path = data_path.parent / 'example_raw_voltage_traces_5s.png'
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f'Voltage Step (ms): {voltage_step_ms:.4f}')
    print(f'Saved Figure: {output_path}')


if __name__ == '__main__':
    main()
