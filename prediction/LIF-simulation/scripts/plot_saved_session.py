import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from lif_simulation.analysis import analyze_spike_trains, compute_hub_firing_rate_groups, validate_hub_structure
from lif_simulation.plotting import (
    plot_firing_rates,
    plot_hub_degree_distributions,
    plot_hub_firing_rate_histogram,
    plot_hub_network,
    plot_network_positions,
    plot_raster,
    plot_resampled_raster,
    plot_spike_train_analysis_summary,
    plot_voltage_heatmap,
    plot_voltage_traces,
)
from lif_simulation.session_views import load_session_bundle


def build_parser():
    """Build the CLI parser for loading and plotting a saved session.

    Args:
        None.

    Returns:
        An ``argparse.ArgumentParser`` configured for the saved-session plotting
        workflow.
    """
    parser = argparse.ArgumentParser(description="Load and visualize a saved session from the modular pipeline.")
    parser.add_argument("session_source", nargs="?", default="latest", help="Session folder, metadata file, or 'latest'.")
    parser.add_argument("--base-dir", default="LIF data")
    parser.add_argument("--recording-index", type=int, default=0)
    parser.add_argument("--output-dir", default=None, help="Optional directory for saved figures.")
    return parser


def _save_or_show(figures, output_dir=None):
    """Show figures interactively or save the generated figure bundle to disk.

    Args:
        figures: Iterable of ``(filename, figure)`` tuples.
        output_dir: Optional directory where figures should be written.

    Returns:
        None. Figures are either displayed or saved to disk.
    """
    if output_dir is None:
        plt.show()
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        fig.savefig(output_path / name, dpi=150, bbox_inches="tight")
    print(f"Saved figures to: {output_path}")


def main():
    """Load a saved session bundle and generate the standard analysis figures.

    Args:
        None.

    Returns:
        None. The function prints a session summary and either displays or saves
        the generated figures.
    """
    args = build_parser().parse_args()
    bundle = load_session_bundle(
        session_source=args.session_source,
        base_dir=args.base_dir,
        recording_index=args.recording_index,
    )

    print(f"Loading session: {bundle['timestamp']}")
    print(f"Session folder: {bundle['session_dir']}")
    print(f"Relative session path: {bundle['relative_session_path']}")
    print(f"Neuron 0 spiked {len(bundle['spike_times'][0])} times")
    if len(bundle['spike_times'][0]) > 0:
        print(f"  First 10 spikes: {list(bundle['spike_times'][0])[:10]}")

    # Surface voltage metadata because newer sessions store raw full-dt traces while older ones may not.
    if bundle['voltage_data'] is not None:
        voltage_trace_mode = bundle['metadata'].get('voltage_trace_mode', 'legacy_sampled')
        requested_voltage_sample_rate = bundle['metadata'].get('requested_voltage_sample_rate')
        print(
            f"Voltage traces: {bundle['voltage_data']['traces'].shape[0]} neurons x "
            f"{bundle['voltage_data']['traces'].shape[1]} time points"
        )
        print(f"Stored voltage step: {bundle['voltage_data']['sample_rate']:.4f} ms")
        print(f"Voltage trace mode: {voltage_trace_mode}")
        if requested_voltage_sample_rate is not None:
            print(f"Requested voltage_sample_rate: {float(requested_voltage_sample_rate):.4f} ms")

    if bundle['stimulation_enabled']:
        print(f"Stimulus onsets loaded: {len(bundle['burst_onset_times'])}")
    else:
        print("No stimulus onsets stored for this recording.")

    print("\n--- Generating Plots ---")
    # Build the standard figure bundle used by the modular notebook path.
    figures = []
    figures.append(
        (
            "session_raster.png",
            plot_raster(
                bundle['spike_data_dict'],
                bundle['cluster_assignments'],
                bundle['recording_duration'],
                title=(
                    f"Recording - Raster Plot ({bundle['recording_duration'] / 1000:.0f}s, "
                    f"{'stimulus-driven' if bundle['stimulation_enabled'] else 'spontaneous firing'})"
                ),
            ),
        )
    )

    if bundle['voltage_data'] is not None:
        figures.append(
            (
                "session_voltage_traces.png",
                plot_voltage_traces(
                    bundle['voltage_data'],
                    neuron_ids=[0, 5, 10, 15, 20, 21, 22, 23, 24, 25],
                    time_range=(0, 2000),
                    spike_data=bundle['spike_data_dict'],
                    title="Stored Voltage Traces (0-2000 ms)",
                ),
            )
        )
        figures.append(
            (
                "session_voltage_heatmap.png",
                plot_voltage_heatmap(
                    bundle['voltage_data'],
                    time_range=(0, 2000),
                    cluster_assignments=bundle['cluster_assignments'],
                    title="Network Voltage Heatmap (0-2 seconds)",
                ),
            )
        )

    figures.append(
        (
            "session_firing_rates.png",
            plot_firing_rates(
                bundle['spike_data_dict'],
                bundle['cluster_assignments'],
                bundle['recording_duration'],
            ),
        )
    )
    figures.append(
        (
            "session_network_layout.png",
            plot_network_positions(
                bundle['network_positions'],
                bundle['network_cluster_info'],
                bundle['network_connections'],
                cluster_assignments=bundle['cluster_assignments'],
                show_connections='sample',
                max_connections=500,
            ),
        )
    )
    figures.append(
        (
            "session_resampled_raster.png",
            plot_resampled_raster(
                bundle['resampled_spikes'],
                bundle['resampled_times'],
                bundle['cluster_assignments'],
                bundle['resampling_frequency'],
                burst_onset_times=bundle['burst_onset_times'],
            ),
        )
    )

    analysis_results = analyze_spike_trains(
        bundle['spike_data_dict'],
        bundle['cluster_assignments'],
        bundle['recording_duration'],
        bundle['network_connections'],
        burst_onset_times=bundle['burst_onset_times'],
    )
    figures.append(("session_spike_train_analysis.png", plot_spike_train_analysis_summary(analysis_results)))

    # Hub analysis is optional because some historical sessions do not store hub metadata.
    if bundle['hub_cluster_info'] is not None:
        print("\n--- Hub Structure Validation ---")
        figures.append(
            (
                "session_hub_network.png",
                plot_hub_network(
                    bundle['network_positions'],
                    bundle['network_connections'],
                    bundle['hub_cluster_info'],
                )[0],
            )
        )
        hub_stats = validate_hub_structure(
            bundle['network_connections'],
            bundle['hub_cluster_info'],
            len(bundle['network_positions']),
        )
        if hub_stats is not None:
            figures.append(("session_hub_degree_distributions.png", plot_hub_degree_distributions(hub_stats)))
            rate_groups = compute_hub_firing_rate_groups(
                bundle['spike_data_dict'],
                bundle['hub_cluster_info']['hub_neuron_ids'],
                bundle['recording_duration'],
            )
            figures.append(("session_hub_firing_rates.png", plot_hub_firing_rate_histogram(rate_groups)))
            print(f"Hub neuron mean firing rate: {rate_groups['hub_mean_rate']:.2f} Hz")
            print(f"Non-hub neuron mean firing rate: {rate_groups['non_hub_mean_rate']:.2f} Hz")
            print(
                "Firing rate ratio (hub/non-hub): "
                f"{rate_groups['rate_ratio']:.2f}x"
                if np.isfinite(rate_groups['rate_ratio'])
                else "Firing rate ratio (hub/non-hub): inf"
            )

    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Session: {bundle['timestamp']}")
    print(f"Recording duration: {bundle['recording_duration'] / 1000:.1f} seconds")
    print(f"Number of neurons: {len(bundle['spike_times'])}")
    print(f"Number of clusters: {len(bundle['network_cluster_info']['cluster_neuron_groups'])}")
    print(f"Total connections: {len(bundle['network_connections'])}")
    print(f"Resampling frequency: {bundle['resampling_frequency']:.1f} Hz")
    total_spikes = sum(len(spikes) for spikes in bundle['spike_data_dict'].values())
    mean_rate = total_spikes / len(bundle['spike_times']) / (bundle['recording_duration'] / 1000.0)
    print(f"Total spikes: {total_spikes}")
    print(f"Mean firing rate: {mean_rate:.2f} Hz")
    print("=" * 60)

    _save_or_show(figures, args.output_dir)


if __name__ == "__main__":
    import numpy as np

    main()