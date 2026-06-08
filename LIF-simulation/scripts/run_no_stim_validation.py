import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from lif_simulation.plotting import (
    plot_combined_network_layout,
    plot_raster,
    plot_voltage_traces,
)
from lif_simulation.probes import run_no_stimulation_validation


def build_parser():
    """Build the CLI parser for the no-stimulation validation workflow.

    Args:
        None.

    Returns:
        An ``argparse.ArgumentParser`` configured for the no-stimulation probe.
    """
    parser = argparse.ArgumentParser(description="Run the no-stimulation near-criticality validation workflow.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-duration-ms", type=float, default=10000.0)
    parser.add_argument("--requested-voltage-sample-rate", type=float, default=0.25)
    parser.add_argument("--no-h-current", action="store_true", help="Disable h-current during validation.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for saved validation figures.")
    return parser


def _emit_figures(figures, output_dir=None):
    """Show figures interactively or save them to the requested output directory.

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
    print(f"Saved validation figures to: {output_path}")


def main():
    """Run the no-stimulation validation and emit the standard figures.

    Args:
        None.

    Returns:
        None. The function prints summary metrics and either displays or saves the
        generated figures.
    """
    args = build_parser().parse_args()
    result = run_no_stimulation_validation(
        seed=args.seed,
        use_h_current=not args.no_h_current,
        test_duration_ms=args.test_duration_ms,
        requested_voltage_sample_rate=args.requested_voltage_sample_rate,
    )

    metrics = result["metrics"]
    print("NO_STIM_TEST")
    print(f"use_h_current={metrics['use_h_current']}")
    print(f"external_stimulation_events={metrics['external_stimulation_events']}")
    print(f"test_duration_s={metrics['test_duration_s']:.1f}")
    print(f"requested_voltage_sample_rate_ms={metrics['requested_voltage_sample_rate_ms']:.2f}")
    print(f"saved_voltage_step_ms={metrics['saved_voltage_step_ms']:.2f}")
    print(f"voltage_trace_mode={metrics['voltage_trace_mode']}")
    print(f"raster_window_s={metrics['raster_window_s']:.1f}")
    print(f"n_neurons={metrics['n_neurons']}")
    print(f"total_spikes={metrics['total_spikes']}")
    print(f"active_neurons={metrics['active_neurons']}")
    print(f"mean_rate_hz={metrics['mean_rate_hz']:.4f}")
    print(f"median_rate_hz={metrics['median_rate_hz']:.4f}")
    print(f"max_rate_hz={metrics['max_rate_hz']:.4f}")
    print(f"max_active_fraction_100ms={metrics['max_active_fraction_100ms']:.4f}")
    print(f"bins_ge_10pct={metrics['bins_ge_10pct']}")
    print(f"bins_ge_25pct={metrics['bins_ge_25pct']}")
    print(f"bins_ge_50pct={metrics['bins_ge_50pct']}")
    print(f"verdict={metrics['verdict']}")

    # Metrics summarize the whole run, but the raster is cropped to the quick-look window.
    spike_data = result["spike_data"]
    raster_spike_data = {
        neuron_id: [spike for spike in spikes if spike <= result["raster_window_ms"]]
        for neuron_id, spikes in spike_data.items()
    }
    # Reuse the shared plotting helpers so notebook and CLI validation stay visually aligned.
    figures = []
    figures.append(
        (
            "no_stim_network_structure.png",
            plot_combined_network_layout(
                result["neuron_positions"],
                result["cluster_info"],
                result["connections"],
                cluster_assignments=result["cluster_assignments"],
                max_connections=result["max_connections_plot"],
                title="No-Stimulation Test: Network Structure",
            ),
        )
    )
    figures.append(
        (
            "no_stim_raster.png",
            plot_raster(
                raster_spike_data,
                result["cluster_assignments"],
                result["raster_window_ms"],
                title=f"No-Stimulation Test Raster ({result['raster_window_ms'] / 1000.0:.1f} s shown)",
            ),
        )
    )
    figures.append(
        (
            "no_stim_voltage_traces.png",
            plot_voltage_traces(
                result["voltage_data"],
                neuron_ids=result["example_neuron_ids"],
                time_range=(0, result["voltage_window_ms"]),
                spike_data=result["spike_data"],
                title=f"No-Stimulation Test Raw Voltage Traces ({result['voltage_window_ms'] / 1000.0:.1f} s window)",
            ),
        )
    )
    _emit_figures(figures, args.output_dir)


if __name__ == "__main__":
    main()