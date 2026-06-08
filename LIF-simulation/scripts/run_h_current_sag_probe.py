import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from lif_simulation.plotting import plot_sag_probe
from lif_simulation.probes import run_h_current_step_probe, summarize_h_current_step_probe


def build_parser():
    """Build the CLI parser for the h-current sag and rebound probe.

    Args:
        None.

    Returns:
        An ``argparse.ArgumentParser`` configured for the sag/rebound probe.
    """
    parser = argparse.ArgumentParser(description="Run the h-current sag/rebound probe.")
    parser.add_argument("--step-current-na", type=float, default=-0.12)
    parser.add_argument("--duration-ms", type=float, default=1200.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output-dir", default=None, help="Optional directory for saved probe figures.")
    return parser


def main():
    """Compare probe responses with and without h-current and emit the figure.

    Args:
        None.

    Returns:
        None. The function prints summary metrics and either displays or saves the
        diagnostic figure.
    """
    args = build_parser().parse_args()
    # Probe the same current step with and without h-current to isolate sag/rebound effects.
    sag_with_h = run_h_current_step_probe(
        use_h_current=True,
        step_current_nA=args.step_current_na,
        duration_ms=args.duration_ms,
        dt=args.dt,
    )
    sag_without_h = run_h_current_step_probe(
        use_h_current=False,
        step_current_nA=args.step_current_na,
        duration_ms=args.duration_ms,
        dt=args.dt,
    )
    summary = summarize_h_current_step_probe(sag_with_h, sag_without_h)

    print("H_CURRENT_SAG_TEST")
    print(f"step_current_nA={summary['step_current_nA']:.3f}")
    print(f"sag_depth_with_h_mV={summary['sag_depth_with_h_mV']:.3f}")
    print(f"sag_depth_without_h_mV={summary['sag_depth_without_h_mV']:.3f}")
    print(f"rebound_with_h_mV={summary['rebound_with_h_mV']:.3f}")
    print(f"rebound_without_h_mV={summary['rebound_without_h_mV']:.3f}")

    fig = plot_sag_probe(sag_with_h, sag_without_h)
    if args.output_dir is None:
        plt.show()
    else:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / "h_current_sag_probe.png", dpi=150, bbox_inches="tight")
        print(f"Saved probe figure to: {output_path / 'h_current_sag_probe.png'}")


if __name__ == "__main__":
    main()