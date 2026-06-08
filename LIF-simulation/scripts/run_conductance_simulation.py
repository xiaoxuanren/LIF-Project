import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lif_simulation.workflows import sequential_simulation_individual_saves


def build_parser():
    """Build the CLI parser for the conductance-based simulation runner.

    Args:
        None.

    Returns:
        An ``argparse.ArgumentParser`` configured for the shared simulation
        workflow.
    """
    parser = argparse.ArgumentParser(
        description="Run the conductance-based clustered LIF simulation pipeline.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--mode", choices=["spontaneous", "stimulus_driven"], default="spontaneous")
    parser.add_argument("--save-dir", default="LIF data")
    parser.add_argument("--n-recordings", type=int, default=20)
    parser.add_argument("--recording-duration", type=float, default=60000.0)
    parser.add_argument("--num-clusters", type=int, default=30)
    parser.add_argument("--min-neurons-per-cluster", type=int, default=12)
    parser.add_argument("--max-neurons-per-cluster", type=int, default=18)
    parser.add_argument("--inhibitory-probability", type=float, default=0.2)
    parser.add_argument("--within-cluster-prob", type=float, default=0.3)
    parser.add_argument("--between-cluster-prob", type=float, default=0.15)
    parser.add_argument("--target-freq", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--voltage-sample-rate", type=float, default=1.0)
    parser.add_argument(
        "--voltage-storage-backend",
        choices=["hdf5_external", "inline_npz"],
        default="hdf5_external",
        help="Storage backend for saved voltage traces.",
    )
    parser.add_argument(
        "--voltage-chunk-samples",
        type=int,
        default=4096,
        help="Number of timesteps buffered before each HDF5 flush.",
    )
    parser.add_argument("--space-size", type=float, default=15.0)
    parser.add_argument("--max-connection-distance", type=float, default=6.0)
    parser.add_argument("--no-h-current", action="store_true", help="Disable h-current dynamics.")
    parser.add_argument("--burst-interval", type=float, default=7000.0)
    parser.add_argument("--cluster-fraction", type=float, default=0.7)
    parser.add_argument("--neurons-per-cluster", type=int, default=6)
    parser.add_argument("--stim-amplitude-min", type=float, default=2.0)
    parser.add_argument("--stim-amplitude-max", type=float, default=3.5)
    parser.add_argument("--stim-duration-min", type=float, default=10.0)
    parser.add_argument("--stim-duration-max", type=float, default=30.0)
    parser.add_argument("--burst-interval-jitter", type=float, default=1500.0)
    parser.add_argument("--hub-fraction", type=float, default=0.1)
    parser.add_argument("--hub-between-prob", type=float, default=0.25)
    parser.add_argument("--hub-weight-scale", type=float, default=1.5)
    parser.add_argument("--hub-reciprocal-factor", type=float, default=2.0)
    return parser


def main():
    """Parse CLI arguments, configure the run mode, and launch the workflow.

    Args:
        None.

    Returns:
        None. The function executes the simulation workflow and prints the saved
        session-metadata location.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Treat spontaneous mode as a no-input control by disabling burst scheduling entirely.
    if args.mode == "spontaneous":
        burst_interval = 12_000_000_000.0
        cluster_fraction = 0.0
        neurons_per_cluster = 0
    else:
        burst_interval = args.burst_interval
        cluster_fraction = args.cluster_fraction
        neurons_per_cluster = args.neurons_per_cluster

    # Keep the CLI thin: the shared workflow owns network construction, simulation, and saving.
    session_metadata = sequential_simulation_individual_saves(
        n_recordings=args.n_recordings,
        recording_duration=args.recording_duration,
        num_clusters=args.num_clusters,
        neurons_per_cluster_range=(args.min_neurons_per_cluster, args.max_neurons_per_cluster),
        inhibitory_probability=args.inhibitory_probability,
        within_cluster_prob=args.within_cluster_prob,
        between_cluster_prob=args.between_cluster_prob,
        target_freq=args.target_freq,
        save_dir=args.save_dir,
        dt=args.dt,
        record_voltage=True,
        voltage_sample_rate=args.voltage_sample_rate,
        voltage_storage_backend=args.voltage_storage_backend,
        voltage_chunk_samples=args.voltage_chunk_samples,
        space_size=args.space_size,
        max_connection_distance=args.max_connection_distance,
        use_h_current=not args.no_h_current,
        burst_interval=burst_interval,
        cluster_fraction=cluster_fraction,
        neurons_per_cluster=neurons_per_cluster,
        stim_amplitude_range=(args.stim_amplitude_min, args.stim_amplitude_max),
        stim_duration_range=(args.stim_duration_min, args.stim_duration_max),
        burst_interval_jitter=args.burst_interval_jitter,
        hub_fraction=args.hub_fraction,
        hub_between_prob=args.hub_between_prob,
        hub_weight_scale=args.hub_weight_scale,
        hub_reciprocal_factor=args.hub_reciprocal_factor,
    )

    print("Session metadata file:")
    print(f"  {session_metadata['session_dir']}\\session_metadata.json")


if __name__ == "__main__":
    main()