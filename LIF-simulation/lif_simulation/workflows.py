import json
import os
from datetime import datetime

from .analysis import report_network_statistics, segment_states
from .models import NetworkWeightParameters
from .network import assign_baseline_drive, create_clustered_network, scale_adaptation_dynamics, scale_excitatory_weights
from .nc_sim_path import ensure_nc_sim_importable
from .session_io import save_network_structure, save_recording_data
from .simulation import simulate_network
from .stimulation import create_periodic_cluster_stimulation
from .voltage_storage import ChunkedHdf5VoltageRecorder


def _obstacles_to_jsonable(obstacles):
    """Return a JSON-serializable view of an nc_sim obstacle grid.

    Accepts ``None``, a plain nested list, or a NumPy array and returns a value
    ``json.dump`` can serialize without importing NumPy here.
    """
    if obstacles is None:
        return None
    if hasattr(obstacles, "tolist"):
        return obstacles.tolist()
    return obstacles


def sequential_simulation_individual_saves(
    n_recordings=15,
    recording_duration=60000,
    num_clusters=20,
    neurons_per_cluster_range=(12, 18),
    inhibitory_probability=0.2,
    within_cluster_prob=0.5,
    between_cluster_prob=0.15,
    target_freq=10,
    save_dir="LIF data",
    dt=0.1,
    record_voltage=True,
    voltage_sample_rate=1.0,
    voltage_storage_backend="hdf5_external",
    voltage_chunk_samples=4096,
    space_size=15,
    max_connection_distance=8.0,
    use_h_current=True,
    background_noise_sigma=0.0,
    spontaneous_baseline_mean=0.11,
    spontaneous_baseline_sd=0.05,
    spontaneous_baseline_seed=0,
    spontaneous_baseline_distribution="lognormal",
    spontaneous_noise_sigma=0.0,
    spontaneous_adaptation_tau_scale=3.0,
    spontaneous_adaptation_increment_scale=1.0,
    spontaneous_exc_weight_scale=0.65,
    spontaneous_burst_frac_thresh=0.12,
    burst_interval=7000,
    cluster_fraction=0.7,
    neurons_per_cluster=6,
    stim_amplitude_range=(2.0, 3.5),
    stim_duration_range=(10, 30),
    burst_interval_jitter=1500,
    hub_fraction=0.1,
    hub_between_prob=0.4,
    hub_weight_scale=1.5,
    hub_reciprocal_factor=2.0,
    network_source="clustered",
    nc_sim_width=1.0,
    nc_sim_height=1.0,
    nc_sim_rho=100.0,
    nc_sim_axon_length=1.0,
    nc_sim_obstacles=None,
):
    """Run a multi-recording simulation session and save each trial to disk.

    Args:
        n_recordings: Number of recordings to generate from the same network.
        recording_duration: Duration of each recording in milliseconds.
        num_clusters: Number of neuronal clusters to generate.
        neurons_per_cluster_range: Inclusive range of neurons sampled per cluster.
        inhibitory_probability: Fraction of neurons designated inhibitory.
        within_cluster_prob: Base connection probability within the same cluster.
        between_cluster_prob: Base connection probability between clusters.
        target_freq: Resampling frequency stored for spike-based analysis outputs.
        save_dir: Root directory used for the session bundle.
        dt: Simulation step in milliseconds.
        record_voltage: Whether to save membrane voltage traces.
        voltage_sample_rate: Legacy voltage sampling interval request.
        voltage_storage_backend: Storage backend used for saved voltage traces.
        voltage_chunk_samples: Number of timesteps buffered per HDF5 chunk flush.
        space_size: Side length of the 2-D spatial layout.
        max_connection_distance: Maximum allowed connection distance.
        use_h_current: Whether to keep the slow h-current update path enabled.
        background_noise_sigma: Standard deviation of the additive membrane-noise
            term assigned to each neuron for stimulus-driven runs. Spontaneous
            mode forces this to zero and uses frozen baseline drive instead.
        spontaneous_baseline_mean: Mean frozen excitatory baseline current used
            when stimulation is disabled.
        spontaneous_baseline_sd: Standard deviation of the frozen baseline draw.
        spontaneous_baseline_seed: Seed for the one-time baseline-current draw.
        spontaneous_exc_weight_scale: Scale applied to recurrent excitatory weights
            when stimulation is disabled.
        spontaneous_burst_frac_thresh: Active-neuron fraction used to detect
            spontaneous burst windows for saved recordings.
        burst_interval: Nominal interval between stimulation bursts in milliseconds.
        cluster_fraction: Fraction of clusters stimulated during each burst.
        neurons_per_cluster: Number of neurons stimulated in each selected cluster.
        stim_amplitude_range: Inclusive stimulus amplitude range in nA.
        stim_duration_range: Inclusive stimulus duration range in milliseconds.
        burst_interval_jitter: Random jitter applied to each burst onset in milliseconds.
        hub_fraction: Fraction of neurons per cluster designated as hubs.
        hub_between_prob: Base inter-cluster connection probability for hub projections.
        hub_weight_scale: Multiplicative weight boost applied to hub-originating edges.
        hub_reciprocal_factor: Probability boost used for hub-to-hub projections.
        network_source: Topology generator to use. ``"clustered"`` (default)
            preserves the existing ``create_clustered_network`` behavior exactly;
            ``"nc_sim"`` grows the topology with the nc_sim spatial axon-growth
            model via ``build_network_from_nc_sim`` (a drop-in replacement). The
            ``nc_sim_*`` arguments are ignored unless this is ``"nc_sim"``.
        nc_sim_width: Culture width in mm passed to nc_sim growth.
        nc_sim_height: Culture height in mm passed to nc_sim growth.
        nc_sim_rho: Neuron density (neurons / mm^2) for nc_sim growth. With
            ``nc_sim_width = nc_sim_height = 1`` this is roughly the neuron count.
        nc_sim_axon_length: Average axon length ``L`` in mm; the structural
            burst-synchrony dial for nc_sim growth.
        nc_sim_obstacles: Optional nc_sim ``H`` obstacle grid (``None`` for a flat
            isotropic culture).

    Returns:
        A session metadata dictionary describing the generated network and recordings.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)

    stimulation_enabled = (
        cluster_fraction > 0
        and neurons_per_cluster > 0
        and burst_interval > 0
        and burst_interval <= recording_duration
    )
    mode_label = "stimulus_driven_bursting" if stimulation_enabled else "spontaneous_firing"
    saved_voltage_dt = float(dt) if record_voltage else None

    print("\n" + "=" * 70)
    print(f"STARTING SEQUENTIAL SIMULATION SESSION: {timestamp}")
    print("=" * 70)
    print(f"Mode: {mode_label}")
    print("Synapse model: conductance-based with legacy weight scaling")
    print(f"H-current enabled: {use_h_current}")
    print(f"Number of recordings: {n_recordings}")
    print(f"Recording duration: {recording_duration / 1000:.0f} seconds")
    print(f"Save directory: {session_dir}")
    print(f"Record voltage: {record_voltage}")
    if record_voltage:
        print(f"Saved voltage resolution: raw full-dt membrane voltage at {saved_voltage_dt:.4f} ms")
        if voltage_storage_backend == "hdf5_external":
            print(f"Voltage storage backend: chunked external HDF5 ({int(voltage_chunk_samples)} samples per flush)")
        else:
            print(f"Voltage storage backend: {voltage_storage_backend}")
        if abs(float(voltage_sample_rate) - saved_voltage_dt) > 1e-12:
            print(
                f"Requested voltage_sample_rate={float(voltage_sample_rate):.4f} ms is ignored for saved traces"
            )
    print(f"Space size: {space_size}")
    print(f"Max connection distance: {max_connection_distance}")
    print(f"Target resampling frequency: {target_freq} Hz")
    print(f"Background membrane noise sigma: {background_noise_sigma}")
    if stimulation_enabled:
        print(f"Burst interval: {burst_interval} ms ({1000 / burst_interval:.2f} Hz) +/- {burst_interval_jitter} ms")
        print(f"Cluster fraction: {cluster_fraction * 100:.0f}%")
        print(f"Neurons per cluster: {neurons_per_cluster}")
    else:
        print("Stimulation: disabled")
        print("Spontaneous drive: frozen excitatory baseline, membrane noise forced to 0")
        print(
            f"Baseline mean={spontaneous_baseline_mean:.3f}, sd={spontaneous_baseline_sd:.3f}, "
            f"seed={spontaneous_baseline_seed}"
        )
        print(f"Excitatory recurrent weight scale: {spontaneous_exc_weight_scale:.3f}")
    print(f"Hub fraction: {hub_fraction * 100:.0f}% | Hub between prob: {hub_between_prob}")
    print(f"Hub weight scale: {hub_weight_scale}x | Hub reciprocal factor: {hub_reciprocal_factor}x")
    print("=" * 70 + "\n")

    print("Creating network...")
    print(f"Network source: {network_source}")
    weight_params = NetworkWeightParameters()
    if network_source == "nc_sim":
        ensure_nc_sim_importable()
        from .nc_sim_adapter import build_network_from_nc_sim

        neurons, synapses, connections, neuron_positions, cluster_info = build_network_from_nc_sim(
            width=nc_sim_width,
            height=nc_sim_height,
            obstacles=nc_sim_obstacles,
            rho=nc_sim_rho,
            axon_length=nc_sim_axon_length,
            num_clusters=num_clusters,
            weight_params=weight_params,
            use_h_current=use_h_current,
            background_noise_sigma=background_noise_sigma if stimulation_enabled else 0.0,
        )
    elif network_source == "clustered":
        neurons, synapses, connections, neuron_positions, cluster_info = create_clustered_network(
            num_clusters=num_clusters,
            neurons_per_cluster_range=neurons_per_cluster_range,
            inhibitory_probability=inhibitory_probability,
            within_cluster_prob=within_cluster_prob,
            between_cluster_prob=between_cluster_prob,
            max_connection_distance=max_connection_distance,
            weight_params=weight_params,
            space_size=space_size,
            hub_fraction=hub_fraction,
            hub_between_prob=hub_between_prob,
            hub_weight_scale=hub_weight_scale,
            hub_reciprocal_factor=hub_reciprocal_factor,
            use_h_current=use_h_current,
            background_noise_sigma=background_noise_sigma if stimulation_enabled else 0.0,
        )
    else:
        raise ValueError(
            f"Unknown network_source={network_source!r}; expected 'clustered' or 'nc_sim'"
        )
    if stimulation_enabled:
        actual_background_noise_sigma = float(background_noise_sigma)
        baseline_currents = [neuron.i_baseline for neuron in neurons]
        exc_weight_scale = 1.0
    else:
        for neuron in neurons:
            neuron.noise_sigma = spontaneous_noise_sigma
        assign_baseline_drive(
            neurons,
            mean=spontaneous_baseline_mean,
            sd=spontaneous_baseline_sd,
            seed=spontaneous_baseline_seed,
            excitatory_only=True,
            distribution=spontaneous_baseline_distribution,
        )
        scale_adaptation_dynamics(
            neurons,
            tau_scale=spontaneous_adaptation_tau_scale,
            increment_scale=spontaneous_adaptation_increment_scale,
        )
        scale_excitatory_weights(synapses, spontaneous_exc_weight_scale, connections)
        actual_background_noise_sigma = float(spontaneous_noise_sigma)
        baseline_currents = [neuron.i_baseline for neuron in neurons]
        exc_weight_scale = float(spontaneous_exc_weight_scale)
        cluster_info.update(
            {
                "baseline_currents": baseline_currents,
                "baseline_drive_mean": float(spontaneous_baseline_mean),
                "baseline_drive_sd": float(spontaneous_baseline_sd),
                "baseline_drive_seed": int(spontaneous_baseline_seed),
                "baseline_excitatory_only": True,
                "exc_weight_scale": exc_weight_scale,
                "noise_sigma": [neuron.noise_sigma for neuron in neurons],
            }
        )
    print("Background presynaptic input: DISABLED")

    network_file = save_network_structure(
        connections,
        neuron_positions,
        cluster_info,
        weight_params,
        timestamp,
        save_dir,
    )

    session_metadata = {
        "timestamp": timestamp,
        "session_dir": session_dir,
        "n_recordings": n_recordings,
        "recording_duration": recording_duration,
        "num_clusters": num_clusters,
        "num_neurons": len(neurons),
        "num_connections": len(connections),
        "target_freq": target_freq,
        "dt": dt,
        "record_voltage": record_voltage,
        "voltage_sample_rate": saved_voltage_dt if record_voltage else None,
        "requested_voltage_sample_rate": float(voltage_sample_rate) if record_voltage else None,
        "voltage_trace_mode": "raw_full_dt" if record_voltage else None,
        "voltage_storage_backend": voltage_storage_backend if record_voltage else None,
        "voltage_chunk_samples": int(voltage_chunk_samples) if record_voltage else None,
        "space_size": space_size,
        "max_connection_distance": max_connection_distance,
        "network_source": network_source,
        "nc_sim_params": {
            "width": float(nc_sim_width),
            "height": float(nc_sim_height),
            "rho": float(nc_sim_rho),
            "axon_length": float(nc_sim_axon_length),
            "obstacles": _obstacles_to_jsonable(nc_sim_obstacles),
            "num_clusters": int(num_clusters),
        },
        "network_file": network_file,
        "mode": mode_label,
        "stimulation_enabled": stimulation_enabled,
        "background_input": False,
        "burst_interval": burst_interval,
        "burst_interval_jitter": burst_interval_jitter,
        "cluster_fraction": cluster_fraction,
        "neurons_per_cluster": neurons_per_cluster,
        "stim_amplitude": stim_amplitude_range,
        "stim_duration": stim_duration_range,
        "hub_fraction": hub_fraction,
        "hub_between_prob": hub_between_prob,
        "hub_weight_scale": hub_weight_scale,
        "hub_reciprocal_factor": hub_reciprocal_factor,
        "n_hub_neurons": len(cluster_info.get("hub_neuron_ids", [])),
        "n_hub_connections": cluster_info.get("n_hub_connections", 0),
        "synapse_model": "conductance-based with legacy weight scaling",
        "use_h_current": bool(use_h_current),
        "requested_background_noise_sigma": float(background_noise_sigma),
        "background_noise_sigma": actual_background_noise_sigma,
        "spontaneous_baseline_mean": float(spontaneous_baseline_mean) if not stimulation_enabled else 0.0,
        "spontaneous_baseline_sd": float(spontaneous_baseline_sd) if not stimulation_enabled else 0.0,
        "spontaneous_baseline_seed": int(spontaneous_baseline_seed) if not stimulation_enabled else None,
        "spontaneous_exc_weight_scale": exc_weight_scale,
        "spontaneous_burst_frac_thresh": float(spontaneous_burst_frac_thresh),
        "h_current_mode": "enabled" if use_h_current else "disabled_skip_update_path",
        "recordings": [],
    }

    for rec_idx in range(n_recordings):
        print(f"\n{'=' * 70}")
        print(f"RECORDING {rec_idx + 1} / {n_recordings}")
        print(f"{'=' * 70}")

        try:
            for neuron in neurons:
                neuron.reset_state()
            for syn in synapses:
                syn.g_syn = 0.0
                syn.pending_spikes = []

            voltage_recorder = None
            voltage_sidecar_path = None
            if record_voltage:
                if voltage_storage_backend == "hdf5_external":
                    n_voltage_samples = int(recording_duration / dt)
                    voltage_sidecar_path = os.path.join(session_dir, f"recording{rec_idx:03d}_voltage.h5")
                    voltage_recorder = ChunkedHdf5VoltageRecorder(
                        voltage_sidecar_path,
                        n_neurons=len(neurons),
                        n_samples=n_voltage_samples,
                        sample_rate_ms=dt,
                        simulation_dt_ms=dt,
                        chunk_samples=voltage_chunk_samples,
                    )
                elif voltage_storage_backend != "inline_npz":
                    raise ValueError(f"Unsupported voltage_storage_backend: {voltage_storage_backend}")

            if stimulation_enabled:
                stimulation_events, burst_onset_times = create_periodic_cluster_stimulation(
                    neurons,
                    cluster_info,
                    burst_interval=burst_interval,
                    cluster_fraction=cluster_fraction,
                    neurons_per_cluster=neurons_per_cluster,
                    stim_amplitude_range=stim_amplitude_range,
                    stim_duration_range=stim_duration_range,
                    simulation_duration=recording_duration,
                    burst_interval_jitter=burst_interval_jitter,
                )
            else:
                stimulation_events, burst_onset_times = [], []

            spike_data, voltage_data = simulate_network(
                neurons,
                synapses,
                stimulation_events,
                dt=dt,
                duration=recording_duration,
                record_voltage=record_voltage,
                voltage_sample_rate=voltage_sample_rate,
                save_raw_voltage=True,
                voltage_recorder=voltage_recorder,
            )

            report_network_statistics(spike_data, neurons, connections, recording_duration)
            burst_windows = None
            interburst_windows = None
            if not stimulation_enabled:
                burst_windows, interburst_windows = segment_states(
                    spike_data,
                    len(neurons),
                    recording_duration,
                    bin_ms=5.0,
                    burst_frac_thresh=spontaneous_burst_frac_thresh,
                    merge_gap_ms=30.0,
                    min_burst_ms=5.0,
                )
                print(
                    f"Detected {len(burst_windows)} burst windows and "
                    f"{len(interburst_windows)} inter-burst windows"
                )
            recording_file = save_recording_data(
                spike_data,
                voltage_data,
                cluster_info,
                rec_idx,
                timestamp,
                save_dir,
                target_freq,
                recording_duration,
                burst_onset_times=burst_onset_times,
                burst_windows=burst_windows,
                interburst_windows=interburst_windows,
            )

            session_metadata["recordings"].append(
                {
                    "index": rec_idx,
                    "file": recording_file,
                    "success": True,
                    "num_spikes": sum(len(spikes) for spikes in spike_data.values()),
                    "n_burst_windows": len(burst_windows) if burst_windows is not None else None,
                    "n_interburst_windows": len(interburst_windows) if interburst_windows is not None else None,
                }
            )
            print(f"Recording {rec_idx + 1} completed successfully!")
        except Exception as exc:
            if 'voltage_recorder' in locals() and voltage_recorder is not None:
                voltage_recorder.close()
            if 'voltage_sidecar_path' in locals() and voltage_sidecar_path and os.path.exists(voltage_sidecar_path):
                os.remove(voltage_sidecar_path)
            print(f"ERROR in recording {rec_idx + 1}: {str(exc)}")
            import traceback

            traceback.print_exc()
            session_metadata["recordings"].append(
                {
                    "index": rec_idx,
                    "file": None,
                    "success": False,
                    "error": str(exc),
                }
            )

    metadata_file = os.path.join(session_dir, "session_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(session_metadata, handle, indent=2)

    print(f"\n{'=' * 70}")
    print("SESSION COMPLETE!")
    print(f"All files saved to: {session_dir}")
    print(f"Session metadata: {metadata_file}")
    print(f"{'=' * 70}\n")
    return session_metadata