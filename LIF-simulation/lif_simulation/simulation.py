import numpy as np


def simulate_network(
    neurons,
    synapses,
    stimulation_events,
    dt=0.1,
    duration=60000,
    record_voltage=True,
    voltage_sample_rate=1.0,
    save_raw_voltage=True,
    voltage_recorder=None,
):
    """Simulate network activity and optionally save membrane voltage traces.

    Args:
        neurons: Sequence of neuron objects updated each timestep.
        synapses: Sequence of synapses that deliver conductance transients.
        stimulation_events: External stimulation tuples sorted by onset time.
        dt: Simulation step in milliseconds.
        duration: Total simulated time in milliseconds.
        record_voltage: Whether to allocate and return voltage traces.
        voltage_sample_rate: Sampling interval used only for legacy downsampled voltage.
        save_raw_voltage: Whether to store every simulation step instead of downsampled traces.
        voltage_recorder: Optional chunked voltage writer used for full-dt storage.

    Returns:
        A tuple of spike-time dictionaries and optional voltage-trace data.
    """
    stimulation_enabled = len(stimulation_events) > 0
    mode_label = (
        "Stimulus-driven bursting" if stimulation_enabled else "Spontaneous activity (no external stimulation)"
    )

    print(f"\nSimulating {duration} ms of network activity...")
    print(f"  Mode: {mode_label}")
    print(f"  External stimulation events: {len(stimulation_events)}")

    num_steps = int(duration / dt)
    synapse_lookup = {}
    for syn in synapses:
        synapse_lookup.setdefault(syn.pre_neuron_id, []).append(syn)
    print(f"  Built synapse lookup: {len(synapse_lookup)} presynaptic neurons")

    spike_data = {neuron.neuron_id: [] for neuron in neurons}
    voltage_data = None
    voltage_sample_steps = None
    n_voltage_samples = None
    voltage_traces = None
    voltage_times = None
    voltage_sample_idx = 0
    saved_voltage_dt = None

    if record_voltage:
        if save_raw_voltage:
            saved_voltage_dt = float(dt)
            n_voltage_samples = num_steps
            voltage_times = np.arange(n_voltage_samples, dtype=np.float32) * saved_voltage_dt
            if voltage_recorder is None:
                voltage_traces = np.zeros((len(neurons), n_voltage_samples), dtype=np.float32)
                print(
                    f"  Voltage recording: raw membrane voltage at every dt step ({saved_voltage_dt:.4f} ms)"
                )
            else:
                print(
                    "  Voltage recording: raw membrane voltage at every dt step "
                    f"({saved_voltage_dt:.4f} ms), chunked to external HDF5"
                )
        else:
            saved_voltage_dt = float(voltage_sample_rate)
            voltage_sample_steps = max(1, int(round(voltage_sample_rate / dt)))
            n_voltage_samples = int(duration / voltage_sample_rate)
            voltage_traces = np.zeros((len(neurons), n_voltage_samples), dtype=np.float32)
            voltage_times = np.arange(n_voltage_samples, dtype=np.float32) * saved_voltage_dt
            print(f"  Voltage recording: downsampled every {saved_voltage_dt:.4f} ms")

    active_stims = {}
    stim_idx = 0
    progress_stride = max(1, num_steps // 10)

    for step in range(num_steps):
        t = step * dt
        if step % progress_stride == 0:
            print(f"  Progress: {100 * step / num_steps:.0f}%")

        while stim_idx < len(stimulation_events) and stimulation_events[stim_idx][0] <= t:
            stim_t, neuron_id, amplitude, duration_stim = stimulation_events[stim_idx]
            active_stims[neuron_id] = (amplitude, stim_t + duration_stim)
            stim_idx += 1

        expired = [nid for nid, (_, end_t) in active_stims.items() if t > end_t]
        for nid in expired:
            del active_stims[nid]

        for neuron in neurons:
            neuron.i_ext = active_stims.get(neuron.neuron_id, (0.0, 0.0))[0]

        for neuron in neurons:
            neuron.g_exc = 0.0
            neuron.g_inh = 0.0

        for syn in synapses:
            g_syn = syn.update(t, dt)
            if syn.is_inhibitory:
                syn.post_neuron.g_inh += g_syn
            else:
                syn.post_neuron.g_exc += g_syn

        for neuron in neurons:
            spiked = neuron.update(t, dt)
            if spiked:
                spike_data[neuron.neuron_id].append(t)
                if neuron.neuron_id in synapse_lookup:
                    for syn in synapse_lookup[neuron.neuron_id]:
                        syn.receive_spike(t)

        if record_voltage:
            if save_raw_voltage:
                step_voltages = np.fromiter(
                    (np.float32(neuron.v) for neuron in neurons),
                    dtype=np.float32,
                    count=len(neurons),
                )
                if voltage_recorder is None:
                    voltage_traces[:, step] = step_voltages
                else:
                    voltage_recorder.write_step(step_voltages)
            elif step % voltage_sample_steps == 0 and voltage_sample_idx < n_voltage_samples:
                sampled_voltages = np.fromiter(
                    (np.float32(neuron.v) for neuron in neurons),
                    dtype=np.float32,
                    count=len(neurons),
                )
                voltage_traces[:, voltage_sample_idx] = sampled_voltages
                voltage_sample_idx += 1

    print("  Progress: 100%")
    print("Simulation complete!\n")

    if record_voltage:
        voltage_data = {
            "times": voltage_times,
            "sample_rate": saved_voltage_dt,
            "save_raw_voltage": bool(save_raw_voltage),
            "simulation_dt": float(dt),
        }
        if save_raw_voltage and voltage_recorder is not None:
            voltage_data.update(voltage_recorder.finalize())
        else:
            voltage_data["traces"] = voltage_traces

    return spike_data, voltage_data