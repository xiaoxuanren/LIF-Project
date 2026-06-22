import numpy as np


class LIFNeuron:
    """Conductance-based LIF neuron with adaptation and optional slow h-current.

    The model combines excitatory and inhibitory synaptic conductances, a simple
    spike-triggered adaptation current, and an optional hyperpolarization-
    activated inward current used for sag and rebound experiments.

    Args:
        neuron_id: Integer id used throughout the network and saved outputs.
        is_inhibitory: Whether the neuron should use the inhibitory parameter set.
        use_h_current: Whether the slow h-current should be enabled at startup.

    Returns:
        An initialized ``LIFNeuron`` instance ready to participate in the network
        simulation.
    """

    def __init__(self, neuron_id, is_inhibitory=False, use_h_current=True, noise_sigma=0.0):
        """Initialize one neuron and load the parameter set for its cell type.

        Args:
            neuron_id: Integer id used throughout the network and saved outputs.
            is_inhibitory: Whether this neuron should use the inhibitory parameter
                set instead of the excitatory one.
            use_h_current: Whether the slow h-current should be active after
                initialization.
            noise_sigma: Standard deviation of the additive membrane-noise term.

        Returns:
            None. The constructor initializes the neuron's static parameters and
            dynamic state in place.
        """
        self.neuron_id = neuron_id
        self.is_inhibitory = is_inhibitory
        self.use_h_current = bool(use_h_current)

        if is_inhibitory:
            self.tau_m = 10.0
            self.v_thresh = -50.0
            self.adaptation_increment = 0.003
            self.tau_adaptation = 40.0
            self.g_h_max = 0.0011
            self.tau_h = 100.0
        else:
            self.tau_m = 20.0
            self.v_thresh = -50.0
            self.adaptation_increment = 0.025
            self.tau_adaptation = 300.0
            self.g_h_max = 0.0023
            self.tau_h = 140.0

        self.v_rest = -61.5
        self.v_reset = -70.0
        self.v_floor = -80.0
        self.R_m = 100.0
        self.tau_ref = 2.0
        self.noise_sigma = float(noise_sigma)

        self.e_h = -35.0
        self.v_half_h = -75.0
        self.k_h = 5.5

        self.v = self.v_rest + np.random.uniform(-2, 2)
        self.t_last_spike = -np.inf
        self.spike_times = []

        self.g_exc = 0.0
        self.g_inh = 0.0
        self.e_exc = 0.0
        self.e_inh = -75.0
        self.i_ext = 0.0
        self.i_baseline = 0.0
        self.i_adapt = 0.0

        self.g_h_max_base = self.g_h_max
        self.adaptation_increment_base = self.adaptation_increment
        self.h_gate = 0.0
        self.i_h = 0.0
        self.set_h_current_enabled(self.use_h_current)

    def _h_inf(self, voltage):
        """Compute the steady-state activation level of the h-current gate.

        Args:
            voltage: Membrane voltage in millivolts.

        Returns:
            The equilibrium h-gate value between 0 and 1 at the supplied voltage.
        """
        return 1.0 / (1.0 + np.exp((voltage - self.v_half_h) / self.k_h))

    def set_h_current_enabled(self, enabled):
        """Enable or disable h-current dynamics for this neuron.

        Args:
            enabled: Whether future updates should include the h-current term.

        Returns:
            None. The method updates ``g_h_max`` and resets the dependent h-current
            state variables so the neuron remains internally consistent.
        """
        self.use_h_current = bool(enabled)
        self.g_h_max = self.g_h_max_base if self.use_h_current else 0.0
        if self.use_h_current:
            self.h_gate = self._h_inf(self.v)
            self.i_h = self.g_h_max * self.h_gate * (self.e_h - self.v)
        else:
            self.h_gate = 0.0
            self.i_h = 0.0

    def set_adaptation_scale(self, scale):
        """Scale the spike-triggered adaptation (K+ AHP) current from its baseline.

        Mirrors the I_h ``g_h_max_base`` / ``set_h_current_enabled`` pattern so the
        adaptation current can be perturbed idempotently from a stored base
        (``scale=0`` blocks it, ``1`` = baseline), regardless of prior scaling.

        Args:
            scale: Non-negative multiplier applied to the baseline per-spike
                adaptation increment.

        Returns:
            None. Updates ``adaptation_increment`` in place.
        """
        if scale < 0.0:
            raise ValueError("adaptation scale must be >= 0")
        self.adaptation_increment = self.adaptation_increment_base * float(scale)

    def set_adaptation_enabled(self, enabled):
        """Enable or block the spike-triggered adaptation current.

        Args:
            enabled: When ``False`` the per-spike adaptation increment is set to
                zero (blocked); when ``True`` the baseline increment is restored.

        Returns:
            None.
        """
        self.set_adaptation_scale(1.0 if enabled else 0.0)

    def update(self, t, dt):
        """Advance the neuron state by one integration step.

        Args:
            t: Current simulation time in milliseconds.
            dt: Simulation time step in milliseconds.

        Returns:
            ``True`` if the neuron crossed threshold and emitted a spike during
            this step, otherwise ``False``.
        """
        spiked = False

        self.i_adapt *= np.exp(-dt / self.tau_adaptation)

        if self.use_h_current:
            h_inf = self._h_inf(self.v)
            self.h_gate += (h_inf - self.h_gate) * dt / self.tau_h
            self.h_gate = np.clip(self.h_gate, 0.0, 1.0)
            self.i_h = self.g_h_max * self.h_gate * (self.e_h - self.v)
        else:
            self.h_gate = 0.0
            self.i_h = 0.0

        if t - self.t_last_spike < self.tau_ref:
            self.v = self.v_reset
            return False

        noise = 0.0
        if self.noise_sigma > 0.0:
            noise = np.random.normal(0.0, self.noise_sigma) * np.sqrt(dt)

        i_exc = self.g_exc * (self.e_exc - self.v)
        i_inh = self.g_inh * (self.e_inh - self.v)
        current_term = self.R_m * (i_exc + i_inh + self.i_h + self.i_ext + self.i_baseline - self.i_adapt)
        dv = ((self.v_rest - self.v) + current_term) * dt / self.tau_m + noise
        self.v += dv

        if self.v < self.v_floor:
            self.v = self.v_floor

        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.t_last_spike = t
            self.spike_times.append(t)
            self.i_adapt += self.adaptation_increment
            spiked = True

        return spiked

    def reset_state(self):
        """Reset all dynamic variables before a new independent recording.

        Args:
            None.

        Returns:
            None. Membrane voltage, conductances, adaptation, spike history, and
            h-current state are reinitialized in place.
        """
        self.v = self.v_rest + np.random.uniform(-2, 2)
        self.t_last_spike = -np.inf
        self.spike_times = []
        self.g_exc = 0.0
        self.g_inh = 0.0
        self.i_ext = 0.0
        self.i_adapt = 0.0
        if self.use_h_current:
            self.h_gate = self._h_inf(self.v)
            self.i_h = self.g_h_max * self.h_gate * (self.e_h - self.v)
        else:
            self.h_gate = 0.0
            self.i_h = 0.0


class ExpSynapse:
    """Exponential conductance synapse with a fixed delivery delay.

    Each presynaptic spike is queued, delivered after ``delay``, and converted
    into an increment of the postsynaptic conductance that then decays
    exponentially.

    Args:
        pre_neuron_id: Integer id of the presynaptic neuron.
        post_neuron: ``LIFNeuron`` instance receiving the conductance change.
        weight: Signed synaptic weight used to derive the conductance increment.
        is_inhibitory: Whether the synapse should use inhibitory kinetics.
        delay: Transmission delay in milliseconds.

    Returns:
        An initialized ``ExpSynapse`` instance that can queue and deliver spikes.
    """

    def __init__(self, pre_neuron_id, post_neuron, weight, is_inhibitory=False, delay=1.0):
        """Initialize one synapse connecting a presynaptic neuron to a target.

        Args:
            pre_neuron_id: Integer id of the presynaptic neuron.
            post_neuron: ``LIFNeuron`` instance receiving this synapse's conductance.
            weight: Signed legacy current-equivalent synaptic weight.
            is_inhibitory: Whether the synapse should use inhibitory kinetics.
            delay: Transmission delay in milliseconds.

        Returns:
            None. The constructor stores the connection definition and derives the
            conductance increment used at spike arrival.
        """
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron = post_neuron
        self.weight = weight
        self.is_inhibitory = is_inhibitory
        self.delay = delay

        if is_inhibitory:
            self.tau = 10.0
            self.reference_driving_force = 15.0
        else:
            self.tau = 3.0
            self.reference_driving_force = 60.0

        self.g_increment = abs(self.weight) / self.reference_driving_force
        self.g_syn = 0.0
        self.pending_spikes = []

    def receive_spike(self, t):
        """Queue a presynaptic spike so it can arrive after the configured delay.

        Args:
            t: Spike time in milliseconds.

        Returns:
            None. The delayed arrival time is appended to ``pending_spikes``.
        """
        self.pending_spikes.append(t + self.delay)

    def update(self, t, dt):
        """Update synaptic conductance and deliver any due presynaptic spikes.

        Args:
            t: Current simulation time in milliseconds.
            dt: Simulation time step in milliseconds.

        Returns:
            The updated synaptic conductance after spike delivery and exponential
            decay.
        """
        while self.pending_spikes and self.pending_spikes[0] <= t:
            self.pending_spikes.pop(0)
            self.g_syn += self.g_increment

        self.g_syn *= np.exp(-dt / self.tau)
        return self.g_syn


class NetworkWeightParameters:
    """Container for the default within-cluster and between-cluster weight ranges.

    These values preserve the project's legacy weight conventions while allowing
    the rest of the simulation code to sample conductance increments from one
    shared configuration object.

    Args:
        None.

    Returns:
        An initialized ``NetworkWeightParameters`` instance containing the default
        weight ranges and lognormal sampling settings.
    """

    def __init__(self):
        """Populate the default synaptic weight ranges used by network generation.

        Args:
            None.

        Returns:
            None. The constructor stores excitatory and inhibitory within-cluster
            and between-cluster ranges, plus lognormal sampling settings.
        """
        self.within_exc_range = (0.25, 0.45)
        self.between_exc_range = (0.20, 0.35)
        self.within_inh_range = (0.35, 0.55)
        self.between_inh_range = (0.28, 0.45)
        self.use_lognormal = True
        self.lognormal_sigma = 0.5