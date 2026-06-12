"""Short-term synaptic depression for the LIF-Project conductance synapse.

`DepressingExpSynapse` subclasses the project's `ExpSynapse` and adds a depleting
resource pool, so recurrent bursts self-terminate (resources run down) and the
inter-burst interval is set by resource recovery -- reproducing the burst-then-
quiet dynamics of real dissociated cultures. Without depression the project's
`ExpSynapse` bursts do not self-limit, which is why culture-realistic regimes
(e.g. the burstexcl regime) cannot be generated with the base synapse.

Model (resource-depletion / simplified Tsodyks-Markram, depression only):
    Each presynaptic spike releases a fraction proportional to the resource R
    currently available (R in [0, 1]); the delivered conductance increment is
    `g_increment * R`. R then drops by `delta_q` and recovers toward 1 with time
    constant `tau_q`:

        on presynaptic spike:   release = R;   R <- R * (1 - delta_q)
        every step:             R <- R + (1 - R) * dt / tau_q

A rested synapse (R = 1) delivers exactly the same increment as the base
`ExpSynapse`, so behavior is unchanged in the low-rate limit and only diverges
under sustained/bursty drive.

Equivalence note: because every synapse from a given presynaptic neuron is driven
by that neuron's spike train, synapses sharing the same (tau_q, delta_q) follow
identical R(t). Per-synapse depression with uniform parameters is therefore
identical to per-neuron depression -- so the parameters tuned in the vectorized
preview (tau_q=4000 ms, delta_q=0.8 for the burstexcl regime) carry over directly.

Drop-in: `DepressingExpSynapse` keeps the exact `update(t, dt) -> g_syn` and
`receive_spike(t)` interface, so `simulate_network` uses it with no changes.
"""

import numpy as np

from .models import ExpSynapse


class DepressingExpSynapse(ExpSynapse):
    """Conductance synapse with activity-dependent short-term depression.

    Args:
        pre_neuron_id: Integer id of the presynaptic neuron.
        post_neuron: ``LIFNeuron`` receiving the conductance change.
        weight: Signed synaptic weight (as in ``ExpSynapse``).
        is_inhibitory: Whether the synapse uses inhibitory kinetics.
        delay: Transmission delay in milliseconds.
        tau_q: Resource-recovery time constant in milliseconds. Larger values
            lengthen the inter-burst interval (slower-bursting cultures).
        delta_q: Fraction of available resource consumed per presynaptic spike,
            in [0, 1). Larger values terminate bursts harder and faster.

    Returns:
        An initialized ``DepressingExpSynapse`` whose released conductance scales
        with the momentary resource level.
    """

    def __init__(self, pre_neuron_id, post_neuron, weight, is_inhibitory=False,
                 delay=1.0, tau_q=4000.0, delta_q=0.8):
        super().__init__(pre_neuron_id, post_neuron, weight, is_inhibitory, delay)
        if not (0.0 <= delta_q < 1.0):
            raise ValueError("delta_q must be in [0, 1).")
        if tau_q <= 0.0:
            raise ValueError("tau_q must be > 0.")
        self.tau_q = float(tau_q)
        self.delta_q = float(delta_q)
        self.R = 1.0
        # pending_spikes now holds (arrival_time, release_fraction) tuples.

    def receive_spike(self, t):
        """Queue a presynaptic spike, capturing the resource released now.

        Args:
            t: Spike time in milliseconds.

        Returns:
            None. Records ``(t + delay, release)`` and depletes the resource pool.
        """
        release = self.R                      # resource available at spike time
        self.R *= (1.0 - self.delta_q)        # deplete
        self.pending_spikes.append((t + self.delay, release))

    def update(self, t, dt):
        """Recover resources, deliver due (depressed) spikes, decay conductance.

        Args:
            t: Current simulation time in milliseconds.
            dt: Simulation step in milliseconds.

        Returns:
            The updated synaptic conductance after delivery and exponential decay.
        """
        # resources recover toward 1
        self.R += (1.0 - self.R) * dt / self.tau_q
        # deliver any due spikes, each scaled by the resource captured at spike time
        while self.pending_spikes and self.pending_spikes[0][0] <= t:
            _, release = self.pending_spikes.pop(0)
            self.g_syn += self.g_increment * release
        self.g_syn *= np.exp(-dt / self.tau)
        return self.g_syn

    def reset_state(self):
        """Reset conductance, resource pool, and pending queue between recordings.

        Returns:
            None. Restores the synapse to its rested state (R = 1).
        """
        self.g_syn = 0.0
        self.R = 1.0
        self.pending_spikes = []
