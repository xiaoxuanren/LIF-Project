"""Tests for short-term synaptic depression (DepressingExpSynapse + wiring).

Covers:
  (a) per-spike resource release depletes R and R recovers toward 1 with tau_q,
  (b) a rested DepressingExpSynapse delivers the same first-spike conductance
      trajectory as the base ExpSynapse,
  (c) create_clustered_network(depressing=True) builds DepressingExpSynapse at
      both the base and hub construction sites, while depressing=False builds the
      plain ExpSynapse (and leaves structure unchanged for a fixed seed).

Runnable two ways:
  * pytest tests/test_depressing_synapse.py
  * python tests/test_depressing_synapse.py

Imports only from lif_simulation.* (never lif_inference, which pulls in torch).
"""

import os
import sys

import numpy as np

# --- make the in-repo package importable --------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_TESTS_DIR)  # .../LIF-Project/LIF-simulation
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from lif_simulation.depressing_synapse import DepressingExpSynapse
from lif_simulation.models import ExpSynapse, LIFNeuron
from lif_simulation.network import create_clustered_network


def test_release_depletes_and_recovers():
    post = LIFNeuron(1)
    syn = DepressingExpSynapse(0, post, weight=1.0, is_inhibitory=False,
                               delay=1.0, tau_q=4000.0, delta_q=0.8)
    assert syn.R == 1.0

    # First (rested) spike: release captured = 1.0, then R drops by (1 - delta_q).
    syn.receive_spike(0.0)
    arrival, release = syn.pending_spikes[-1]
    assert arrival == 1.0           # t + delay
    assert release == 1.0           # rested release
    assert np.isclose(syn.R, 0.2)   # 1.0 * (1 - 0.8)

    # Second spike depletes further; its release equals R available at spike time.
    syn.receive_spike(0.0)
    assert np.isclose(syn.pending_spikes[-1][1], 0.2)
    assert np.isclose(syn.R, 0.04)  # 0.2 * (1 - 0.8)

    # Recovery toward 1 with tau_q, matching the analytic curve and monotonic.
    rec = DepressingExpSynapse(0, post, weight=1.0, tau_q=4000.0, delta_q=0.8)
    rec.receive_spike(0.0)
    r0 = rec.R
    dt = 1.0
    total_ms = 4000.0  # one recovery time constant
    prev = rec.R
    for step in range(int(total_ms / dt)):
        rec.update((step + 1) * dt, dt)
        assert rec.R >= prev - 1e-12  # monotonically recovering
        prev = rec.R

    analytic = 1.0 - (1.0 - r0) * np.exp(-total_ms / 4000.0)
    assert np.isclose(rec.R, analytic, atol=3e-3)
    assert r0 < rec.R < 1.0


def test_rested_depressing_matches_base_first_spike():
    base = ExpSynapse(0, LIFNeuron(1), weight=2.5, is_inhibitory=False, delay=1.0)
    dep = DepressingExpSynapse(0, LIFNeuron(2), weight=2.5, is_inhibitory=False,
                               delay=1.0, tau_q=4000.0, delta_q=0.8)

    # Same derived conductance increment (rested release = 1.0).
    assert np.isclose(base.g_increment, dep.g_increment)

    base.receive_spike(0.0)
    dep.receive_spike(0.0)

    dt = 0.1
    delivered = False
    for step in range(200):  # 20 ms: covers delivery at t = 1.0 ms and decay
        t = (step + 1) * dt
        g_base = base.update(t, dt)
        g_dep = dep.update(t, dt)
        assert np.isclose(g_base, g_dep, rtol=1e-9, atol=1e-12)
        if g_base > 0:
            delivered = True
    assert delivered  # the single spike was actually delivered


def _small_network(depressing, **overrides):
    np.random.seed(0)  # fixed seed: structure is identical with/without depression
    params = dict(
        num_clusters=3,
        neurons_per_cluster_range=(6, 6),
        inhibitory_probability=0.2,
        within_cluster_prob=0.9,
        between_cluster_prob=0.5,
        max_connection_distance=50.0,
        space_size=8,
        hub_fraction=0.4,
        hub_between_prob=0.9,
        depressing=depressing,
    )
    params.update(overrides)
    return create_clustered_network(**params)


def test_create_clustered_network_synapse_types():
    # depressing=False -> plain ExpSynapse only.
    _, syn_base, _, _, info_base = _small_network(depressing=False)
    assert len(syn_base) > 0
    assert all(isinstance(s, ExpSynapse) for s in syn_base)
    assert not any(isinstance(s, DepressingExpSynapse) for s in syn_base)

    # depressing=True -> DepressingExpSynapse everywhere, incl. hub edges.
    _, syn_dep, _, _, info_dep = _small_network(depressing=True, tau_q=3000.0, delta_q=0.5)
    assert len(syn_dep) > 0
    assert all(isinstance(s, DepressingExpSynapse) for s in syn_dep)
    assert info_dep["n_hub_connections"] > 0  # the hub construction site is exercised
    assert all(s.tau_q == 3000.0 and s.delta_q == 0.5 for s in syn_dep)

    # Same seed => the depression toggle does not change the wiring.
    assert len(syn_dep) == len(syn_base)
    assert info_dep["n_hub_connections"] == info_base["n_hub_connections"]


def _run_all():
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - simple standalone runner
            failures += 1
            print(f"FAIL  {name}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
