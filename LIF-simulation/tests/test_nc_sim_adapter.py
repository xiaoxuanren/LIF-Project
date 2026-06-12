"""Tests for the nc_sim -> LIF-Project topology adapter.

These verify that ``build_network_from_nc_sim`` is a faithful drop-in for
``create_clustered_network``:

  * the connection table reproduces nc_sim ``grow_W``'s exact nonzero structure
    once rebuilt by ``build_ground_truth`` (no transpose ambiguity),
  * synaptic signs obey E/I identity and Dale's principle (every source column
    is single-signed; excitatory sources positive, inhibitory negative),
  * the returned network round-trips through ``save_network_structure``.

The test is runnable two ways:
  * ``pytest tests/test_nc_sim_adapter.py``
  * ``python tests/test_nc_sim_adapter.py`` (no pytest required)

``build_ground_truth`` is loaded directly from ``lif_inference/shared_data.py``
by file path because ``lif_inference/__init__.py`` imports torch, which is not
needed (and may be absent) for this numpy-only round-trip check.
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np

# --- make the in-repo packages importable -------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_TESTS_DIR)  # .../LIF-Project/LIF-simulation
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from lif_simulation.models import NetworkWeightParameters
from lif_simulation.nc_sim_adapter import build_network_from_nc_sim
from lif_simulation.nc_sim_path import ensure_nc_sim_importable
from lif_simulation.session_io import save_network_structure

ensure_nc_sim_importable()  # put <repo>/external on sys.path so `import nc_sim` works


def _load_build_ground_truth():
    """Load ``build_ground_truth`` from shared_data.py by path (avoids torch)."""
    shared = os.path.join(_PKG_PARENT, "lif_inference", "shared_data.py")
    spec = importlib.util.spec_from_file_location("lif_inference_shared_data", shared)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_ground_truth


build_ground_truth = _load_build_ground_truth()

# Small, fast, reproducible network. width = height = 1 and rho = 100 => ~100
# neurons; axon_length 0.8 mm is long enough to guarantee a non-empty topology.
_EI_RATIO = 0.8
_PARAMS = dict(
    width=1.0,
    height=1.0,
    rho=100.0,
    axon_length=0.8,
    ei_ratio=_EI_RATIO,
    num_clusters=4,
    seed=0,
    verbose=False,
)


def _build_with_captured_W():
    """Build via the adapter, capturing the exact ``grow_W`` matrix it consumed.

    Wrapping ``grow_W`` records the precise dense ``W[post, pre]`` the adapter
    used, so the structural assertion does not depend on reproducing nc_sim's
    RNG draws.
    """
    import nc_sim

    captured = {}
    original = nc_sim.axons.grow_W

    def _capturing_grow_W(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["W"] = np.array(result, copy=True)
        return result

    nc_sim.axons.grow_W = _capturing_grow_W
    try:
        network = build_network_from_nc_sim(**_PARAMS)
    finally:
        nc_sim.axons.grow_W = original
    return network, captured["W"]


def test_binary_structure_matches_grow_W():
    (neurons, synapses, connections, positions, info), W = _build_with_captured_W()
    n = len(neurons)

    assert W.shape == (n, n)
    n_nonzero = int(np.count_nonzero(W))
    assert n_nonzero > 0, "nc_sim produced an empty topology; raise rho/axon_length"
    # The adapter emits exactly one edge per nonzero of W.
    assert len(connections) == n_nonzero

    # build_ground_truth rebuilds B[post, pre]; grow_W already returns [post, pre].
    _, B = build_ground_truth(connections, n)
    assert np.array_equal(B != 0, W != 0)


def test_ei_signs_and_dale():
    (neurons, synapses, connections, positions, info), W = _build_with_captured_W()
    n = len(neurons)
    n_exc = int(n * _EI_RATIO)

    weights, _ = build_ground_truth(connections, n)  # signed weights, [post, pre]
    saw_exc = False
    saw_inh = False
    for pre in range(n):
        column = weights[:, pre]
        nonzero = column[column != 0]
        if nonzero.size == 0:
            continue
        # Dale: a presynaptic source projects with a single sign.
        single_signed = np.all(nonzero > 0) or np.all(nonzero < 0)
        assert single_signed, f"source {pre} mixes signs (Dale violated)"
        if pre < n_exc:
            assert np.all(nonzero > 0), f"excitatory source {pre} has non-positive weights"
            saw_exc = True
        else:
            assert np.all(nonzero < 0), f"inhibitory source {pre} has non-negative weights"
            saw_inh = True

    # The small network should exercise both populations.
    assert saw_exc, "no excitatory source columns were emitted"
    assert saw_inh, "no inhibitory source columns were emitted"


def test_save_network_structure_round_trips():
    neurons, synapses, connections, positions, info = build_network_from_nc_sim(**_PARAMS)
    weight_params = NetworkWeightParameters()

    with tempfile.TemporaryDirectory() as tmp:
        timestamp = "test_nc_sim"
        path = save_network_structure(connections, positions, info, weight_params, timestamp, tmp)
        assert os.path.exists(path)

        # Read inside a context so the lazy NpzFile handle is closed before the
        # TemporaryDirectory is torn down (Windows cannot delete an open file).
        with np.load(path, allow_pickle=True) as loaded:
            saved_connections = loaded["connections"]
            saved_positions = loaded["neuron_positions"].astype(float)

            assert saved_connections.shape == connections.shape
            for original_row, saved_row in zip(connections, saved_connections):
                assert int(original_row[0]) == int(saved_row[0])  # pre
                assert int(original_row[1]) == int(saved_row[1])  # post
                assert float(original_row[2]) == float(saved_row[2])  # weight
                assert str(original_row[3]) == str(saved_row[3])  # type

            assert np.allclose(saved_positions, positions.astype(float))


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
