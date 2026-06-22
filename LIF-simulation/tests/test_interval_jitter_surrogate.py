"""Tests for the interval-jitter connectivity surrogate null.

Covers build_interval_jitter_surrogates (lif_inference/shared_data.py):
  (a) per-neuron spike counts are preserved exactly,
  (b) the output stays binary,
  (c) no spike is moved across a recording boundary,
  (d) on an injected A->B (3 ms lag) edge plus a shared-burst common-input pair,
      the 3 ms coincidence drops to ~chance (monosynaptic timing destroyed) while
      the 50 ms-smoothed co-activation correlation stays high (common input kept) --
      unlike circular shift, which destroys the co-activation too.

Runnable two ways:
  * pytest tests/test_interval_jitter_surrogate.py
  * python tests/test_interval_jitter_surrogate.py

shared_data.py is numpy-only, so it is loaded directly by path to avoid importing
the lif_inference package (which pulls in torch).
"""

import importlib.util
import os
import sys

import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_TESTS_DIR)  # .../LIF-Project/LIF-simulation
_SHARED_PATH = os.path.join(_PKG_PARENT, "lif_inference", "shared_data.py")

_spec = importlib.util.spec_from_file_location("lif_shared_data_under_test", _SHARED_PATH)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

build_interval_jitter_surrogates = _shared.build_interval_jitter_surrogates
build_segmentwise_circular_shift_surrogates = _shared.build_segmentwise_circular_shift_surrogates

JITTER_BINS = 25


def _random_binary(rng, n_neurons, T, rate=0.02):
    return (rng.random((n_neurons, T)) < rate).astype(np.float32)


def test_preserves_per_neuron_counts():
    rng = np.random.default_rng(0)
    sm = _random_binary(rng, n_neurons=12, T=4000, rate=0.03)
    out, = build_interval_jitter_surrogates([sm], rng=np.random.default_rng(1),
                                            jitter_bins=JITTER_BINS)
    assert out.shape == sm.shape
    assert np.array_equal(out.sum(axis=1), sm.sum(axis=1))  # exact per-neuron counts
    assert out.dtype == sm.dtype


def test_output_is_binary():
    rng = np.random.default_rng(2)
    sm = _random_binary(rng, n_neurons=8, T=3000, rate=0.05)
    out, = build_interval_jitter_surrogates([sm], rng=np.random.default_rng(3),
                                            jitter_bins=JITTER_BINS)
    assert np.all((out == 0) | (out == 1))


def test_no_spike_crosses_recording_boundary():
    rng = np.random.default_rng(4)
    T1, T2 = 2000, 3000
    T = T1 + T2
    sm = _random_binary(rng, n_neurons=10, T=T, rate=0.04)
    # Spikes hugging both sides of the boundary must not leak across it.
    sm[:, T1 - 1] = 1.0
    sm[:, T1] = 1.0
    boundaries = [0, T1, T]
    out, = build_interval_jitter_surrogates([sm], boundaries=boundaries,
                                            rng=np.random.default_rng(5),
                                            jitter_bins=JITTER_BINS)
    # Per-segment, per-neuron counts preserved => nothing moved across the boundary.
    assert np.array_equal(out[:, :T1].sum(axis=1), sm[:, :T1].sum(axis=1))
    assert np.array_equal(out[:, T1:].sum(axis=1), sm[:, T1:].sum(axis=1))


def _lag_coincidence(a_row, b_row, lag):
    """Count of (A at t, B at t+lag) coincidences."""
    return float(np.sum(a_row[:-lag] * b_row[lag:]))


def _smoothed_corr(c_row, d_row, win=50):
    kernel = np.ones(win, dtype=np.float64)
    cs = np.convolve(c_row.astype(np.float64), kernel, mode="same")
    ds = np.convolve(d_row.astype(np.float64), kernel, mode="same")
    if cs.std() == 0 or ds.std() == 0:
        return 0.0
    return float(np.corrcoef(cs, ds)[0, 1])


def _make_edge_and_common_input(rng):
    """Neurons: 0=A, 1=B (A->B at 3-bin lag); 2=C, 3=D (shared 100-bin bursts)."""
    T = 6000
    lag = 3                       # 3 ms at dt=1 ms; well inside the 25-bin jitter window
    sm = np.zeros((6, T), dtype=np.float32)

    a_times = rng.choice(np.arange(100, T - 100), size=200, replace=False)
    sm[0, a_times] = 1.0
    sm[1, a_times + lag] = 1.0    # monosynaptic: B fires 3 bins after each A spike

    # Common input: C and D both fire inside the same 100-bin bursts (coarse,
    # > jitter window) -> co-activation should survive interval jitter.
    for b0 in range(150, T - 200, 300):
        for nid in (2, 3):
            offs = rng.choice(np.arange(100), size=25, replace=False)
            sm[nid, b0 + offs] = 1.0

    for nid in (4, 5):           # uncorrelated filler
        sm[nid, rng.choice(np.arange(T), size=150, replace=False)] = 1.0
    return sm, lag


def test_destroys_fine_timing_keeps_common_input():
    sm, lag = _make_edge_and_common_input(np.random.default_rng(7))

    jit, = build_interval_jitter_surrogates([sm], rng=np.random.default_rng(8),
                                            jitter_bins=JITTER_BINS)
    circ, = build_segmentwise_circular_shift_surrogates([sm], rng=np.random.default_rng(8))

    # (d.1) monosynaptic 3-bin coincidence collapses to ~chance under jitter.
    real_coin = _lag_coincidence(sm[0], sm[1], lag)
    jit_coin = _lag_coincidence(jit[0], jit[1], lag)
    chance = sm[0].sum() * sm[1].sum() / sm.shape[1]   # expected random overlaps
    assert real_coin > 150
    assert jit_coin < real_coin * 0.2
    assert jit_coin < chance + 25

    # (d.2) coarse co-activation of the common-input pair is retained under jitter
    # but destroyed by circular shift.
    real_corr = _smoothed_corr(sm[2], sm[3])
    jit_corr = _smoothed_corr(jit[2], jit[3])
    circ_corr = _smoothed_corr(circ[2], circ[3])
    assert real_corr > 0.5
    assert jit_corr > 0.4
    assert circ_corr < 0.3
    assert jit_corr > circ_corr + 0.3


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
