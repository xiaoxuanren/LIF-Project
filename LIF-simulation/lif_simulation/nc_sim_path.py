"""Path shim so ``import nc_sim`` resolves the bundled git submodule.

nc_sim (github.com/akkeh/nc_sim, GPL-3) is included as a git submodule at
``<repo>/external/nc_sim`` instead of being installed or copied into this
package, keeping the GPL-3 boundary clean. ``ensure_nc_sim_importable``
prepends ``<repo>/external`` to ``sys.path`` so the submodule's top-level
``nc_sim`` package becomes importable, without touching the adapter or vendoring
any GPL-3 source.

Callers that need nc_sim (the workflow/probe dispatch and the adapter test) call
``ensure_nc_sim_importable()`` once before ``import nc_sim``. As an alternative,
add ``<repo>/external`` to ``PYTHONPATH`` and the shim becomes a no-op.
"""

import os
import sys

# This file lives at <repo>/LIF-simulation/lif_simulation/nc_sim_path.py, so the
# repository root is three directories up and the submodule sits at
# <repo>/external/nc_sim.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXTERNAL_DIR = os.path.join(_REPO_ROOT, "external")


def ensure_nc_sim_importable():
    """Make ``import nc_sim`` resolve the bundled submodule at ``<repo>/external/nc_sim``.

    Idempotent: prepends ``<repo>/external`` to ``sys.path`` only when it is not
    already present (e.g. via a ``PYTHONPATH`` entry).

    Returns:
        The ``<repo>/external`` directory that was placed on ``sys.path``.
    """
    if _EXTERNAL_DIR not in sys.path:
        sys.path.insert(0, _EXTERNAL_DIR)
    return _EXTERNAL_DIR
