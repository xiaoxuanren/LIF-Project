# LIF Simulation Workspace Instructions

This workspace is intended to continue the `xiaoxuanren/LIF-simulation` project.

- Treat `LIF_network_simulation_network_burst_conductance_modular.ipynb` (plus the `lif_simulation/` package) as the current source of truth for simulation work unless the user explicitly asks to use a legacy notebook.
- Preserve the core project meaning: a clustered recurrent conductance-based LIF network with excitatory, inhibitory, and hub neurons, legacy weight scaling, spike-frequency adaptation, and a slow h-current.
- Treat the current active target as explicit: keep the model conductance-based and support an h-current feature that can be activated or deactivated as needed for ablation or validation.
- Keep the current clustered-network semantics in mind when reasoning about edits: default within-cluster connection probability is higher than between-cluster probability, hub neurons preferentially strengthen or increase long-range inter-cluster connectivity, and connectivity is distance-limited.
- Be explicit about the difference between reduced biophysical plausibility and full biophysical realism. This project is biologically informed LIF modeling, not a full ion-channel model.
- If h-current behavior is changed, preserve clarity about whether the implementation disables the current by parameter choice, by a config flag, or by bypassing the update path.
- When discussing or editing connectivity inference, keep ground-truth network structure, candidate-edge generation, learned scores, threshold selection, and evaluation metrics conceptually separate.
- If a change affects entry points, workflow order, parameter defaults, saved outputs, or script roles, update `SCRIPTS_SUMMARY.md` in the same task when that file exists.
- If a change materially alters project scope or the recommended workflow, also update `README.md` when that file exists.
- Prefer concise scientific reasoning, minimal code changes, and documentation that reflects the current implementation rather than aspirational behavior.