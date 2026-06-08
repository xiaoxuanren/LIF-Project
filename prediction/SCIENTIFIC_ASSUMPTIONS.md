# Scientific Assumptions

This note is for stable project assumptions that the agent should preserve unless a task explicitly changes them.

## Model Level

- The project is a biologically informed LIF simulation, not a full ion-channel model.
- The active target is a conductance-based recurrent LIF network.
- The h-current is part of the intended model family, but it should be possible to disable it for controlled comparisons.
- If h-current is disabled, the implementation should make that mechanism explicit and easy to describe.
- Spike-frequency adaptation remains part of the core neuron behavior unless an experiment explicitly removes it.

## Network Meaning

- The network is clustered and spatially organized.
- Within-cluster connectivity is expected to exceed between-cluster connectivity by default.
- Connectivity is distance-limited.
- Hub neurons are a structural feature, not just a label. They preferentially support stronger or more likely long-range inter-cluster connectivity.

## Interpretation Boundaries

- Conductance-based synapses increase biophysical plausibility, but the model is still a reduced LIF abstraction.
- Rendered spike waveforms in saved voltage traces are visualization choices unless the code explicitly changes the underlying membrane dynamics.
- Inferred connectivity must stay conceptually separate from ground-truth simulated connectivity.

## Documentation Policy

- If simulation defaults, toggles, outputs, or workflow order change, update `SCRIPTS_SUMMARY.md` when it exists in the repo.
- If the public project story changes materially, update `README.md` too.
- Summary documents should describe the implementation that exists, not the implementation that is merely intended.