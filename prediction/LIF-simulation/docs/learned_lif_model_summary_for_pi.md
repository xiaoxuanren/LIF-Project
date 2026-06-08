# Learned LIF Connectivity Model: PI Summary

## Short Version

This model takes recorded spike trains from the simulated network and asks a simple question: for each neuron, which nearby neurons best explain when it fires? Instead of using a generic classifier, it fits a small differentiable leaky integrate-and-fire model to each postsynaptic neuron. The model learns which candidate presynaptic neurons matter, how strong their influence is, and what synaptic delay best matches the observed spike timing. After training, the learned weights are read out as the estimated connectivity map.

In practical terms, this is not mainly a free-running network simulator. It is an inference model that uses LIF-like dynamics as an interpretable bridge between spike timing and connectivity. The goal is to recover who is connected to whom, not just to predict spikes with a black-box model.

## Slightly Fuller Explanation

The input to the model is a session of spike recordings, optionally across multiple recordings from the same network, plus neuron positions and the list of candidate neurons to consider for each target neuron. Spike times are converted into a binned binary spike matrix. For each postsynaptic neuron, the script first proposes a manageable candidate set of possible inputs using a hybrid rule: mostly spatially nearby neurons, plus a smaller set of neurons whose spikes consistently occur just before the target neuron spikes. That step keeps the problem computationally feasible while still allowing nonlocal but temporally plausible connections to enter the model.

The core model is deliberately simple. For one postsynaptic neuron at a time, it computes a weighted sum of delayed presynaptic spikes, updates a soft leaky integrate-and-fire membrane variable, and produces a spike probability. The trainable quantities are the candidate synaptic weights, a discrete delay distribution for each candidate input, and a few membrane parameters shared across neurons. If a candidate input repeatedly helps explain the timing of the target neuron's spikes, its learned weight grows; if not, sparsity penalties push it toward zero. The absolute value of the final learned weight is then used as the connectivity score.

One important design choice is that the model is trained on short event windows around real spikes rather than on entire recordings end to end. The recordings are extremely sparse, so full-sequence training tends to reward the trivial solution of predicting silence. Event-window training forces the optimizer to spend most of its time on informative moments: real spikes and matched negative windows away from spikes. Validation is also done on held-out data from the same neurons, not held-out neurons, because each postsynaptic neuron has its own learned weight row. Holding out whole neurons would test parameters that were never trained and would give a misleading sense of performance.

The main reason for this design is interpretability. A purely black-box temporal model can rank edges, but it is harder to explain why a connection was selected. Here, the learned parameters still have a mechanistic meaning: input strength, synaptic delay, leak, threshold, and reset. That makes the output easier to compare with the known simulated ground truth and easier to defend scientifically. The tradeoff is that the model is still an approximation. It only searches within a candidate set, it uses simplified LIF dynamics rather than full biophysics, and its scores are best interpreted as evidence for directed influence within that candidate set.

## What The Model Estimates

The main thing the model estimates is a connectivity score for each candidate presynaptic to postsynaptic pair. In the implementation, that score comes from the learned synaptic weight. For each postsynaptic neuron j, the model learns one weight per candidate input i, written as W_{j,i}. Large-magnitude weights mean that candidate input helps explain the target neuron's spike timing; weights near zero mean that candidate does not contribute much.

It also estimates a delay profile for each candidate edge. Instead of forcing one fixed synaptic delay, the model learns a discrete distribution over a small set of possible delays. This allows it to represent uncertainty or a mixture of likely delays while staying differentiable.

In addition, the model learns a small set of membrane parameters shared across all neurons in the fit: a leak factor, a threshold, a spike nonlinearity slope, and a reset strength. So the model is estimating both edge-specific quantities and a compact set of global dynamical parameters.

What it is not estimating is a full biophysical neuron model. It does not learn conductances, ion-channel parameters, or detailed synaptic kinetics. It is a simplified dynamical inference model whose main purpose is to recover effective directed connectivity from spike timing.

## Equation Used

For a postsynaptic neuron j, the model first forms a delayed version of each candidate presynaptic spike train. If x_i(t) is the spike train of candidate neuron i, then the delayed effective input is:

$$
	ilde{x}_{j,i}(t) = \sum_{d=0}^{D-1} p_{j,i}(d)\,x_i(t-d)
$$

where p_{j,i}(d) is the learned probability of delay d for that candidate edge.

Those delayed inputs are then combined into a synaptic drive:

$$
I_j(t) = \sum_i W_{j,i}\,\tilde{x}_{j,i}(t)
$$

The membrane update is a soft discrete-time LIF equation:

$$
V_j(t) = \alpha V_j(t-1) + I_j(t)
$$

The model converts membrane voltage into a spike probability using a sigmoid rather than a hard threshold:

$$
\hat{s}_j(t) = \sigma\left(\beta\,(V_j(t) - \theta)\right)
$$

and then applies a soft reset:

$$
V_j(t) \leftarrow V_j(t) - r\,\hat{s}_j(t)
$$

where \alpha is the leak factor, \theta is the threshold, \beta controls how sharp the spike nonlinearity is, and r is the reset strength.

Training minimizes a spike-prediction loss plus a sparsity penalty on the weights:

$$
\mathcal{L} = \mathrm{BCE}(\hat{s}, s) + \lambda \lVert W \rVert_1
$$

So the model is fit to explain observed postsynaptic spikes, while the L1 term pushes unnecessary candidate edges toward zero.

## How Training And Validation Data Are Constructed

This is an important point because the model is not trained directly on connectivity labels. Instead, it is trained to predict postsynaptic spikes from candidate presynaptic spikes.

The raw input is one recording, or several recordings from the same simulated network, converted into a binary spike matrix with neurons along one axis and time bins along the other. From that matrix, the script builds short event windows rather than training on the entire recording in one pass.

There are two kinds of windows. Positive windows are centered on real postsynaptic spikes. Negative windows are centered on times that are sufficiently far from any postsynaptic spike. Each window contains a warmup segment, prespike context, and postspike context. The warmup segment lets the membrane state settle before the loss is computed. This design avoids spending most of the optimization on long stretches of zeros.

For each sample, the model sees the candidate presynaptic spike trains in that window, the true postsynaptic spike train in the same window, and the identity of the postsynaptic neuron whose private weight row should be used.

Validation is built from the same neurons, not different neurons. If multiple recordings are available, the script holds out one or more entire recordings for validation. If there is only a single recording, it splits each neuron's event windows into training and validation subsets. This is necessary because each postsynaptic neuron has its own learned weight row. Holding out whole neurons would leave those parameters untrained and would not be a meaningful test.

Within a single run, the script therefore has a training set and a validation set, but not a fully separate independent test set in the usual machine-learning sense. The known ground-truth connectivity from the simulation is used afterward to score the learned weights with metrics such as AUC, average precision, and F1. A stricter test is to train on one session and then check how well the same approach generalizes on another session or another simulated network.

## One-Sentence Takeaway

This model uses an interpretable, differentiable LIF neuron as an inference engine: it takes spike trains in, learns which candidate inputs best explain each neuron's firing, and turns the learned synaptic weights into a connectivity estimate.