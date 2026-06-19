# From the perceptron to the HeatMPGNN — lecture notes

Study notes for the heat-shield GNN surrogate (`HeatMPGNN`). Reads bottom-up:
every box in the model is built from the perceptron. Figure:
[model_onepager.png](../../heat_python/figs/model_onepager.png). Code:
[gnn_model.py](../../heat_python/gnn_model.py).

## The build-up (each step adds one idea)

1. **Perceptron** — one neuron: weighted sum of inputs + bias, then an activation.
   `z = w·x + b`, `a = f(z)`. Geometrically it draws *one straight boundary*. In
   PyTorch this is `nn.Linear`. It's the atom of every box in the figure.
2. **Layer → MLP / FCN** — a layer is many perceptrons in parallel (a matrix
   multiply `Wx + b`). Stack layers with activations between them = MLP. The
   activation is essential: without it, stacked linear layers collapse back to
   one. A wide-enough MLP can approximate any continuous function.
3. **Modern MLP block (`ResMLPBlock`)** — a plain 2-layer MLP plus three fixes
   that make deep stacks trainable: **GELU** (smooth activation), **LayerNorm**
   (rescale the vector to a sane range), **residual/skip connection** (`x + f(x)`,
   gives gradients a clean path and lets the block learn a correction). This is
   the gray box in the figure; same shape as a transformer feed-forward block.
4. **Features + why one big FCN fails** — a cell becomes a feature vector
   `[T, ρ, ρᵢ, porosity]` (z-score normalized). You can't flatten the whole mesh
   into one FCN: it locks to a fixed cell count, ignores locality, shares no
   weights, and explodes in size. Fix: apply one *small shared* network *per
   cell* using only local neighbors.
5. **The graph** — write the mesh as nodes (cells) + edges (neighbors). Edge
   features are **relative**: `[neighbor − self, dx]`, because conduction depends
   on the temperature *difference* and spacing, not absolute values. Two **ghost
   cells** at the ends carry the boundary forcing (they feed neighbors but aren't
   updated). Shared weights across the graph = runs on *any mesh length*.
6. **Message passing (the GNN step, K=1)** — per cell: encode self + each edge →
   build a **message** from each neighbor (shared message MLP) → combine
   (concatenate `[self, left, right]`, since 1D has exactly 2 ordered neighbors) →
   update MLP → decode. One round (K=1) = each cell hears only its direct
   neighbors, matching conduction's nearest-neighbor stencil. K=2 made it worse
   here (over-reach).
7. **Output = a delta of the free variables** — predicts the *change*
   `[dT, d_rho_i...]`, added on: `new = old + delta` (like the solver's own step,
   easier to learn). Only the independent variables are predicted; bulk density
   and porosity are *derived* from the species densities (avoids inconsistency).
8. **Rollout in time + training** — feed the output back as the next input
   (autoregressive recurrence, RNN-like; ghost forcing re-imposed each step).
   One-step training **drifts**: small errors compound. Fix: **multi-step
   training** — unroll M steps, loss over the whole rollout, backprop-through-time,
   so the model sees its own slightly-wrong outputs and learns to self-correct.
   This took the model from hundreds-of-K drift to ~7.6 K (paper §4.2).

## Key clarifications (from the Q&A)

- **Non-separable data (XOR).** A single line can't separate XOR; the hidden
  layer re-represents the data so the classes become linearly separable, then the
  last layer cuts. The XOR boundary is a degree-2 / diagonal-stripe shape, not a
  line. If data *truly* overlaps (same input, different output), nothing separates
  it perfectly — that's a data floor, not a model flaw.
- **Why many epochs?** Gradient descent only knows the *local* downhill direction
  and takes small steps on a curved loss surface, from a random start. No
  closed-form answer once layers are nonlinear. An epoch = one pass over the data
  (in noisy mini-batches). Many passes = enough small steps to reach the bottom.
- **What you minimize** = the **loss** (prediction error), not the boundary gap.
  The boundary settling into place is a side effect. MLPs minimize loss; they do
  not maximize margin (that's SVMs).
- **Hidden state** = a cell's internal scratch vector (`hidden_dim=64` numbers),
  not a physical quantity. It's the *output of a function* `f(input, weights)`,
  recomputed every run. Training fixes the **weights**; the hidden state is what
  those weights produce. Each cell has its own (shared weights, different inputs).
  Fully deterministic — computable by hand.
- **MLP-per-cell vs GNN** = message passing. A per-cell MLP sees only its own
  state (couldn't learn conduction). A GNN adds neighbor info via edges. A GNN is
  *built from* MLPs + that routing.
- **Residual connection = skip connection** (the additive kind: `x + f(x)`).
- **Transformer connection.** Multi-step training is RNN-lineage
  (backprop-through-time), not a transformer. But: the block shape ≈ transformer
  feed-forward; **message passing ≈ attention** (attention is message passing on a
  fully-connected graph); the rollout ≈ autoregressive token generation.
- **Tokens.** A token = a chunk of text (~¾ of a word, ~4 chars). "Token usage"
  counts text volume processed: input tokens (your message + full history +
  context + files) and output tokens (what's generated). 1,000 tokens ≈ 750 words.

## The model in one line

A weight-shared, per-cell message-passing GNN: encode each cell and its relative
edges, pass one round of neighbor messages (K=1, matching the conduction stencil),
predict a per-cell state-delta, and roll it forward in time with multi-step
training for stability.
