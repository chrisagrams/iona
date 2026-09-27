# Methods

How the training machinery works, in plain terms, with credit to whoever invented it. For
WHY each choice was made (and what was rejected), see PLAN.md → Design decisions.

## GradCache: exact contrastive gradients for batches that don't fit in memory

**Not our method.** Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan, *Scaling Deep
Contrastive Learning Batch Size under Memory Limited Setup*, RepL4NLP @ ACL 2021
(arXiv:2101.06983; reference code: github.com/luyug/GradCache). Our implementation is
`gradcache_step` in `msdelta/finetuning/contrastive/contrastive.py` (a reimplementation, not their library). Our
additions are length-trimmed chunks and the decision to turn gradient checkpointing off.

**The problem.** Gradient accumulation (run a few samples, backprop, repeat, sum) only
works when each sample's loss is independent. SupCon is not: each anchor's loss is
normalised over EVERY other spectrum in the batch, so chunks of 4 would each see only 3
negatives, which is a different and much weaker loss. A full 255-spectrum forward with
gradients doesn't fit either, because the DeltaMZBias attention bias costs memory
∝ peaks² per spectrum, and a tile runs out at about 16 spectra of 512 peaks.

**The trick.** The loss depends on the weights θ only through the embeddings e_i = f_θ(x_i),
so dL/dθ = Σ_i (∂L/∂e_i)·(∂e_i/∂θ). The first factor needs the whole batch but is cheap,
because it only involves the 255 × 1280 embedding matrix. The second factor is per
spectrum, so it can be computed chunk by chunk.

1. **Embed everything without gradients.** Chunks of 4 run under `no_grad`, and only the
   embeddings are kept. The RNG state is saved before each chunk.
2. **Loss on the embeddings alone.** SupCon runs over the full 255 × 255 similarity matrix,
   and backprop goes only as far as the embeddings, giving the cached gradients
   g_i = ∂L/∂e_i.
3. **Re-embed each chunk with gradients** (replaying its saved RNG state, so dropout matches
   pass 1) and backprop g_i through it. The weight gradients summed over chunks equal the
   full-batch gradient exactly; `tests/test_contrastive.py` checks this against a direct
   full-batch backward. The KL anchor is per spectrum, so it is added here, chunk by chunk.

**Cost.** Peak memory is one chunk's activations, whatever the batch size. Compute is two
forward passes per spectrum, about 1.3–1.5× a plain step.

**Our additions.**
- **Length trimming** (`--gradcache_trim_padding`). Spectra are sorted by peak count before
  chunking, and each chunk is padded only to its own longest spectrum (group labels are
  permuted along with them). This is exact because padding is masked.
- **No gradient checkpointing** (`--gradient_checkpointing false`). Checkpointing recomputes
  forwards to save memory, but GradCache already bounds memory, so it only added a third
  forward pass.
- Together, per step: 50m 20.5 → 5.1 s, 400m 42.9 → 13.2 s (`pbs/diag/gradcache_bench.pbs`).

**Consequence for batch design.** Everything about negatives (same-mass blocks, random
groups, per-anchor masks) happens in step 2 on the small embedding matrix. Steps 1 and 3
never see it, so no batch design is harder or easier for GradCache.

## Pairformer: a single + pair spectrum encoder (optional architecture)

**Not our architecture.** The Pairformer is the trunk of AlphaFold 3: Josh Abramson et al.,
*Accurate structure prediction of biomolecular interactions with AlphaFold 3*, Nature 630
(2024). Its triangle multiplicative updates and triangle attention come from AlphaFold 2:
John Jumper et al., *Highly accurate protein structure prediction with AlphaFold*, Nature
596 (2021). The adaptation to MS/MS spectra was written on the `exp_pairformer` branch
(a1275c0, "Added pairformer arch") and revised on `sweep/pairformer-aurora` (up to
bc2037b, including fd64faa, which stops the pair stream leaking the masked intensity). Our
port is `msdelta/models/pairformer.py`, selected with `MSDeltaConfig.architecture =
"pairformer"`; the default stays `"transformer"`.

**What it is.** The transformer keeps one vector per peak and adds a fixed, learned
function of Δm/z to the attention logits. The Pairformer keeps two states: s, one vector
per peak, and z, one vector per peak PAIR. Both are updated in every layer. z gives the
attention bias for s, so the bias can depend on the rest of the spectrum and change with
depth, not only on the m/z difference of the two peaks.

**Taken from AlphaFold 3 (single-sequence form, without MSA or diffusion modules):**
- the block order of AF3 Algorithm 17: refine z, read a per-head bias from z, then update s
  with gated self-attention that uses that bias (Alg. 24 without diffusion conditioning),
  then a SwiGLU transition (Alg. 11);
- triangle multiplication, outgoing and incoming (Alg. 12/13). This is the main idea: z_ij
  is updated from z_ik and z_jk, summed over every third peak k. The cost is cubic in the
  peak count;
- triangle attention around the starting and ending node (Alg. 14/15). It is optional and
  off by default, because it is the most memory-intensive part;
- the outer-product-mean from s into z (AF3 Alg. 9 for one sequence). Its output
  projection is zero at initialisation, so it starts as a no-op.

**Adapted for spectra by the source branch (ported as written):**
- z is initialised as W_a s_i + W_b s_j + W_c f_ij. W_a and W_b are different, so z has a
  direction. f_ij starts with the same signed-Δm/z Fourier features that the transformer's
  `DeltaMZBias` uses. Chemistry priors follow, and each can be turned off: Fourier features
  of the mass defect of |Δ|; a Gaussian soft match of |Δ| to a list of neutral losses,
  residues and sugars (σ in ppm of the heavier peak); ¹³C isotope spacing (k = 1, 2); and
  the relative log intensity;
- a masked peak enters the relative-intensity feature through a learned stand-in, not
  through its true value. Without this, the pretraining label leaks through z (the source
  measured the leak at skill 0.985);
- the ablation ladder `pair_update = static | transition | triangle`, and a write-back
  switch.

**Changed or left out in our port.** The source's precursor features (the fragment
complementarity term, and precursor m/z plus charge fed in through adaptive LayerNorm)
are left out. This keeps the encoder on this repo's `(mz, log_intensity, attention_mask)`
interface. The source's `pairformer_intrinsic` variant also dropped them, because in that
data the precursor was derived from the identification label. The per-peak token is the
transformer's intensity embedding plus a projection of Fourier(m/z). The source used
Fourier features of intensity here. Fourier frequencies are fixed, as everywhere in this
package. The module tree has the same parts as the transformer's (`embed`, `bias_module`,
`blocks`, `norm`). `blocks[i]` returns the single state, so the layer-mix hooks, the
bias-curve panels and every task head work without changes. `bias_module.evaluate` shows
only the pure-Δm/z part of the learned bias.

**Cost.** z takes B·N²·c_z memory per layer. Triangle multiplication takes O(B·N³·c) compute,
and on the source branch it used 87% of step time. The peaks cap is therefore the strongest
cost setting. `tests/test_pairformer.py` checks the mask invariance, permutation
equivariance and the absence of a label leak.
