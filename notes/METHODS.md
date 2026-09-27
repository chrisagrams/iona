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
