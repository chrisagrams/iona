"""K189-P: length-grouped GLOBAL batches for pretraining.

Why not ``transformers``' ``group_by_length``: its ``LengthGroupedSampler`` sorts megabatches of 50 x the
PER-DEVICE batch, and accelerate deals consecutive per-device batches to the ranks round-robin
(``BatchSamplerShard``: rank r gets batches r, r + W, ...). One optimizer step's W x A micro-batches then
span half a megabatch's length range, and the step waits for its longest one. Here the unit is the global
batch (per-device x accumulation x world size): every global batch is a block of similar lengths, so all
ranks pad to about the same length in every micro-step.

Order, per epoch (identical on every rank -- same seed, same epoch):
  1. a random permutation of the dataset;
  2. cut into megabatches of ``megabatches`` global batches; sort each by length, longest first;
  3. cut each megabatch into global batches; shuffle the global batches' order (the longest one is moved to
     the front, so an out-of-memory shows at the first step, as ``LengthGroupedSampler`` does).
Every index appears exactly once per epoch. With ``pad_to_multiple_of`` the collator rounds each batch's
longest spectrum up, so torch.compile sees at most max_peaks / pad_to_multiple_of shapes.
"""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
from torch.utils.data import Sampler


def spectrum_lengths(dataset) -> np.ndarray:
    """Peaks per row of a preprocessed ``datasets.Dataset`` (its ``mz`` list column), without a Python loop."""
    column = dataset.data.column("mz")
    lengths = pc.list_value_length(column).to_numpy(zero_copy_only=False)
    if dataset._indices is not None:  # a selected / shuffled view: map through the indices mapping
        lengths = lengths[dataset._indices.column(0).to_numpy()]
    return lengths.astype(np.int32)


class GlobalLengthGroupedSampler(Sampler[int]):
    """Yield dataset indices so that each run of ``global_batch`` consecutive indices has similar lengths."""

    def __init__(self, lengths, global_batch: int, megabatches: int = 50, seed: int = 0):
        if global_batch <= 0 or megabatches <= 0:
            raise ValueError("global_batch and megabatches must be positive")
        self.lengths = np.asarray(lengths)
        self.global_batch = int(global_batch)
        self.megabatches = int(megabatches)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lengths)

    def order(self) -> np.ndarray:
        n = len(self.lengths)
        rng = np.random.default_rng([self.seed, self.epoch])
        perm = rng.permutation(n)
        mega = self.global_batch * self.megabatches
        blocks = []
        for start in range(0, n, mega):
            chunk = perm[start:start + mega]
            chunk = chunk[np.argsort(-self.lengths[chunk], kind="stable")]
            blocks += [chunk[i:i + self.global_batch] for i in range(0, len(chunk), self.global_batch)]
        if not blocks:
            return perm
        # Shuffle the global batches, keeping each one whole; the final (possibly short) batch stays last so
        # every other batch is a full global batch.
        tail = blocks.pop() if len(blocks[-1]) < self.global_batch else None
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        if blocks:
            longest = max(range(len(blocks)), key=lambda i: self.lengths[blocks[i][0]])
            blocks[0], blocks[longest] = blocks[longest], blocks[0]
        if tail is not None:
            blocks.append(tail)
        return np.concatenate(blocks)

    def __iter__(self):
        return iter(self.order().tolist())
