"""Fine-tune the spectrum encoder so its embedding space separates peptides.

The alignment tower failed for a reason that had nothing to do with the student: the
frozen pretrained encoder's embedding space barely distinguishes peptides at all. On the
replicate corpus only 1% of peptide+charge groups are cleanly separated -- replicate
spectra of one peptide sit 0.074 apart on average and spectra of DIFFERENT peptides sit
0.100 apart, a margin of 0.026 on a scale where the space spans [0, 4]. Nothing asked
the pretrained model for that property: it was trained to predict masked peak
intensities, not to place replicates together.

So train for it directly, with two terms:

**Supervised contrastive** on the pooled embedding. Replicates of the same peptide and
charge are positives, everything else in the batch is a negative. This is the property
retrieval needs, optimised rather than hoped for.

**KL to the original model's head**, as a regulariser. The pretrained intensity head
emits one logit per peak and pretraining softmaxes those into a distribution over peaks,
so "stay close to what you used to predict" is exactly a KL divergence and needs no new
machinery. Without it, nothing stops the encoder from discarding the spectrum chemistry
that makes it worth starting from -- it could satisfy the contrastive term with any
arbitrary peptide-specific signature. With it, the encoder has to separate peptides
while still explaining the peaks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Sampler

from msdelta.reranking import pool_sequence, pooled_width


class GroupBatchSampler(Sampler[list[int]]):
    """P groups x K replicates per batch, because random batches have no positives.

    The corpus holds roughly 13 replicates of each of ~940 peptides. Drawing 48 rows at
    random gives about one positive PAIR per batch, so a contrastive loss would spend
    almost every step on negatives alone and learn very little. Sampling K replicates
    from each of P groups guarantees P*(K-1)*K/2 positives per batch by construction.

    Groups with fewer than K members are sampled with replacement: dropping them would
    quietly bias training toward peptides that happen to be observed often, which is
    exactly the population the metric is then evaluated on.

    RESHUFFLES BY ITSELF. The epoch counter advances inside __iter__ rather than waiting
    for set_epoch (FT14). HF Trainer builds this dataloader once and never calls
    set_epoch on a custom batch_sampler -- accelerate forwards set_epoch to
    `batch_sampler.sampler`, an attribute this class does not have -- so the counter sat
    at 0 and every epoch replayed byte-identical batches. With K replicates drawn from
    each group, one epoch touches K/mean_group_size of the corpus (K=2 over ~13
    replicates is 15.6%), and without reshuffling that same 15.6% was the ONLY data the
    model ever saw, no matter how many epochs were run. set_epoch still works and
    overrides the counter; nothing depends on the caller remembering to use it.
    """

    def __init__(self, groups, groups_per_batch: int = 12, replicates: int = 4,
                 seed: int = 0, drop_last: bool = True):
        if replicates < 2:
            raise ValueError("replicates must be >= 2 or there are no positive pairs")
        if groups_per_batch < 2:
            raise ValueError(
                "groups_per_batch must be >= 2 or a batch holds one peptide and has no "
                "NEGATIVES: every off-diagonal entry is a positive, the loss collapses "
                "to the constant log(replicates - 1), and nothing is learned")
        self.groups_per_batch = groups_per_batch
        self.replicates = replicates
        self.seed = seed
        self.drop_last = drop_last
        self.members: dict[int, np.ndarray] = {}
        groups = np.asarray(groups)
        for group in np.unique(groups):
            self.members[int(group)] = np.flatnonzero(groups == group)
        self.epoch = 0

    def __len__(self) -> int:
        return max(len(self.members) // self.groups_per_batch, 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        # Seed as a pair, not a sum: seed+epoch collides (seed 0 epoch 1 would draw the
        # same batches as seed 1 epoch 0), which would make a seed sweep partly a
        # relabelling of one trajectory rather than independent runs.
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(list(self.members))
        for start in range(0, len(order) - self.groups_per_batch + 1,
                           self.groups_per_batch):
            batch: list[int] = []
            for group in order[start : start + self.groups_per_batch]:
                pool = self.members[int(group)]
                take = rng.choice(pool, size=self.replicates,
                                  replace=len(pool) < self.replicates)
                batch.extend(int(i) for i in take)
            yield batch
        # Advance regardless of whether anyone calls set_epoch. See the class docstring.
        self.epoch += 1


class PairBatchSampler(Sampler[list[int]]):
    """Batches of PAIRS: half same-peptide, half different, ratio under direct control.

    The alternative to PK sampling. PK builds a batch of P peptides x K spectra and
    hands the whole thing to a softmax loss, which couples every row to every other --
    so the batch cannot be split, and memory caps how many peptides a single step can
    see. DeltaMZBias is O(batch * peaks^2), which is what forced P=2, K=2 and left every
    anchor with one positive and two negatives.

    A pair loss decomposes: each pair contributes independently, so a step can be made
    of as many small minibatches as you like via ordinary gradient accumulation, and the
    number of distinct peptides a step sees stops being a memory question. That is the
    whole point of the formulation.

    Each yielded batch is 2 * pairs_per_batch row indices, arranged so that rows
    (2i, 2i+1) are a pair. `positive_fraction` of them are same-peptide.

    RESHUFFLES BY ITSELF. The epoch counter advances inside __iter__ rather than waiting
    for set_epoch, because GroupBatchSampler took its epoch from a setter that nothing
    ever called (FT14) and consequently replayed identical batches for entire runs. A
    sampler that can only be correct if the caller remembers something is a sampler that
    will eventually be wrong.
    """

    def __init__(self, groups, pairs_per_batch: int = 8,
                 positive_fraction: float = 0.5, seed: int = 0,
                 batches_per_epoch: int | None = None):
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be strictly between 0 and 1: at 0 "
                             "nothing is ever pulled together, at 1 nothing is pushed "
                             "apart, and either way the loss is degenerate")
        if pairs_per_batch < 1:
            raise ValueError("pairs_per_batch must be >= 1")
        groups = np.asarray(groups)
        self.members: dict[int, np.ndarray] = {}
        for group in np.unique(groups):
            index = np.flatnonzero(groups == group)
            # A group with one spectrum can never supply a positive pair. It can still
            # serve as a negative, so it is kept for that and excluded from positives.
            self.members[int(group)] = index
        self.positive_pool = [g for g, m in self.members.items() if len(m) >= 2]
        if len(self.members) < 2:
            raise ValueError("need at least two groups to form a negative pair")
        if not self.positive_pool:
            raise ValueError("no group has two spectra, so no positive pair exists")
        self.pairs_per_batch = pairs_per_batch
        self.positive_fraction = positive_fraction
        self.seed = seed
        self.epoch = 0
        total_rows = int(sum(len(m) for m in self.members.values()))
        # Default epoch length covers the corpus once in expectation, so "epochs" means
        # roughly what it means elsewhere rather than one pass over groups.
        self.batches_per_epoch = batches_per_epoch or max(
            1, total_rows // (2 * pairs_per_batch))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        n_positive = max(1, min(self.pairs_per_batch - 1,
                                round(self.pairs_per_batch * self.positive_fraction)))
        groups = list(self.members)
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for _ in range(n_positive):
                pool = self.members[int(rng.choice(self.positive_pool))]
                batch.extend(int(i) for i in rng.choice(pool, size=2, replace=False))
            for _ in range(self.pairs_per_batch - n_positive):
                a, b = rng.choice(groups, size=2, replace=False)
                batch.append(int(rng.choice(self.members[int(a)])))
                batch.append(int(rng.choice(self.members[int(b)])))
            yield batch
        # Advance regardless of whether anyone calls set_epoch. See the class docstring.
        self.epoch += 1


def pair_contrastive_loss(embeddings: Tensor, groups: Tensor, margin: float = 1.0,
                          positive_weight: float = 1.0) -> dict[str, Tensor]:
    """Contrastive loss over adjacent PAIRS (Hadsell et al., 2006), not in-batch softmax.

    Rows arrive as (2i, 2i+1) pairs from PairBatchSampler. Same peptide: pull together.
    Different: push apart until they are at least `margin` apart, then stop caring.

    Embeddings are L2-normalised upstream, so the distance lives in [0, 2] and relates
    to cosine by d^2 = 2 - 2cos. A margin of 1.0 therefore asks different peptides to
    reach cosine 0.5 or below, which is a real separation without demanding
    orthogonality from spectra that genuinely share fragments.

    `positive_weight` rebalances the two terms when the sampler's ratio is not 1:1, so
    the mix can be tuned without the loss silently following it.

    WHAT THIS GIVES UP, stated because it is the reason not to default to it: a softmax
    loss puts every negative in one denominator, so the hardest negative automatically
    receives the most gradient. Independent pair terms weight every negative equally and
    contribute exactly zero once past the margin. The compensation is that pairs
    decompose, so batch size stops being bounded by memory.
    """
    if embeddings.shape[0] % 2:
        raise ValueError(f"pair loss needs an even number of rows, got "
                         f"{embeddings.shape[0]}")
    left, right = embeddings[0::2], embeddings[1::2]
    same = groups[0::2] == groups[1::2]
    distance = (left - right).norm(dim=-1)
    positive = distance.pow(2)
    negative = F.relu(margin - distance).pow(2)
    loss = torch.where(same, positive_weight * positive, negative).mean()
    return {
        "loss": loss,
        # Reported separately: a run where the positive term collapses while the
        # negative term does nothing looks identical in the total.
        "pair_positive": positive[same].mean().detach() if same.any()
                         else embeddings.new_zeros(()),
        "pair_negative": negative[~same].mean().detach() if (~same).any()
                         else embeddings.new_zeros(()),
        "pair_positive_fraction": same.float().mean().detach(),
        "pair_distance_same": distance[same].mean().detach() if same.any()
                              else embeddings.new_zeros(()),
        "pair_distance_diff": distance[~same].mean().detach() if (~same).any()
                              else embeddings.new_zeros(()),
    }


def supervised_contrastive_loss(embeddings: Tensor, groups: Tensor,
                                temperature: float = 0.07) -> Tensor:
    """SupCon (Khosla et al., NeurIPS 2020): every same-group pair is a positive.

    A thin wrapper over pytorch_metric_learning's SupConLoss, which this repository
    already depends on and which MSDeltaForRetrieval -- the downstream consumer -- uses
    for the same objective. Two implementations that agree today is a maintenance trap:
    the risk is not that they differ now, it is that someone fixes or tunes one.

    The hand-written version this replaces was verified numerically identical to the
    library across P=2/K=2, P=4/K=3 and P=8/K=2 at both temperatures, and on every
    degenerate input our samplers can produce. They diverged in exactly one case, a
    batch holding a single group: ours returned a real number whose gradient pushes
    toward uniformity among positives, the library returns 0. The library's behaviour is
    the better one -- with no negatives the softmax denominator holds only positives, so
    there is no discriminative signal and a gradient there is meaningless -- and
    GroupBatchSampler refuses to construct that batch anyway.

    ONE TRAP WORTH KEEPING, from the implementation that was deleted. Writing this by
    hand, the natural way to select an anchor's positives is to multiply log_prob by a
    boolean mask. That is wrong: log_prob holds -inf at the self-masked diagonal, and
    -inf * False is NaN, not 0, which silently poisons the whole batch. Select with
    torch.where instead of scaling by the mask.
    """
    return _SUPCON[temperature](F.normalize(embeddings.float(), dim=-1), groups)


class _SupConCache(dict):
    """One SupConLoss per temperature; it holds no state beyond that scalar."""

    def __missing__(self, temperature: float):
        from pytorch_metric_learning.losses import SupConLoss
        self[temperature] = SupConLoss(temperature=temperature)
        return self[temperature]


_SUPCON = _SupConCache()


def head_kl(logits: Tensor, reference_logits: Tensor, attention_mask: Tensor) -> Tensor:
    """KL(reference || current) over each spectrum's distribution across its real peaks.

    Matches how pretraining uses this head: it softmaxes the per-peak logits into a
    distribution and takes a KL against the target. Padding is masked to -inf so it
    takes no probability mass, and spectra are averaged rather than summed so the term
    does not scale with peak count.
    """
    valid = attention_mask.bool()
    current = logits.float().masked_fill(~valid, float("-inf")).log_softmax(dim=-1)
    reference = reference_logits.float().masked_fill(~valid, float("-inf")).log_softmax(dim=-1)
    # Computed termwise rather than through F.kl_div. Both tensors hold -inf at padded
    # positions, and F.kl_div would evaluate exp(-inf) * (-inf - -inf) = 0 * NaN = NaN
    # there, making the loss NaN for any batch containing padding -- which is all of
    # them. Zeroing the padded terms explicitly keeps the same value without the NaN.
    terms = reference.exp() * (reference - current)
    return torch.where(valid, terms, torch.zeros_like(terms)).sum(-1).mean()


class LayerMixPooler(nn.Module):
    """One trainable scalar per depth, mixed into a single d_model embedding.

    Every embedding this project has measured came off ONE layer -- almost always the
    last. The layer probe (job 8841973) showed that is the wrong layer at every scale:
    the separation ratio peaks mid-stack and sags in the final blocks, by 1.53 vs 1.43 at
    50m and 1.44 vs 1.35 at 200m. Picking the best single layer is a discrete search over
    a curve whose shape moves with scale. This learns the mixture instead, so the choice
    of depth becomes a trained parameter rather than a hyperparameter.

    Softmax over the scalars, ELMo-style, so the mix is a convex combination: the
    weights read directly as "how much of each depth", and no layer can be up-weighted
    without another giving way. `gamma` restores the overall scale that the softmax
    removes, since a convex combination cannot change magnitude on its own.

    PER-LAYER NORMALISATION IS ON BY DEFAULT, and it is not cosmetic. Raw block outputs
    differ by an order of magnitude across depth -- the 200m probe measured in-group
    distances of 0.0044 at block 2 against 0.0596 at block 15. Mixing those unnormalised
    means the deepest layers dominate the sum no matter what the weights say, so the
    "learned" mixture would be decided by scale before training started. A LayerNorm
    without affine parameters puts every depth on comparable footing, costs no
    parameters, and makes the learned weights mean what they appear to mean.

    Output is d_model, not 2*d_model: a sequence mean only, no max concatenation.
    """

    def __init__(self, num_layers: int, hidden_size: int, normalise: bool = True,
                 learn_scale: bool = True):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"need at least one layer to mix, got {num_layers}")
        # Zeros, so softmax starts uniform: the mixture begins as the mean of all depths
        # and training moves it, rather than starting at some arbitrary preference.
        self.mix = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(())) if learn_scale else None
        self.norm = (nn.LayerNorm(hidden_size, elementwise_affine=False)
                     if normalise else None)

    @property
    def weights(self) -> Tensor:
        """The mixture as it would be applied, for logging."""
        return torch.softmax(self.mix, dim=0)

    def forward(self, states: list[Tensor], attention_mask: Tensor) -> Tensor:
        if len(states) != self.mix.numel():
            raise ValueError(f"expected {self.mix.numel()} layer states, got {len(states)}")
        stacked = torch.stack([self.norm(h) if self.norm is not None else h
                               for h in states], dim=0)          # (L, B, S, D)
        weights = torch.softmax(self.mix, dim=0).to(stacked.dtype)
        mixed = (stacked * weights[:, None, None, None]).sum(0)   # (B, S, D)
        mask = attention_mask.unsqueeze(-1).to(mixed.dtype)
        pooled = (mixed * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        if self.gamma is not None:
            pooled = pooled * self.gamma
        return pooled


def encoder_layer_states(encoder: nn.Module, mz: Tensor, log_intensity: Tensor,
                         attention_mask: Tensor) -> tuple[list[Tensor], Tensor]:
    """Run the encoder and keep every block's output, plus the embedding it started from.

    MSDeltaModel.forward returns only the final normalised state, so the intermediates
    have to be captured. Forward hooks rather than a reimplemented forward: the block
    loop carries the bias tensor, the padding mask and a gradient-checkpointing branch,
    and a copy of it here would be a second thing to keep in step with the model.

    The captured tensors are the real block outputs, so they carry gradients and the
    mixture trains the encoder through every depth it draws on.

    Gradient checkpointing is refused rather than silently mishandled: under it each
    block runs twice, once under no_grad to find the boundaries and once to recompute,
    so a hook fires twice per block and the first tensor is detached from the graph.
    Taking the second is correct but depends on recompute order, which is not a thing to
    rely on quietly.
    """
    if getattr(encoder, "gradient_checkpointing", False) and encoder.training:
        raise RuntimeError(
            "layer-mix pooling cannot read intermediate states under gradient "
            "checkpointing; disable one of them"
        )
    captured: dict[int, Tensor] = {}
    handles = [encoder.embed.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__(0, out))]
    for index, block in enumerate(encoder.blocks, start=1):
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, index=index: captured.__setitem__(index, out)))
    try:
        final = encoder(mz=mz, log_intensity=log_intensity,
                        attention_mask=attention_mask).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    expected = len(encoder.blocks) + 1
    if len(captured) != expected:
        raise RuntimeError(f"captured {len(captured)} of {expected} layer states")
    return [captured[i] for i in range(expected)], final


def layer_mix_width(model: nn.Module) -> int:
    """d_model: a sequence mean, with no max concatenation."""
    return _hidden_size(model)


def _hidden_size(model: nn.Module) -> int:
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        hidden = getattr(model.config.encoder, "hidden_size")
    return hidden


class MSDeltaForContrastive(nn.Module):
    """Spectrum encoder trained to separate peptides while still explaining peaks."""

    def __init__(self, model: nn.Module, reference: nn.Module | None = None,
                 pooling: str = "mean+max", temperature: float = 0.07,
                 kl_weight: float = 1.0, layer_mix_norm: bool = True,
                 pair_loss: bool = False, pair_margin: float = 1.0,
                 pair_positive_weight: float = 1.0):
        super().__init__()
        self.model = model
        self.reference = reference
        # pooling="layer_mix" replaces the fixed readout with a trained one over depth.
        # Built here rather than passed in so its size always matches this encoder.
        self.layer_mix = None
        if pooling == "layer_mix":
            encoder = getattr(model, "msdelta", model)
            self.layer_mix = LayerMixPooler(
                num_layers=len(encoder.blocks) + 1,   # + the pre-block embedding
                hidden_size=_hidden_size(model),
                normalise=layer_mix_norm,
            )
        if self.reference is not None:
            # Frozen AND in eval mode: dropout would make the regularisation target move
            # every step, and the encoder would chase noise instead of staying put.
            self.reference.requires_grad_(False)
            self.reference.eval()
        self.pooling = pooling
        self.temperature = temperature
        self.kl_weight = kl_weight
        # Pair mode replaces the in-batch softmax with independent per-pair terms; see
        # pair_contrastive_loss. It expects rows arranged as (2i, 2i+1) pairs, which is
        # what PairBatchSampler yields, so the two must be switched on together.
        self.pair_loss = pair_loss
        self.pair_margin = pair_margin
        self.pair_positive_weight = pair_positive_weight

    def train(self, mode: bool = True):
        super().train(mode)
        if self.reference is not None:
            self.reference.eval()
        return self

    # Trainer calls these on the model it is given, and this is a plain nn.Module rather
    # than a PreTrainedModel, so they have to be forwarded by hand. Only the trainable
    # encoder gets them: the reference runs under no_grad and stores no activations, so
    # checkpointing it would add recomputation for no saving.
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.model, "is_gradient_checkpointing", False))

    def embed(self, mz, log_intensity, attention_mask) -> tuple[Tensor, Tensor]:
        encoder = getattr(self.model, "msdelta", self.model)
        if self.layer_mix is not None:
            # `hidden` stays the FINAL state even when the embedding mixes every depth:
            # it feeds the intensity head for the KL term, whose job is to hold the
            # pretraining behaviour still. That behaviour lives at the output, so
            # regularising a mixture there would constrain something the head never used.
            states, hidden = encoder_layer_states(encoder, mz, log_intensity,
                                                  attention_mask)
            pooled = self.layer_mix(states, attention_mask)
        else:
            hidden = encoder(mz=mz, log_intensity=log_intensity,
                             attention_mask=attention_mask).last_hidden_state
            pooled = pool_sequence(hidden, attention_mask, self.pooling)
        return F.normalize(pooled.float(), dim=-1), hidden

    def forward(self, mz, log_intensity, attention_mask, group,
                reference_logits=None, return_dict: bool = True,
                return_loss: bool = True):
        embeddings, hidden = self.embed(mz, log_intensity, attention_mask)
        extra: dict[str, Tensor] = {}
        if self.pair_loss:
            pair = pair_contrastive_loss(embeddings, group, self.pair_margin,
                                         self.pair_positive_weight)
            contrastive = pair.pop("loss")
            extra = pair
        else:
            contrastive = supervised_contrastive_loss(embeddings, group,
                                                      self.temperature)

        kl = embeddings.new_zeros(())
        if self.kl_weight > 0:
            logits = self.model.intensity_head(hidden)
            if reference_logits is None:
                if self.reference is None:
                    raise ValueError("kl_weight > 0 needs a reference model or cached "
                                     "reference_logits")
                with torch.no_grad():
                    reference_hidden = getattr(
                        self.reference, "msdelta", self.reference
                    )(mz=mz, log_intensity=log_intensity,
                      attention_mask=attention_mask).last_hidden_state
                    reference_logits = self.reference.intensity_head(reference_hidden)
            kl = head_kl(logits, reference_logits, attention_mask)

        loss = contrastive + self.kl_weight * kl
        if not return_dict:
            return (loss, embeddings)
        return {"loss": loss, "contrastive": contrastive.detach(), "kl": kl.detach(),
                "embeddings": embeddings, **extra}


def embedding_size(model: nn.Module, pooling: str = "mean+max") -> int:
    if pooling == "layer_mix":
        return layer_mix_width(model)
    return pooled_width(_hidden_size(model), pooling)


@torch.no_grad()
def group_separation_summary(model, dataset, collator, device, max_rows: int = 2000,
                             batch_size: int = 16) -> dict[str, float]:
    """Embed a validation split and report whether replicates now cluster.

    The pretrained encoder scores `clean` = 0.010 on this corpus -- 1% of peptide+charge
    groups have every replicate nearer to each other than to any other peptide. If
    contrastive training does not raise that substantially, it has not done its job,
    whatever the loss curve says.

    This is a PROXY for retrieval, not the task. See retrieval_summary, which scores the
    same rows on Hit@1, MAP@100 and R@5 so the two can be compared rather than one
    assumed to stand for the other.
    """
    from msdelta.reranking import group_separation_metrics

    embeddings, groups = embed_dataset(model, dataset, collator, device,
                                       max_rows=max_rows, batch_size=batch_size)
    if embeddings is None:
        return {}
    return group_separation_metrics(embeddings, groups, "sep_spectrum")


def embed_dataset(model, dataset, collator, device, max_rows: int = 2000,
                  batch_size: int = 16):
    """Embed rows and return (embeddings, group ids). Shared by both evaluations.

    Factored out so the separation ratio and the retrieval metrics are computed over
    exactly the same rows in the same order. Two evaluations that disagree because they
    embedded different subsets would be the worst possible outcome here, since the whole
    point is to find out whether one predicts the other.
    """
    from msdelta.reranking import peptide_key

    was_training = model.training
    model.eval()
    rows = list(dataset)[:max_rows]
    embeddings = []
    try:
        # no_grad is not an optimisation here, it is the difference between running and
        # not. model.eval() alone still builds the autograd graph, and .cpu() keeps it
        # alive through grad_fn, so every batch's DeltaMZBias intermediates -- which are
        # (batch, peaks, peaks, 2*n_freqs) and measured in gigabytes -- are retained for
        # the whole pass. Job 8848808 filled a 64 GB tile on its first encoder that way,
        # and shrinking the batch did not help because the leak scales with the NUMBER
        # of batches, not their size.
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                batch = {k: v.to(device) for k, v in collator(chunk).items()}
                pooled, _ = model.embed(batch["mz"], batch["log_intensity"],
                                        batch["attention_mask"])
                embeddings.append(pooled.cpu())
    finally:
        model.train(was_training)
    if not embeddings:
        return None, None
    groups = np.unique(
        np.array([peptide_key(r["peptide"], int(r.get("charge", 0))) for r in rows]),
        return_inverse=True)[1]
    return torch.cat(embeddings), groups


def retrieval_metrics_exact(embeddings: Tensor, groups) -> dict[str, float]:
    """Hit@1, R@5 and MAP@100 by exact search, without faiss.

    faiss is imported at module level by msdelta.retrieval and is not installed in this
    environment -- not in the venv, not in the frameworks module -- so that path cannot
    run here at all. It is also unnecessary: the evaluation embeds at most a couple of
    thousand spectra, and an exact all-pairs cosine similarity at that size is a single
    matmul. An approximate index would add a dependency, a build, and a recall ceiling
    in exchange for nothing.

    DEFINITIONS, stated because retrieval metrics are named inconsistently. The last
    three are the metric-learning standards and carry their literature names:

      Hit@1        the nearest OTHER spectrum is the same peptide. Identical to
                   Precision@1; reported under both names so our older tables still
                   line up.
      R@5          of a query's relevant spectra, the fraction appearing in its top 5.
                   NOT the Recall@K of the retrieval literature, which is the FRACTION
                   OF QUERIES with at least one hit in the top K. Kept for continuity
                   with earlier tables; prefer the three below.
      Precision@1  Musgrave et al. name for Hit@1.
      R-Precision  of a query's R relevant spectra, the fraction that appear in its own
                   top R. Adapts the cutoff to each query.
      MAP@R        average precision over the top R, R being THAT QUERY's number of
                   relevant spectra, zero-padded past the last hit. The metric Musgrave
                   et al. recommend, and the right one here: our group sizes run 11 to
                   120 with a median of 13, so a fixed cutoff asks a much harder
                   question of a 120-replicate peptide than an 11-replicate one.
      MAP@100      average precision over the top 100, ALL relevant spectra in the
                   denominator. Retained because every earlier contrastive number in
                   this project is quoted in it. Only 1 group of 1000 exceeds 100
                   members, so it almost never truncates here.

    A query whose peptide has no other spectrum is excluded: it has no correct answer
    available, and scoring it as a miss would understate retrieval by however many
    singletons the split happens to contain.

    Musgrave, Belongie and Lim, "A Metric Learning Reality Check", ECCV 2020,
    arXiv:2003.08505 -- the source of MAP@R and R-Precision, and of the argument that
    Recall@K saturates and hides ranking quality.
    """
    e = F.normalize(embeddings.float(), dim=-1)
    g = torch.as_tensor(groups, dtype=torch.long)
    sim = e @ e.T
    n = len(e)
    eye = torch.eye(n, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    relevant = (g[:, None] == g[None, :]) & ~eye
    n_rel = relevant.sum(1)
    scorable = n_rel > 0
    if not scorable.any():
        return {}

    order = sim.argsort(dim=1, descending=True)
    hit = relevant.gather(1, order)                      # relevance in rank order

    at5 = hit[:, :5].sum(1).float() / n_rel.clamp(min=1).float()
    k = min(100, n - 1)
    top = hit[:, :k].float()
    # precision@i at each rank where a relevant item sits, averaged over all relevant.
    csum = top.cumsum(1)
    ranks = torch.arange(1, k + 1, dtype=torch.float32).unsqueeze(0)
    ap = ((csum / ranks) * top).sum(1) / n_rel.clamp(min=1).float()

    # MAP@R and R-Precision: the cutoff is PER QUERY, so build a mask of rank < n_rel
    # rather than slicing a fixed width. Ranks past a query's R contribute nothing.
    # n columns, not n-1: argsort ranks all n items. The self-match sorts last (its
    # similarity is -inf) and is never relevant, so it contributes nothing.
    all_ranks = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(0)
    within_r = all_ranks <= n_rel.unsqueeze(1).float()
    hit_f = hit.float()
    # R-Precision: hits inside the query's own top R, over R.
    r_prec = (hit_f * within_r).sum(1) / n_rel.clamp(min=1).float()
    # MAP@R: precision@i at each hit inside the top R, divided by R -- the zero padding
    # past the last hit is what the division by R (not by the hit count) accomplishes.
    csum_all = hit_f.cumsum(1)
    ap_r = (((csum_all / all_ranks) * hit_f) * within_r).sum(1) / n_rel.clamp(min=1).float()

    p1 = float(hit[scorable, 0].float().mean())
    return {
        "Hit@1": p1,
        "Precision@1": p1,          # the literature name for the same quantity
        "R-Precision": float(r_prec[scorable].mean()),
        "MAP@R": float(ap_r[scorable].mean()),
        "R@5": float(at5[scorable].mean()),
        "MAP@100": float(ap[scorable].mean()),
    }


def retrieval_summary(model, dataset, collator, device, max_rows: int = 2000,
                      batch_size: int = 16) -> dict[str, float]:
    """Hit@1, MAP@100 and R@5 -- the task, rather than a proxy for it.

    THE SEPARATION RATIO IS NOT THE GOAL. Contrastive training exists here to serve
    retrieval and reranking, and every contrastive result in this project has been
    scored on out-group over in-group mean distance instead. That is a reasonable proxy
    and it has never been checked against the thing it proxies for. The two can come
    apart: the ratio is an average over all pairs, while retrieval depends only on the
    nearest few, so a model that tightens the bulk of the distribution while leaving the
    hardest confusions untouched improves the ratio and not the task.

    Reported alongside the ratio on the same rows, so the correlation between them can
    be measured across a grid rather than assumed.
    """
    embeddings, groups = embed_dataset(model, dataset, collator, device,
                                       max_rows=max_rows, batch_size=batch_size)
    if embeddings is None or len(embeddings) < 3:
        return {}
    counts = np.bincount(groups)
    if (counts > 1).sum() < 2:
        return {}
    scores = retrieval_metrics_exact(embeddings, groups)
    if not scores:
        return {}
    return {f"retrieval/{k}": v for k, v in scores.items()} | {
        "retrieval/queries": float(len(embeddings)),
        "retrieval/groups": float(len(counts)),
        "retrieval/scorable_groups": float((counts > 1).sum()),
    }


def subset_by_group(dataset, max_samples: int, group_key, min_members: int = 4):
    """Cap a split by whole GROUPS, not by row.

    subset_splits takes a contiguous slice, which is right for denoise and wrong here.
    The corpus interleaves replicates, so the first 1,200 rows touch 625 of the 1,000
    peptide+charge groups with about two members each -- and a contrastive smoke test on
    that would sample almost entirely with replacement, train on duplicate spectra, and
    tell you nothing about whether the objective works. Keeping whole groups preserves
    the replicate structure the loss depends on.
    """
    if max_samples <= 0:
        return dataset
    keys = np.array([group_key(row) for row in dataset])
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    keep, total = set(), 0
    for key, size in sorted(counts.items(), key=lambda kv: -kv[1]):
        if size < min_members:
            continue
        if total + size > max_samples and keep:
            break
        keep.add(key)
        total += size
    indices = [i for i, key in enumerate(keys) if key in keep]
    return dataset.select(indices)


def _rng_state(device) -> tuple:
    """CPU and device RNG, so a replayed forward draws the same dropout masks."""
    device_state = None
    if device.type == "xpu" and torch.xpu.is_available():
        device_state = torch.xpu.get_rng_state(device)
    elif device.type == "cuda" and torch.cuda.is_available():
        device_state = torch.cuda.get_rng_state(device)
    return torch.get_rng_state(), device_state


def _restore_rng(state: tuple, device) -> None:
    cpu_state, device_state = state
    torch.set_rng_state(cpu_state)
    if device_state is None:
        return
    if device.type == "xpu":
        torch.xpu.set_rng_state(device_state, device)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(device_state, device)


def gradcache_step(model, batch, chunk_size: int, accelerator=None) -> dict[str, Tensor]:
    """One optimizer step with a contrastive batch far larger than memory allows.

    The epochs ladder settled what the constraint is. Three epochs took the separation
    ratio to 6.94; ten epochs took it DOWN to 4.82 with `clean` falling to zero. The
    contrastive loss was already 0.005 against a chance value of 1.099 after three
    epochs, so the training task was solved and further steps only overfit it -- and the
    task is trivially easy because a batch of four spectra offers four negatives. More
    steps cannot help. More NEGATIVES can, and DeltaMZBias at O(batch * peaks^2) caps the
    batch at four spectra of 512 peaks on a tile.

    GradCache (Gao et al., 2021) breaks that link. The contrastive loss needs every
    embedding at once, but it does not need every ACTIVATION at once:

      1. embed every chunk under no_grad, keeping only the embeddings
      2. compute the loss over all of them and get dL/dembedding
      3. re-embed each chunk WITH grad and backprop that cached gradient through it

    Peak memory is one chunk's activations regardless of batch size; the cost is
    embedding each chunk twice. The KL term is per-sample, so it rides along in step 3
    where the activations already exist.

    The gradient is exact, not an approximation, and `tests/test_contrastive.py` asserts
    that against a direct full-batch backward.
    """
    keys = ("mz", "log_intensity", "attention_mask")
    total = len(batch["group"])
    chunks = [{k: batch[k][i:i + chunk_size] for k in keys}
              for i in range(0, total, chunk_size)]

    # 1. embeddings only, no graph -- capturing the RNG state before each chunk.
    #
    # The recomputation in step 3 must be BIT-IDENTICAL to this one, and in train mode
    # it is not: dropout draws fresh masks, so the two passes embed the same spectra
    # differently and the cached gradient no longer belongs to the graph it is pushed
    # through. Measured before this was added, the full-batch and GradCache losses
    # disagreed at the third decimal (1.9495 vs 1.9409) and the worst parameter gradient
    # was off by 27x. Replaying the seed per chunk makes the passes identical.
    states = []
    cached = []
    with torch.no_grad():
        for chunk in chunks:
            states.append(_rng_state(chunk["mz"].device))
            cached.append(model.embed(chunk["mz"], chunk["log_intensity"],
                                      chunk["attention_mask"])[0])

    # 2. the loss over the WHOLE batch, differentiated only w.r.t. the embeddings.
    leaves = [e.detach().requires_grad_(True) for e in cached]
    contrastive = supervised_contrastive_loss(
        torch.cat(leaves), batch["group"], model.temperature)
    contrastive.backward()
    grads = [leaf.grad for leaf in leaves]

    # 3. re-embed with grad and push the cached gradient through, adding the KL term
    #    here because this is where the activations exist.
    kl_total = torch.zeros((), device=contrastive.device)
    for chunk, grad, state in zip(chunks, grads, states):
        _restore_rng(state, chunk["mz"].device)
        embeddings, hidden = model.embed(chunk["mz"], chunk["log_intensity"],
                                         chunk["attention_mask"])
        surrogate = (embeddings * grad).sum()
        if model.kl_weight > 0 and model.reference is not None:
            logits = model.model.intensity_head(hidden)
            with torch.no_grad():
                reference_hidden = getattr(
                    model.reference, "msdelta", model.reference
                )(mz=chunk["mz"], log_intensity=chunk["log_intensity"],
                  attention_mask=chunk["attention_mask"]).last_hidden_state
                reference_logits = model.reference.intensity_head(reference_hidden)
            # Scaled by the chunk's share of the batch so the total matches what a
            # single full-batch step would have computed.
            share = len(chunk["mz"]) / total
            kl = head_kl(logits, reference_logits, chunk["attention_mask"])
            kl_total = kl_total + kl.detach() * share
            surrogate = surrogate + model.kl_weight * kl * share
        if accelerator is not None:
            accelerator.backward(surrogate)
        else:
            surrogate.backward()

    return {"loss": (contrastive.detach() + model.kl_weight * kl_total),
            "contrastive": contrastive.detach(), "kl": kl_total}
