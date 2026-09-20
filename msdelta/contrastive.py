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
    """

    def __init__(self, groups, groups_per_batch: int = 12, replicates: int = 4,
                 seed: int = 0, drop_last: bool = True):
        if replicates < 2:
            raise ValueError("replicates must be >= 2 or there are no positive pairs")
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
        rng = np.random.default_rng(self.seed + self.epoch)
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


def supervised_contrastive_loss(embeddings: Tensor, groups: Tensor,
                                temperature: float = 0.07) -> Tensor:
    """SupCon: every same-group pair is a positive, not just one.

    Plain InfoNCE assumes a single positive per anchor. Here a batch holds K replicates
    of each peptide, so all K-1 of them are positives and the loss averages over them --
    which is both more signal per step and the right objective, since no replicate is
    privileged over another.
    """
    embeddings = F.normalize(embeddings.float(), dim=-1)
    logits = embeddings @ embeddings.T / temperature
    # Exclude self-similarity, which is 1/temperature and would dominate every row.
    self_mask = torch.eye(len(embeddings), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    positives = (groups[:, None] == groups[None, :]) & ~self_mask
    counts = positives.sum(1)
    if not (counts > 0).any():
        return logits.new_zeros(())

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # torch.where, not multiplication by the mask: log_prob holds -inf on the diagonal
    # from the self-masking above, and -inf * False is NaN rather than 0, which poisons
    # the whole batch. Select instead of scale.
    contributions = torch.where(positives, log_prob, torch.zeros_like(log_prob))
    # Mean log-probability over an anchor's positives, then averaged over the anchors
    # that have any. An anchor whose group is a singleton in this batch contributes
    # nothing rather than a zero that would dilute the mean.
    valid = counts > 0
    per_anchor = contributions.sum(1)[valid] / counts[valid]
    return -per_anchor.mean()


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


class MSDeltaForContrastive(nn.Module):
    """Spectrum encoder trained to separate peptides while still explaining peaks."""

    def __init__(self, model: nn.Module, reference: nn.Module | None = None,
                 pooling: str = "mean+max", temperature: float = 0.07,
                 kl_weight: float = 1.0):
        super().__init__()
        self.model = model
        self.reference = reference
        if self.reference is not None:
            # Frozen AND in eval mode: dropout would make the regularisation target move
            # every step, and the encoder would chase noise instead of staying put.
            self.reference.requires_grad_(False)
            self.reference.eval()
        self.pooling = pooling
        self.temperature = temperature
        self.kl_weight = kl_weight

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
        hidden = encoder(mz=mz, log_intensity=log_intensity,
                         attention_mask=attention_mask).last_hidden_state
        pooled = F.normalize(pool_sequence(hidden, attention_mask, self.pooling).float(),
                             dim=-1)
        return pooled, hidden

    def forward(self, mz, log_intensity, attention_mask, group,
                reference_logits=None, return_dict: bool = True,
                return_loss: bool = True):
        embeddings, hidden = self.embed(mz, log_intensity, attention_mask)
        contrastive = supervised_contrastive_loss(embeddings, group, self.temperature)

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
                "embeddings": embeddings}


def embedding_size(model: nn.Module, pooling: str = "mean+max") -> int:
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        hidden = getattr(model.config.encoder, "hidden_size")
    return pooled_width(hidden, pooling)


@torch.no_grad()
def group_separation_summary(model, dataset, collator, device, max_rows: int = 2000,
                             batch_size: int = 16) -> dict[str, float]:
    """Embed a validation split and report whether replicates now cluster.

    This is the number the whole exercise exists to move. The pretrained encoder scores
    `clean` = 0.010 on this corpus -- 1% of peptide+charge groups have every replicate
    nearer to each other than to any other peptide. If contrastive training does not
    raise that substantially, it has not done its job, whatever the loss curve says.
    """
    from msdelta.reranking import group_separation_metrics, peptide_key

    was_training = model.training
    model.eval()
    rows = list(dataset)[:max_rows]
    embeddings = []
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = {k: v.to(device) for k, v in collator(chunk).items()}
            pooled, _ = model.embed(batch["mz"], batch["log_intensity"],
                                    batch["attention_mask"])
            embeddings.append(pooled.cpu())
    finally:
        model.train(was_training)
    if not embeddings:
        return {}
    groups = np.unique(
        np.array([peptide_key(r["peptide"], int(r.get("charge", 0))) for r in rows]),
        return_inverse=True)[1]
    return group_separation_metrics(torch.cat(embeddings), groups, "sep_spectrum")


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
