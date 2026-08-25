"""Distributed representation diagnostics owned by Hugging Face Trainer."""
from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .data import EmbeddingEvalCollator
from .probe import (
    _LOSSES,
    _isotope_labels,
    parse_charge,
    precursor_mz,
    probe_metrics_from_representations,
)
from .retrieval import _label_index, retrieval_metrics_tm


def build_embedding_transform(n_components: int = 256) -> Pipeline:
    """Established, serializable PCA whitening + cosine normalization."""
    return Pipeline([
        ("pca", PCA(
            n_components=n_components,
            whiten=True,
            svd_solver="randomized",
            random_state=0,
        )),
        ("normalize", Normalizer(norm="l2")),
    ])


@dataclass
class DiagnosticCorpora:
    calibration: object
    probes: object
    internal_retrieval: object
    internal_binned: np.ndarray
    external_retrieval: object | None = None


class DistributedDiagnosticRunner:
    """Run compiled inference on every rank and fit/score on global rank zero."""

    def __init__(
        self,
        trainer,
        corpora: DiagnosticCorpora,
        *,
        max_peaks: int,
        batch_size: int,
        num_workers: int,
        pca_components: int = 256,
        max_peak_samples: int = 80_000,
        max_pair_samples: int = 80_000,
    ):
        self.trainer = trainer
        self.accelerator = trainer.accelerator
        self.corpora = corpora
        self.pca_components = pca_components
        self.max_peak_samples = max_peak_samples
        self.max_pair_samples = max_pair_samples
        collator = EmbeddingEvalCollator(max_peaks=max_peaks)

        def loader(dataset):
            return self.accelerator.prepare_data_loader(DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=True,
            ))

        self.loaders = {
            "calibration": loader(corpora.calibration),
            "probes": loader(corpora.probes),
            "internal": loader(corpora.internal_retrieval),
        }
        if corpora.external_retrieval is not None:
            self.loaders["external"] = loader(corpora.external_retrieval)
        self.last_transform: Pipeline | None = None

    def _gather_variable(self, value: torch.Tensor) -> torch.Tensor:
        """Gather a variable leading dimension without gathering Python objects."""
        count = torch.tensor([value.shape[0]], device=self.accelerator.device)
        counts = self.accelerator.gather(count).long().cpu()
        padded = self.accelerator.pad_across_processes(value, dim=0, pad_index=0)
        gathered = self.accelerator.gather(padded).cpu()
        width = padded.shape[0]
        pieces = [
            gathered[i * width:i * width + int(n)]
            for i, n in enumerate(counts.tolist())
        ]
        return torch.cat(pieces, dim=0) if pieces else gathered[:0]

    def _sample_representations(
        self,
        tokens: torch.Tensor,
        batch: dict,
        total_spectra: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        peak_rep, peak_iso, peak_mz = [], [], []
        pair_rep, pair_loss = [], []
        loss_vals = np.array(list(_LOSSES.values()))
        per_spectrum = max(1, self.max_peak_samples // max(1, total_spectra) * 4)

        for b, pc in enumerate(batch["peptide_charge"]):
            K = int(batch["peak_count"][b])
            if K == 0:
                continue
            mz = batch["mz"][b, :K].float().cpu().numpy()
            row_id = int(batch["row_id"][b])
            rng = np.random.default_rng(row_id)
            z = parse_charge(pc)
            if z and 1 <= z <= 5:
                iso = _isotope_labels(np.argsort(mz), mz, z)
                take = min(K, per_spectrum)
                selected = rng.choice(K, size=take, replace=False)
                selected_t = torch.as_tensor(selected, device=tokens.device)
                peak_rep.append(tokens[b, selected_t])
                peak_iso.append(torch.as_tensor(iso[selected], device=tokens.device))
                peak_mz.append(torch.as_tensor(mz[selected], device=tokens.device))

            if K >= 4:
                dm = mz[:, None] - mz[None, :]
                ii, jj = np.where(dm > 0)
                delta = dm[ii, jj]
                positive = np.abs(delta[:, None] - loss_vals).min(1) < 0.01
                pos = np.where(positive)[0]
                neg = np.where(~positive)[0]
                npos = min(len(pos), 4)
                if npos:
                    chosen = np.concatenate([
                        rng.choice(pos, npos, replace=False),
                        rng.choice(neg, min(npos, len(neg)), replace=False),
                    ])
                    ti = torch.as_tensor(ii[chosen], device=tokens.device)
                    tj = torch.as_tensor(jj[chosen], device=tokens.device)
                    pair_rep.append(torch.cat([tokens[b, ti], tokens[b, tj]], dim=-1))
                    pair_loss.append(torch.as_tensor(
                        positive[chosen], device=tokens.device, dtype=torch.long
                    ))

        D = tokens.shape[-1]
        device = tokens.device
        return (
            torch.cat(peak_rep).to(torch.bfloat16) if peak_rep else torch.empty(0, D, device=device, dtype=torch.bfloat16),
            torch.cat(peak_iso) if peak_iso else torch.empty(0, device=device, dtype=torch.long),
            torch.cat(peak_mz) if peak_mz else torch.empty(0, device=device),
            torch.cat(pair_rep).to(torch.bfloat16) if pair_rep else torch.empty(0, 2 * D, device=device, dtype=torch.bfloat16),
            torch.cat(pair_loss) if pair_loss else torch.empty(0, device=device, dtype=torch.long),
        )

    @torch.no_grad()
    def _extract(self, name: str, *, representations: bool = False):
        model = self.trainer.model_wrapped
        was_training = model.training
        model.eval()
        pooled_local, ids_local, meta_local = [], [], []
        peak_local = [[], [], []]
        pair_local = [[], []]
        dataset = getattr(self.corpora, {
            "calibration": "calibration",
            "probes": "probes",
            "internal": "internal_retrieval",
            "external": "external_retrieval",
        }[name])
        try:
            for batch in self.loaders[name]:
                model_inputs = {
                    "mz": batch["mz"],
                    "log_int": batch["log_int"],
                    "key_padding_mask": batch["key_padding_mask"],
                    "eval_mode": "representations" if representations else "embedding",
                }
                with self.trainer.compute_loss_context_manager():
                    output = model(**model_inputs)
                pooled_local.append(output.embeddings.detach().to(torch.float32))
                ids_local.append(batch["row_id"])
                if representations:
                    pc = batch["peptide_charge"]
                    numeric = []
                    for b, value in enumerate(pc):
                        z = parse_charge(value)
                        K = int(batch["peak_count"][b])
                        numeric.append([
                            precursor_mz(value) or float("nan"),
                            K,
                            float(batch["log_tic"][b]),
                            z if z and 1 <= z <= 5 else 0,
                            float(batch["mz"][b, :K].max()) if K else float("nan"),
                        ])
                    meta_local.append(torch.tensor(
                        numeric, device=self.accelerator.device, dtype=torch.float32
                    ))
                    sampled = self._sample_representations(
                        output.token_embeddings.detach(), batch, len(dataset)
                    )
                    for target, value in zip(peak_local + pair_local, sampled):
                        target.append(value)
        finally:
            if was_training:
                model.train()

        pooled = self._gather_variable(torch.cat(pooled_local))
        row_ids = self._gather_variable(torch.cat(ids_local))
        order = torch.argsort(row_ids)
        # DistributedSampler-style padding can repeat rows; stable IDs make the
        # final corpus exact and ordered.
        ordered_ids = row_ids[order]
        keep = torch.ones_like(ordered_ids, dtype=torch.bool)
        keep[1:] = ordered_ids[1:] != ordered_ids[:-1]
        result = {
            "embeddings": pooled[order][keep].float().numpy(),
            "row_ids": ordered_ids[keep].numpy(),
        }
        if representations:
            meta = self._gather_variable(torch.cat(meta_local))[order][keep]
            result.update({
                "prec": meta[:, 0].numpy(),
                "pcount": meta[:, 1].numpy(),
                "logtic": meta[:, 2].numpy(),
                "charge": meta[:, 3].numpy(),
                "maxmz": meta[:, 4].numpy(),
                "peak_rep": self._gather_variable(torch.cat(peak_local[0])).float().numpy()[:self.max_peak_samples],
                "peak_iso": self._gather_variable(torch.cat(peak_local[1])).numpy()[:self.max_peak_samples],
                "peak_mz": self._gather_variable(torch.cat(peak_local[2])).numpy()[:self.max_peak_samples],
                "pair_rep": self._gather_variable(torch.cat(pair_local[0])).float().numpy()[:self.max_pair_samples],
                "pair_loss": self._gather_variable(torch.cat(pair_local[1])).numpy()[:self.max_pair_samples],
            })
        return result

    def _retrieval_metrics(
        self, prefix, embeddings, labels, transform, *, pairwise=False,
    ):
        raw = Normalizer(norm="l2").transform(embeddings)
        cleaned = transform.transform(embeddings)
        raw_metrics = retrieval_metrics_tm(
            raw, labels, self.accelerator.device, ks=(5,), pairwise=pairwise
        )
        pca_metrics = retrieval_metrics_tm(
            cleaned, labels, self.accelerator.device, ks=(5,), pairwise=pairwise
        )
        names = {"P@1": "Hit@1", "mAP": "MAP"} if prefix == "replicate_retrieval" else {}
        result = {
            f"{prefix}/{names.get(k, k)}": v for k, v in raw_metrics.items()
        }
        result.update({
            f"{prefix}/{names.get(k, k)}_pca": v for k, v in pca_metrics.items()
        })
        return result

    def run(self) -> dict[str, float]:
        started = time.monotonic()
        calibration = self._extract("calibration")
        probes = self._extract("probes", representations=True)
        internal = self._extract("internal")
        external = self._extract("external") if "external" in self.loaders else None
        metrics: dict[str, float] = {}

        if self.accelerator.is_main_process:
            if calibration["embeddings"].shape[0] <= self.pca_components:
                raise ValueError(
                    f"PCA-{self.pca_components} needs at least "
                    f"{self.pca_components + 1} calibration spectra"
                )
            self.last_transform = build_embedding_transform(self.pca_components)
            self.last_transform.fit(calibration["embeddings"])
            metrics.update(probe_metrics_from_representations({
                "spec": probes["embeddings"],
                **{k: probes[k] for k in (
                    "prec", "pcount", "logtic", "charge", "maxmz",
                    "peak_rep", "peak_iso", "peak_mz", "pair_rep", "pair_loss",
                )},
            }))

            internal_rows = self.corpora.internal_retrieval
            internal_labels = _label_index([
                internal_rows[int(i)]["peptide_charge"] for i in internal["row_ids"]
            ])
            metrics.update(self._retrieval_metrics(
                "retrieval", internal["embeddings"], internal_labels,
                self.last_transform, pairwise=True,
            ))
            baseline = Normalizer(norm="l2").transform(self.corpora.internal_binned)
            baseline_metrics = retrieval_metrics_tm(
                baseline, internal_labels, self.accelerator.device, ks=(5,)
            )
            metrics["retrieval/binned_mAP"] = baseline_metrics["mAP"]
            metrics["retrieval/gap_vs_binned"] = (
                metrics["retrieval/mAP"] - baseline_metrics["mAP"]
            )

            if external is not None:
                rows = self.corpora.external_retrieval
                labels = _label_index([
                    rows[int(i)]["peptide_charge"] for i in external["row_ids"]
                ])
                metrics.update(self._retrieval_metrics(
                    "replicate_retrieval", external["embeddings"], labels,
                    self.last_transform,
                ))
            metrics["diagnostics_runtime"] = time.monotonic() - started

        self.accelerator.wait_for_everyone()
        # Hugging Face's stock W&B callback rewrites ``eval_*`` to ``eval/*``.
        # Keeping that convention here gives us eval/probe/*,
        # eval/retrieval/*, etc. without a custom logging integration.
        return {f"eval_{key}": value for key, value in metrics.items()}

    def save_transform(self, path) -> None:
        if self.accelerator.is_main_process and self.last_transform is not None:
            joblib.dump(self.last_transform, path)
