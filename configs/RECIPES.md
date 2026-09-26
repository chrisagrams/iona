# Fine-tuning recipes

The configurations behind the reported results, one arm per model. Paths are the Aurora
(ALCF) ones they ran with; replace `--pretrained_path` and the teacher/cache paths for
another system. Launch any arm with `pbs/aurora-finetune-sweep.pbs`, e.g.

    qsub -v SWEEP_ROOT=configs/sweep-conlong,ARMS=s400m_t0002_pk256_ep12_seed0,SWEEP_MODULE=msdelta.finetune_contrastive,SKIP_GRID_CHECK=1 pbs/aurora-finetune-sweep.pbs

(`SKIP_GRID_CHECK=1`: the grid generators are not part of this repository.)

## Denoising (`msdelta.finetune_denoise`)

Same recipe at every size: lr 2e-4, encoder learning rate 0.5x the head's, 4 epochs, head
width 512, effective batch 12 (one spectrum per tile, 12 tiles). The reported points vary
only the pretraining checkpoint in `--pretrained_path` and the seed. From scratch (`--random_init true`) is
the same except encoder learning rate 1.0x (nothing pretrained to preserve).

| model | pretrained (540k steps) | from scratch |
|---|---|---|
| 50M | `sweep-ckpt-denoise-ends/50m_ck540k_seed1` | `sweep-denoise-scratch/lr2e4_ep4_b12` |
| 100M | `sweep-ckpt-denoise-ends/100m_ck540k_seed1` | `sweep-denoise-scratch-scale/100m_ep04_seed1` |
| 200M | `sweep-ckpt-denoise-ends-big/200m_ck540k_seed1` | `sweep-denoise-scratch-scale/200m_ep04_seed1` |
| 400M | `sweep-ckpt-denoise-late400/400m_ck540k_seed1` | `sweep-denoise-scratch-scale/400m_ep04_seed1` |

## Contrastive spectrum embeddings (`msdelta.finetune_contrastive`)

Supervised contrastive loss (temperature 0.002, KL weight 10) with GradCache
(`--gradcache_chunk 4`), mean+max pooling, two stages:

| model | stage 1: replicate corpus | stage 2: ms-contrastive-100k (from stage 1) |
|---|---|---|
| 400M | `sweep-conlong/s400m_t0002_pk256_ep12_seed0` (12 epochs) | `sweep-con100k-best/cont400m_ep01_seed0` |
| 50M | `sweep-conlong/s050m_t0002_pk256_ep24_seed1` (24 epochs) | `sweep-con100k-best/cont050m_ep01_seed1` |

## Peptide embedder aligned to the spectrum encoder (`msdelta.finetune_align`)

Teacher embeddings are precomputed first (`pbs/precompute_align_sharded.pbs`).

| teacher | config |
|---|---|
| contrastive 400M (final) | `a1-align-100k-400m-c7final` |
| contrastive 50M | `a1-align-100k-050m-c7s600` |

## Evaluation and PSM rescoring

`pbs/eval_grouped_retrieval.pbs`, `pbs/eval_zeroshot_layers.pbs`, `pbs/eval_align_test.pbs`;
rescoring with `pbs/rerank_psm*.pbs` (see `RERANK.md`).
