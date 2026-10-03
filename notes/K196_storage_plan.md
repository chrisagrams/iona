# K196: storage reduction plan for UIC-HPC (2026-10-03)

**This is a plan only. Nothing has been deleted, moved or changed.**
- Each item needs the user's approval.
- Deletions follow the deletion protocol: notes/K110_checkpoint_inventory.md §4, using pbs/tools/k110_delete.sh.

## Situation

| Item | Value |
|---|---|
| Project UIC-HPC (all members) | **10.18 T used of a 10 T soft quota**, hard limit 11 T |
| Grace left (08:35 UTC Oct 3) | 6d22h, so it ends around **Oct 10 06:30 UTC** |
| `khuss/msdelta` | **4,898 GiB = 5.26 TB** (437k entries) |
| Live job 8901080 (K188 consensus all-checkpoint) | 622 GiB of checkpoints so far, still writing until about 14–15 UTC |

When the grace ends, the soft quota acts as a hard limit, and every write in the project fails.

Source: one read-only `find -printf` walk at 08:40 UTC (nice/ionice).
- Per-run table: results/raw/diag/k196_storage/runs_2026-10-03.tsv
- Top-level summary: summary_2026-10-03.txt
- D1 candidate list: D1_candidate_checkpoints.tsv

### Where the space is

| Area | GiB | Notes |
|---|---:|---|
| runs/ intermediate `checkpoint-*` | 2,870 | Weights only: K110a-lite removed the optimizer states. Each checkpoint holds model.safetensors (2/3) and encoder/model.safetensors (1/3). |
| runs/ `final/` | 896 | 2,224 runs |
| runs/ other | 113 | Logs, diag dirs |
| huggingface/ | 448 | Hub raw: MSConsensus-100M 178, psm-rerank-hek-hct116 66, massive_kb_v1_shuffled 51. Arrow caches: massive_kb 138, ms-contrastive-100k 9 |
| data/ | 210 | p2-cap150-half 156 (datasets/ cache 128 + preprocessed 29), massive-kb-contrastive 45, probe-cap512 5, stage0-cap150 4 |
| baselines/ | 209 | ms2rescore_out 165 |
| pretrained/ | 44 | Chris's checkpoints, used by the all-checkpoint scaling |
| rerank-psm/ + align-targets-* | 93 | |
| everything else | ~15 | |

## Phase 1: delete, no tape needed (fixes the quota by itself)

| ID | What | GiB | Why it is safe |
|---|---|---:|---|
| **K196-D1** | Intermediate `checkpoint-*` of finished runs (job-ID runs with `final/model.safetensors`, not live, checkpoint not cited anywhere in the repo). The 78 GiB of cited checkpoints (8860522 300/600/900, 8872141, 8851663) are kept. | **1,686** | Only finals are scored. This is the "full K110a" the user described as "we would only need the models". The largest jobs: 8878459 K66 400m 371, 8880712 K155 finished arms 357, 8875261 191, 8882196 K163 182, 8875262 95. |
| K196-D1-lite (alternative) | Same checkpoints, but keep `encoder/` | 1,143 (keeps 543) | Keeps snapshot evaluation possible. Encoder snapshots could go to tape instead (K196-H5). |
| **K196-D2** | Paused no-consensus K155 remainder: 48 run dirs of 8880712 with no final, only partial checkpoints | **428** | K188-C (user 2026-10-01): the K155 remainder is not resumed. |
| **K196-D3** | Regenerable caches. massive_kb arrow cache `huggingface/datasets/parquet/default-3ba6…` (133) and `default-a0d1…` (4.7), from Sept 11; `data/p2-cap150-half/datasets/` (128); `baselines/.uvcache` (7.4) | **273** | Training reads `p2-cap150-half/preprocessed/` (train.py `load_pretraining_datasets_from_disk`), not the cache. Each cache is rebuilt from the hub parquet on demand. Verify first that preprocessed/ loads with the cache renamed. |
| **K196-D4** | Diag and validation leftovers: runs/armcount 18, armcount12 9, gridexact 8, p2-cost 9, p2-smoke 7, sibling2 3, k168 3, quarantine 3, pf-timing 2, allocfix2 2, ddp2 1, `validate-sweep-*` (35 dirs) 69, plus 189 core dumps (3.2, mostly diag/k114) | **132** | The outputs of pbs/diag/* scripts. Their results are in notes/OBSERVATIONS and results/raw/diag. `runs/allocfix` stays (K110 rule). |
| **K196-D5** (later) | Intermediate checkpoints of K188 8901080, after its six scoring jobs finish and the results are committed | ~650–800 | Same rule as D1 |

| Scenario | Frees | Project after |
|---|---|---|
| D1 + D2 + D3 + D4 | 2,519 GiB = **2.70 TB** | ~7.5 TB (khuss/msdelta ~2.6 TB) |
| plus D5 | another ~0.7–0.85 TB | |
| D1-lite instead of D1 | 0.58 TB less | |

## Phase 2: tape (HPSS), for things we may need later but not now

**Prerequisite (user action):** HPSS needs a keytab at `~/.hpss/.ktb_khuss`. None exists. Per the ALCF docs, access is requested from ALCF support. That is an outward email, so it is the user's call. I can draft it.

| ID | What | GiB | Notes |
|---|---|---:|---|
| **K196-H1** | `final/` of old runs, jobs < 8860000: Sept 10–24 HP, denoise and old-recipe sweeps; 1,765 runs not listed in any arm file | **701** | One htar per job, each member ≤ 1.5 GiB. The 39 arm-listed old finals (26 GiB) stay on disk. |
| **K196-H2** | baselines/ms2rescore_out | 165 | MS2Rescore intermediates (C11/R3). The scored results are already in the repo. |
| **K196-H3** | rerank-psm/ (41) and align-targets-* (52) | 93 | Alignment and rerank artifacts. The alignment caveats are still open (PLAN.md), so tape, not delete. |
| **K196-H4** | data/massive-kb-contrastive (+ dryrun, exclusion) | 45 | C18 (MassIVE-KB training) is deferred to camera-ready. |
| K196-H5 (optional) | encoder/ snapshots, if D1-lite is chosen | 543 | |
| K196-H6 (user's call) | HF hub raw copies of massive_kb_v1_shuffled (51) and psm-rerank-hek-hct116 (66) | 117 | Both can be re-downloaded at the pinned revisions, so delete, tape or keep. Tape is the safe choice if the upstream repos could change. |

Total for H1–H4: about 1.0 TiB. After both phases, khuss/msdelta is about 1.6 TB, plus whatever stays from D5/H6.

### How the transfer would run

- Use htar, which streams straight to HPSS, so no local tar and no extra Lustre space. Write one archive per job or group: `htar -cvf /home/khuss/msdelta/<group>.tar <paths>`.
  - htar limits: 64 GB per member file; our largest file is 3.3 GiB, so this is fine. Path length is a problem: htar limits member path length, and some of our relative paths reach 279 characters. Run htar from inside each job's parent dir so members get short relative names, and put any path that is still too long through `hsi put`.
  - Run it from a login node under nice/ionice. The docs put hsi and htar on the login nodes. Alternative: Globus `alcf#dtn_hpss`.
- **Verify before deleting the disk copy:**
  1. compare `htar -tvf` against the local file list (names and sizes);
  2. extract one member with `htar -xf` into test-scratch and compare checksums.
- Removing the disk copy afterwards is a deletion, so it follows the full protocol.
- Keep an index in the repo: notes/K196_tape_index.md, listing what is in which archive.

## Keep on disk

- pretrained/ (44)
- huggingface MSConsensus-100M (178): the source for the K187 cap-512 build
- ms-contrastive-100k cache (9): used by the live K188 runs
- data/p2-cap150-half/preprocessed (29), probe-cap512, stage0-cap150
- eval-data, embeddings
- finals of jobs ≥ 8860000 (~150): the current recipe, K66/K155/K163/C27/C7 encoders
- arm-listed finals
- runs/p2 (37): the K180–K186 Pairformer runs
- pf-* (14): the user's early Pairformer runs
- everything of live job 8901080 until it is scored

## Order of work (after approval)

1. **Today:**
   - D3 and D4 first: small and quick, and they test the tooling.
   - Then D2, then D1 (staged 1 → 10 → all).
   - Each step: rebuild the list live (fresh qstat, find and repo grep), dry run, test on a redundant copy, then staged live.
   - Log everything in $S/k196-logs/ and DECISIONS.
2. **Risk until then:** the live K188 job adds checkpoints. If the project gets near the **11 T hard limit**, its writes fail. Watch `lfs quota` until phase 1 frees space.
3. **After K188 scoring:** D5.
4. **When the keytab arrives:** H1–H4 (and H5/H6 if chosen), verify, then remove the disk copies under the protocol.
