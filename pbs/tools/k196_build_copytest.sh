#!/bin/bash
# K196 step 3: build the redundant copy (real data, same relative paths) + trap cases.
set -eu  # no pipefail: find|head pipelines end with SIGPIPE
S=/lus/flare/projects/UIC-HPC/khuss/msdelta; T=$S/k196-test
[[ -e $T ]] && { echo "$T exists"; exit 1; }
mkdir -p $T/runs $T/outside $T/diag $T/huggingface/datasets/parquet $T/huggingface/hub $T/data/p2-cap150-half
C="ionice -c3 nice -n 19"
R=$S/runs; TR=$T/runs
# D1 (full copies) + symlink traps
$C cp -a $R/sweep-attn_seed0-8863044 $TR/
$C cp -a $R/sweep-attn_seed2-8863044 $TR/
$C cp -a $R/sweep-attn_seed1-8863044 $T/outside/attn1
ln -s ../outside/attn1 $TR/sweep-attn_seed1-8863044                      # trap: symlinked run dir
ck=$(ls -d $TR/sweep-attn_seed2-8863044/checkpoint-* | head -1)
mv $ck $T/outside/ck_target; ln -s ../../outside/ck_target $ck            # trap: symlinked checkpoint
# Protected checkpoints in a D1 run (real final, checkpoints without the big weights)
$C rsync -a --exclude='checkpoint-*/**.safetensors' $R/sweep-cont050m_ep01_seed0-8860522 $TR/
$C rsync -a --exclude='checkpoint-*/**.safetensors' $R/sweep-200m_ck540k_seed1-8860472 $TR/   # protected whole run
# D2 (full copy) + hidden provenance dir
$C cp -a $R/sweep-s050m_ck050k_lr4e-4_p170k2_seed0-8880712 $TR/
$C cp -a $R/.configs-8880712 $TR/
# Live job run (trap): structure without weights
$C rsync -a --exclude='*.safetensors' $R/sweep-s050m_ck050k_lr4e-4_p170k2_cons_seed0-8901080 $TR/
# D4: a validate dir (full, has wandb symlink), k168 (no weights), the smaller real core dump
$C cp -a "$R/validate-sweep-gradcache-confirm-8855965.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov" $TR/
$C rsync -a --exclude='*.safetensors' $R/k168 $TR/
mkdir -p $T/diag/k119; $C cp -a $S/diag/k119/. $T/diag/k119/
mkdir -p $T/diag/k114; find $S/diag/k114 -maxdepth 1 -type f ! -name 'core.*' -exec cp -a {} $T/diag/k114/ \;
# Protected Pairformer dirs (no weights)
for d in pf-timing pf-baseline; do $C rsync -a --exclude='*.safetensors' $R/$d $TR/; done
mkdir -p $TR/p2; find $R/p2 -maxdepth 2 -type f -size -1M | head -20 | while read f; do mkdir -p $TR/p2/$(dirname ${f#$R/p2/}); cp -a "$f" $TR/p2/${f#$R/p2/}; done
# D3 subsets + never-touch neighbours
P=$S/huggingface/datasets/parquet
for d in default-a0d1b82e4b324277 default-2289e536a60ca07c; do
  mkdir -p $T/huggingface/datasets/parquet/$d
  (cd $P/$d && find . -type f | sort | head -4 | while read f; do mkdir -p $T/huggingface/datasets/parquet/$d/$(dirname $f); cp -a $f $T/huggingface/datasets/parquet/$d/$f; done)
done
mkdir -p $T/huggingface/datasets/chrisagrams___ms-denoise-100k; (cd $S/huggingface/datasets/chrisagrams___ms-denoise-100k && find . -type f -size -50M | head -5 | while read f; do mkdir -p $T/huggingface/datasets/chrisagrams___ms-denoise-100k/$(dirname $f); cp -a $f $T/huggingface/datasets/chrisagrams___ms-denoise-100k/$f; done)
$C cp -a $S/huggingface/hub/datasets--chrisagrams--ms2-peptide-replicate-retrieval $T/huggingface/hub/
D=$S/data/p2-cap150-half; TD=$T/data/p2-cap150-half
cp -a $D/README.txt $D/raw $TD/
mkdir -p $TD/preprocessed/train; cp -a $D/preprocessed/dataset_dict.json $TD/preprocessed/; cp -a $D/preprocessed/train/state.json $D/preprocessed/train/dataset_info.json $D/preprocessed/train/data-00000-of-00096.arrow $TD/preprocessed/train/
mkdir -p $TD/datasets; (cd $D/datasets && find . -type f | sort | head -3 | while read f; do mkdir -p $TD/datasets/$(dirname $f); cp -a $f $TD/datasets/$f; done)
# core.py trap
f=baselines/ms2rescore/lib/python3.11/site-packages/numpy/ma/core.py; mkdir -p $T/$(dirname $f); cp -a $S/$f $T/$f
echo built; du -sh $T
