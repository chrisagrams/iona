#!/bin/bash
# K196 D1-lite step 3: redundant copy (real data, same relative paths) + traps.
set -eu
S=/lus/flare/projects/UIC-HPC/khuss/msdelta; T=$S/k196-test; R=$S/runs; TR=$T/runs; C="ionice -c3 nice -n 19"
[[ -e $T ]] && { echo "$T exists"; exit 1; }
mkdir -p $TR $T/outside
# real D1-lite runs
$C cp -a $R/sweep-s050m_ck220k_lr1e4_kl100_t003_seed0-8856116 $TR/
$C cp -a $R/sweep-s050m_ck220k_lr1e4_kl100_t003_seed1-8856116 $TR/
$C cp -a $R/sweep-s050m_ck220k_lr1e4_kl100_t003_seed2-8856116 $TR/
# trap: denoise checkpoint without encoder/ (real)
$C rsync -a --exclude='checkpoint-*/' $R/denoise-ft-100m-8839881 $TR/; $C cp -a $R/denoise-ft-100m-8839881/checkpoint-1500 $TR/denoise-ft-100m-8839881/
# trap: protected checkpoint-300 (real) in a run with a real final
mkdir -p $TR/sweep-cont050m_ep01_seed0-8860522; $C cp -a $R/sweep-cont050m_ep01_seed0-8860522/{final,checkpoint-300,checkpoint-1062} $TR/sweep-cont050m_ep01_seed0-8860522/
# trap: live run that looks finished (fake weights)
r=$TR/sweep-s100m_ck010k_lr4e-4_p170k2_cons_seed0-8901080; mkdir -p $r/final $r/checkpoint-265/encoder
for f in $r/final/model.safetensors $r/checkpoint-265/model.safetensors $r/checkpoint-265/encoder/model.safetensors; do head -c 2000000 /dev/urandom > $f; done
# trap: run without final (fake weights)
r=$TR/sweep-fake_nofinal-8800001; mkdir -p $r/checkpoint-10/encoder; head -c 2000000 /dev/urandom > $r/checkpoint-10/model.safetensors; head -c 2000000 /dev/urandom > $r/checkpoint-10/encoder/model.safetensors
# traps: symlinked weights / symlinked encoder copy / symlinked run (built from seed2's real files)
r=$TR/sweep-s050m_ck220k_lr1e4_kl100_t003_seed2-8856116; ck=$(ls -d $r/checkpoint-* | head -1)
mkdir -p $T/outside/w; mv $ck/model.safetensors $T/outside/w/model.safetensors; ln -s ../../../outside/w/model.safetensors $ck/model.safetensors
r2=$TR/sweep-fake_symenc-8800002; mkdir -p $r2/final $r2/checkpoint-10/encoder; head -c 2000000 /dev/urandom > $r2/final/model.safetensors; head -c 2000000 /dev/urandom > $r2/checkpoint-10/model.safetensors
head -c 2000000 /dev/urandom > $T/outside/enc.safetensors; ln -s ../../../../outside/enc.safetensors $r2/checkpoint-10/encoder/model.safetensors
$C cp -a $R/sweep-s050m_ck220k_lr1e4_kl100_t003_seed1-8856116 $T/outside/runcopy; ln -s ../outside/runcopy $TR/sweep-fake_symrun-8800003
echo built; du -sh $T
