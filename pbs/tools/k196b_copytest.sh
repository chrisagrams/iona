#!/bin/bash
# K196b deletion protocol step 3: run pbs/tools/k196_delete.sh scoredft for real on a REDUNDANT COPY with trap cases,
# and verify that only the intended paths vanished and everything else is byte-identical.
#   bash pbs/tools/k196b_copytest.sh
set -u
S=/lus/flare/projects/UIC-HPC/khuss/msdelta; R=$S/runs; C=$S/k196b-copytest; L=$S/k196-logs/k196b
REPO=$(cd "$(dirname "$0")/../.." && pwd)
[[ -e $C ]] && { echo "copy dir $C exists -- remove it first"; exit 2; }
mkdir -p "$C/runs" "$C/scored"
T1=sweep-s050m_ck010k_lr4e-4_p170k2_cons_seed0-8901080; T2=sweep-s050m_ck010k_lr4e-4_p170k2_cons_seed1-8901080
cp -a "$R/$T1" "$R/$T2" "$C/runs/"                                      # real targets, full copies
skel() { rsync -a --exclude '*.safetensors' --exclude 'optimizer.pt' "$R/$1/" "$C/runs/$2/"; }
skel sweep-s050m_ck540k_lr4e-4_p170k2_cons_seed0-8882196 sweep-s050m_ck540k_lr4e-4_p170k2_cons_seed0-8882196  # 540k twin
skel sweep-s050m_ck010k_lr4e-4_p170k2_cons_seed0-8884292 sweep-s050m_ck010k_lr4e-4_p170k2_cons_seed0-8884292  # smoke run, scores name another run
skel sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed0-8901080 sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed0-8879772  # named after a queued job
skel sweep-s050m_ck050k_lr4e-4_p170k2_cons_seed0-8901080 sweep-s050m_ck050k_lr4e-4_p170k2_cons_seed0-8901080  # yeast score removed
skel sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed1-8901080 sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed1-8901080  # protected
skel sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed2-8901080 sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed2-8901080  # not listed
ln -s "$C/runs/$T1" "$C/runs/sweep-s100m_ck010k_lr4e-4_p170k2_cons_seed0-8901080"                               # symlink
mkdir -p "$C/pretrained/x"; echo keep > "$C/pretrained/x/f"
for s in validation test oodval mouse human yeast; do cp -a "$REPO/results/raw/finetune/contrastive/cons-allck-$s" "$C/scored/"; done
rm "$C/scored/cons-allck-yeast/s050m_ck050k_lr4e-4_p170k2_cons_seed0.json"
cat > "$C/list.txt" <<LIST
runs/$T1
runs/$T2
runs/sweep-s050m_ck540k_lr4e-4_p170k2_cons_seed0-8882196
runs/sweep-s050m_ck010k_lr4e-4_p170k2_cons_seed0-8884292
runs/sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed0-8879772
runs/sweep-s050m_ck050k_lr4e-4_p170k2_cons_seed0-8901080
runs/sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed1-8901080
runs/sweep-s100m_ck010k_lr4e-4_p170k2_cons_seed0-8901080
runs/sweep-s400m_ck010k_lr4e-4_p170k2_cons_seed0-8901080
runs/../pretrained
pretrained/x
LIST
echo "runs/sweep-s050m_ck120k_lr4e-4_p170k2_cons_seed1-8901080" > "$C/protect.txt"
man() { (cd "$C" && find . -path ./scored -prune -o \( -type f -o -type l \) -print0 | sort -z | xargs -0 -r md5sum 2>/dev/null; find . -path ./scored -prune -o -type l -printf '%p -> %l\n') ; }
man > "$C.before"
K196_ROOT=$C K196_PROTECT=$C/protect.txt K196_SCORED=$C/scored bash "$REPO/pbs/tools/k196_delete.sh" scoredft "$C/list.txt" --live | tee "$L/copytest.txt"
man > "$C.after"
echo "--- vanished (expect only $T1 and $T2):"; comm -23 <(sort "$C.before") <(sort "$C.after") | awk '{print $2}' | cut -d/ -f2-3 | sort -u
echo "--- appeared or changed (expect none):"; comm -13 <(sort "$C.before") <(sort "$C.after")
