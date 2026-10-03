"""K196: build the deletion lists (relative to $S) from a FRESH scan; see notes/K196_storage_plan.md.
Writes $S/k196-logs/{protect,D2,D3,D4,D1}.txt. Read-only: lists only, deletes nothing."""
import os, re, subprocess, sys
S = "/lus/flare/projects/UIC-HPC/khuss/msdelta"; REPO = os.path.expanduser("~/code/msdelta"); OUT = f"{S}/k196-logs"
JOB = re.compile(r"-(\d{7})$")
qs = subprocess.run(["qstat", "-u", os.environ["USER"]], capture_output=True, text=True, check=True).stdout
live = {l.split(".")[0] for l in qs.splitlines() if re.match(r"^\d+\.", l)}
runs = sorted(r for r in os.listdir(f"{S}/runs") if not r.startswith("."))  # .configs-<job>: provenance, kept

# Protect: every checkpoint the repo names, plus whole dirs that must stay.
grep = subprocess.run(
    ["grep", "-rhoE", r"[A-Za-z0-9_.+-]+-[0-9]{7}/checkpoint-[0-9]+", "--exclude-dir=.git", "--exclude-dir=.venv",
     "--exclude-dir=.venv-2026", "--exclude-dir=wandb", "--exclude-dir=logs", "--exclude-dir=.claude", "--exclude=.keys", "--exclude-dir=k196_storage", REPO],
    capture_output=True, text=True).stdout.split()
protect = {f"runs/{g}" for g in grep}
protect |= {f"runs/{r}" for r in runs if r.startswith("pf-") or r in ("p2", "p1-stage0", "allocfix")}
protect.add("runs/sweep-200m_ck540k_seed1-8860472")  # K110l: upload source in sweeps/upload_denoise_models.py
protect |= {f"runs/{r}" for r in runs if (m := JOB.search(r)) and m[1] in live}

def has_final(r):
    f = f"{S}/runs/{r}/final/model.safetensors"
    return os.path.isfile(f) and os.path.getsize(f) > 1_000_000

D2 = [f"runs/{r}" for r in runs if r.endswith("-8880712") and not os.path.exists(f"{S}/runs/{r}/final")]
D3 = ["huggingface/datasets/parquet/default-3ba6cfa2577938d3", "huggingface/datasets/parquet/default-a0d1b82e4b324277",
      "data/p2-cap150-half/datasets"]
D4 = [f"runs/{r}" for r in ("armcount", "armcount12", "gridexact", "p2-cost", "p2-smoke", "sibling2", "k168", "quarantine",
                            "pf-timing", "allocfix2", "ddp2")]
D4 += [f"runs/{r}" for r in runs if r.startswith("validate-sweep-")]
core = subprocess.run(["find", S, "-xdev", "-type", "f", "-name", "core.*", "-not", "-path", f"{S}/runs/validate-sweep-*"],
                      capture_output=True, text=True, check=True).stdout.split()
inside = tuple(f"{S}/{d}/" for d in D4)
COREDUMP = re.compile(r"/core\.x\d+c\d+s\d+b\d+n\d+\.\d+$")  # core.<aurora host>.<pid>; NOT core.py / core.h
def is_coredump(c):
    return bool(COREDUMP.search(c)) and "core file" in subprocess.run(["file", "-b", c], capture_output=True, text=True).stdout
D4 += sorted(os.path.relpath(c, S) for c in core if not c.startswith(inside) and is_coredump(c))
D1 = []
for r in runs:
    m = JOB.search(r)
    if not m or m[1] in live or not has_final(r) or f"runs/{r}" in protect:
        continue
    for c in sorted(os.listdir(f"{S}/runs/{r}")):
        p = f"runs/{r}/{c}"
        if re.fullmatch(r"checkpoint-\d+", c) and os.path.isdir(f"{S}/{p}") and not os.path.islink(f"{S}/{p}") and p not in protect:
            D1.append(p)
for name, items in (("protect", sorted(protect)), ("D1", D1), ("D2", D2), ("D3", D3), ("D4", D4)):
    with open(f"{OUT}/{name}.txt", "w") as f:
        f.write("".join(i + "\n" for i in items))
    print(name, len(items))
print("live", sorted(live))
