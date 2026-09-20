import re, sys, collections
log = open(sys.argv[1]).read().splitlines()
runs, cur = collections.OrderedDict(), None
for line in log:
    m = re.match(r'^#+ (\S+) step (\S+)', line)
    if m:
        cur = (m.group(1), m.group(2)); runs[cur] = {}
        continue
    m = re.match(r'^\s{2}(mean|layer\d+/mean)\s+([\d.]+)', line)
    if m and cur:
        runs[cur][m.group(1)] = float(m.group(2))

order = ["50m", "100m", "200m"]
steps = ["10000", "50000", "100000", "150000", "LAST"]
last = {s: [k[1] for k in runs if k[0] == s][-1] if any(k[0]==s for k in runs) else None
        for s in order}

print("OUTPUT-LAYER ratio (`mean` pooling), matched pretraining step")
print(f"  {'step':>8} " + "".join(f"{s:>9}" for s in order))
for st in steps:
    row = f"  {st:>8} "
    for s in order:
        key = (s, last[s] if st == "LAST" else st)
        row += f"{runs.get(key, {}).get('mean', float('nan')):>9.2f}"
    print(row + ("   <- each model's own frontier" if st == "LAST" else ""))
print(f"  {'(step)':>8} " + "".join(f"{last[s] or '-':>9}" for s in order))

print("\nBEST SINGLE LAYER at each point")
print(f"  {'step':>8} " + "".join(f"{s:>14}" for s in order))
for st in steps:
    row = f"  {st:>8} "
    for s in order:
        d = runs.get((s, last[s] if st == "LAST" else st), {})
        ls = {k: v for k, v in d.items() if k.startswith("layer")}
        if ls:
            b = max(ls, key=ls.get)
            row += f"{ls[b]:>9.2f} {b[5:7]:>4}"
        else:
            row += f"{'-':>14}"
    print(row)

vals = {s: [runs[k]['mean'] for k in runs if k[0] == s and 'mean' in runs[k]] for s in order}
print("\nWITHIN-MODEL spread over its own trajectory (the resolution of any comparison)")
for s in order:
    v = [x for x in vals[s] if x == x]
    if v:
        print(f"  {s:<5} min {min(v):.2f}  max {max(v):.2f}  spread {max(v)-min(v):.2f}")
post = [x for s in order for x in vals[s][1:]]
print(f"  excluding step 10,000: all points lie in "
      f"[{min(post):.2f}, {max(post):.2f}], spread {max(post)-min(post):.2f}")
