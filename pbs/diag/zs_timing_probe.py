"""Diagnostic: time every per-layer step of eval_zeroshot_layers (ABTT fit, retrieval scoring) and log
the device of its tensors, without changing the module. Same CLI as msdelta.eval_zeroshot_layers."""
import sys, time
import torch
import msdelta.contrastive as C
import msdelta.eval_zeroshot_layers as Z

_score, _fit = C.retrieval_metrics_topk, Z.fit_abtt
n = {"score": 0, "fit": 0}

def score(emb, groups, *a, **k):
    t = time.time(); out = _score(emb, groups, *a, **k)
    if hasattr(torch, "xpu") and torch.xpu.is_available(): torch.xpu.synchronize()
    n["score"] += 1
    if n["score"] <= 12 or n["score"] % 40 == 0:
        print(f"[probe] score #{n['score']}: emb {tuple(emb.shape)} {emb.dtype} on {emb.device}, "
              f"device arg {k.get('device')}, {time.time() - t:.2f}s", flush=True)
    return out

def fit(x, m):
    t = time.time(); out = _fit(x, m); n["fit"] += 1
    if n["fit"] <= 6 or n["fit"] % 20 == 0:
        print(f"[probe] abtt fit #{n['fit']}: {tuple(x.shape)} on {x.device}, {time.time() - t:.2f}s "
              f"(torch threads {torch.get_num_threads()})", flush=True)
    return out

C.retrieval_metrics_topk = score
Z.fit_abtt = fit
print(f"[probe] xpu available {torch.xpu.is_available()}, devices {torch.xpu.device_count() if torch.xpu.is_available() else 0}, "
      f"threads {torch.get_num_threads()}", flush=True)
sys.argv = ["eval_zeroshot_layers"] + sys.argv[1:]
raise SystemExit(Z.main())
