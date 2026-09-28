# Pairformer port review (branch p1-pairformer: 6c03366, c15a28f, 1dff3b0), 2026-09-28

Read-only review with CPU probes (scratchpad pf_review/). No correctness bug found in masking,
label leakage through the pair stream, initialisation, gradient checkpointing, bf16 range or
config round trips. Findings and proposed ablations below; nothing here is decided (see
DECISIONS.md for what the user approves).

## Findings

| # | severity | where | issue | fix |
|---|---|---|---|---|
| 1 | likely bug (inherited from source) | pairformer.py mass-defect features | log-spaced non-integer frequencies on frac(Δm): defect −1 mDa (frac 0.999) and +1 mDa (0.001) look unrelated (L2 5.5 vs 0.36 for a 2 mDa step); losses just under an integer (CO, CO2, O, SO3, HPO3) are split from H2O/NH3/C2H4 | integer frequencies, or feed the signed defect Δ−round(Δ); needs an ablation |
| 2 | risk (docs wrong) | triangle attention chunking | chunking does not bound TRAINING memory (autograd keeps every chunk): ~3.2·B·N³·H·4 bytes per module (~5.5 GB at B32/N150/H4) | checkpoint each chunk, or correct docs; only matters with triangle attention on (off by default) |
| 3 | risk | architecture field | OLDER code loads a Pairformer checkpoint as a transformer with a warning (41 missing / 143 unexpected keys) -> mostly random model | fail on missing keys in fine-tune loaders; keep Pairformer checkpoints away from old-code evals |
| 4 | risk | pair_* defaults | test-sized defaults; source experiments used loss-bank sigma 10 ppm (default 20), mass-defect 32 frequencies, Fourier 64/0.01/1000 | the P1 card must set every pair_* field explicitly |
| 5 | divergence | token embedding | Pairformer tokens carry absolute m/z, transformer tokens do not -> any win is confounded; LayerNorms have learnable affine (source had none) | control arm pair_single_use_mz=False; document |
| 6 | divergence (undocumented) | METHODS | the source's "intrinsic" variant added charge-aware loss-dictionary / isotope features (z=1..3); the port has neither | document; ablation (code) |
| 7 | risk (BOTH architectures, all pretrained models) | processing_msdelta.py log_intensity normalisation | intensities are divided by the max over ALL peaks before masking; if the base peak is masked, no visible peak equals 1.0 -> tells the model a masked peak is the base peak (label information). Fair between arms, inflates absolute pretraining scores | normalise over visible peaks in the collator, or record as a caveat (not probed) |
| 8 | risk (memory) | triangle multiplication | ~12×z kept for backward: ~45 GB for 10 layers at B32/N150/C64 without checkpointing | gradient checkpointing is REQUIRED |
| 9 | nit | pair_bias_scale | cannot be set via config_overrides (default None rejected) | sentinel 0.0 |
| 10 | nit | fourier clamp | absolute m/z > 2000 clamped in token features | raise clamp or accept |
| 11 | nit (design) | gates / residual init | not AF-style (gate bias 0 not 1; only write-back zero-init) | ablation |
| 12 | info | git | branch is based on 14507a8; dev_finetune_02 moved on | rebase/merge before landing |

## Proposed ablations (ranked; cost vs one transformer debug run T at N=150)

1. pair update {static, transition, triangle} × write-back {off, on} + transformer control, 2 seeds (~25-30 T): is cubic mixing worth ~8× compute?
2. absolute m/z confound: Pairformer with/without m/z tokens; transformer + m/z tokens (small code change) (~3 T)
3. matched parameters vs matched compute (wall-clock / FLOPs) vs the transformer
4. pair input features: Fourier-only, leave-one-out (mass defect, loss dictionary, isotope, relative intensity), sigma 10 vs 20 ppm; later integer-frequency defect and charge-aware features (code)
5. peaks cap {64, 100, 150, 256} × architecture (triangle cost ∝ N³)
6. pair width c_z {16, 32, 64}, write-back {8, 16}
7. pair refinement in fewer layers / shared pair weights (code)
8. triangle attention on/off (only if triangle multiplication wins; add chunk checkpointing first)
9. symmetric vs directional pair representation (code)
10. AF-style init (code, low priority)
11. precursor features (code + data; leakage/double-counting caveats; always report MAP@R without the filter)
12. (user, 2026-09-28, K114-P) **Decoupled / parallel streams:** several single blocks per pair update, calibrated so K single blocks take about as long as one pair update, and run CONCURRENTLY (single stream uses the latest pair state z(m) while z(m+1) is computed), so the triangle ops are off the single stream's critical path. Related to #7 (pair updates in fewer layers) but adds the parallel execution. Open points: one-update staleness in both directions (bias read by the single stream, s read by the write-back); where parallelism comes from (two streams on one tile overlap only partially -- triangle ops are memory-bound at N=150; two tiles = model parallel with an s + bias (B x heads x N x N) exchange per round); K from measured step times (pair cost ~N³ vs single ~N²). Cheap first step: the RATIO alone (K single blocks per pair update, sequential), compute-matched (#3); build the parallel version only if a larger K does not hurt quality.

Fair evaluation: same data / cap / batch / steps (+ a wall-clock-matched transformer), ≥2 seeds;
pretraining ranked on a mask-ratio-independent score (port of the source's MaskedIntensityCallback,
small code change); downstream denoise AUROC/AUPRC (linear probe + fine-tune) and contrastive
MAP@R with/without filter; always report step time, peak memory, FLOPs.

## Update 2026-09-28: triangle-attention memory (K102, benchmark 8875808)
Finding #2 addressed on branch k102-triattn-memory: flags `pair_tri_attn_checkpoint_chunks` and
`pair_tri_attn_impl` ("naive" | "sdpa"), both off by default. B32 N150 fwd+bwd bf16 per module: naive
128 ms / 4.40 GB -> SDPA 57.5 ms / 2.85 GB -> SDPA+checkpointing 69.1 ms / 2.08 GB; XPU uses the fused
memory-efficient kernel. No custom kernel for now (K115-P / K116-P pending).

## Update 2026-09-28: bf16 check (K115, job 8875855) -> SDPA is the default
Relative L2 error vs an fp32 reference at B32 N150 (output / dz / worst param grad): naive bf16 autocast
5.1e-3 / 5.1e-3 / 6.4e-3; SDPA 4-D 5.1e-3 / 5.1e-3 / 6.6e-3 (0.47x the time, 0.65x the peak memory);
5-D math path 5.1e-3 / 5.1e-3 / 7.1e-3 (worse with peaked attention); all-bf16 dz ~20% worse. Every
autocast variant sits at the ~5e-3 bf16 floor of the projections; the kernel choice is below it.
`pair_tri_attn_impl` now defaults to "sdpa" (user K115). Checkpointing is numerically identical.

