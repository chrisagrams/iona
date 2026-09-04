# `msdelta/train/` — running a pretraining job

The top of the dependency graph: this package imports `model/`, `data/`, and `eval/`, and
**nothing imports it back**.

| file | contents |
| --- | --- |
| `cli.py` | `main()` and `MSDeltaTrainer`; the whole run assembly |
| `args.py` | `ModelArguments`, `DataArguments`, `MSDeltaTrainingArguments` |
| `callbacks.py` | `_InlineCallback` base + six diagnostic callbacks + `build_callbacks()` |
| `wandb_distributed.py` | `init_wandb_run` — one client per node, one shared run |
| `__main__.py` | lets `python -m msdelta.train` work (the Polaris launcher uses it) |

## Entry points

```bash
msdelta-train --args_file configs/msdelta-base-50m/training.args    # console script
python -m msdelta.train --args_file …                               # what PBS launches
```

Both land on `msdelta.train.cli:main`.

`__init__.py` is deliberately **import-light** — it does not pull in `cli`. That keeps
`from msdelta.train.args import DataArguments` cheap instead of dragging in matplotlib, wandb,
and the whole `msdelta.eval` package. This is why the console script points at
`msdelta.train.cli:main` rather than `msdelta.train:main`.

## What `main()` does, in order

1. Parse the three arg dataclasses (`--args_file` supported).
2. Load `MSDeltaConfig`, apply `--config_overrides`, then **re-validate** (`update_from_string`
   does not validate on its own).
3. Build the processor, honouring `DataArguments` overrides over `preprocessor_config.json`.
4. Init W&B (one client per node) with the fully resolved config.
5. Preprocess datasets under `main_process_first` so rank 0 populates the shared cache.
6. Slice the eval set to `validation_batches × per_device_eval_batch_size`.
7. Optionally build the denoising datasets with a **second** processor (1024 peaks, no threshold).
8. Attach callbacks via `build_callbacks`, train, and on rank 0 save `final/` plus the last bias
   panels.

## One subtlety worth knowing

```python
class MSDeltaTrainer(Trainer):
    def get_decay_parameter_names(self, model):
        return [n for n in super().get_decay_parameter_names(model) if not n.endswith(".freqs")]
```

**Learned Fourier frequencies are excluded from weight decay.** If you rename that parameter in
`model/fourier.py`, you silently start decaying your frequencies toward zero — and
`FourierProbeCallback`'s `fourier/*_drift_log10` is where you would notice, eventually.

## Callbacks

All are interval-driven from `MSDeltaTrainingArguments`; see
[../../configs/README.md](../../configs/README.md) for the flags. Five run on rank 0 only:

| callback | flag | what it answers |
| --- | --- | --- |
| `BiasPanelCallback` | `--bias_curve_steps` | what do the per-head Δm/z curves look like? |
| `AlignmentCallback` | `--probe_steps` | do bias peaks land on real chemistry above chance? |
| `LinearProbeCallback` | `--probe_steps` | what is linearly decodable from the frozen encoder? |
| `FourierProbeCallback` | `--probe_steps` | are the learned frequencies alive and useful? |
| `RetrievalCallback` | `--probe_steps` | do replicate spectra embed together, vs. a binned baseline? |
| `ReplicateRetrievalCallback` | `+ --replicate_retrieval_repo` | same, on an external benchmark |

`DenoisingProbeCallback` (`--denoise_steps`) is the exception: it is **not** an `_InlineCallback`
and runs on **all** ranks, because it spins up a nested distributed `Trainer`. It guards against
double-firing with `last_step`, logs only on rank 0, and ends with a `torch.distributed.barrier()`.
The probe itself forks and restores RNG and `requires_grad` state, so it cannot perturb the outer
run — see [`../eval/README.md`](../eval/README.md).

## Distributed

`wandb_distributed.init_wandb_run` returns `None` for `LOCAL_RANK != 0`, so there is one W&B
client per **node**. Multi-node runs use `mode="shared"` with the hostname as the label and
**require a preset `WANDB_RUN_ID`** — it raises if it is missing, because the alternative is
silently splitting one job across several runs. Only rank 0 writes the config and the finish
state.

`pbs/polaris-pretrain.pbs` derives `gradient_accumulation_steps` from `GLOBAL_BATCH_SIZE` and the
detected GPU count, and exports the MPI→torch.distributed rank variables (`PMI_RANK` → `RANK`,
etc.).
