# `msdelta/` — package layout

Four subpackages, split by role. The dependency direction is strictly one-way:

```
                     ┌──────────────┐
                     │  data/       │  spectra in, tensors out
                     │  chemistry   │  + the reference mass tables
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │  model/      │  the architecture (depends on nothing internal)
                     └──────┬───────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
          ┌──────▼───────┐      ┌──────▼───────┐
          │  eval/       │◄─────┤  train/      │  train/ orchestrates eval/
          │  diagnostics │      │  the run     │
          └──────────────┘      └──────────────┘
```

`model/` imports nothing from the other three. `data/` imports only `data/chemistry`.
`eval/` imports `model/` and `data/chemistry`. `train/` sits on top and imports all three.
**Nothing imports `train/`.** If you find yourself adding an import that points backwards up
this diagram, something belongs in a different package.

| package | contents | README |
| --- | --- | --- |
| [`model/`](model/) | `configuration.py`, `modeling.py`, `fourier.py`, `experimental.py` | [model/README.md](model/README.md) |
| [`data/`](data/) | `processing.py`, `loading.py`, `chemistry.py` | [data/README.md](data/README.md) |
| [`train/`](train/) | `cli.py`, `args.py`, `callbacks.py`, `wandb_distributed.py` | [train/README.md](train/README.md) |
| [`eval/`](eval/) | `probe.py`, `alignment.py`, `retrieval.py`, `denoising.py`, `embedding.py`, `viz.py` | [eval/README.md](eval/README.md) |

## Import conventions

* **Absolute imports everywhere** (`ruff` rule `TID252` bans relative ones) — with one
  deliberate exception below.
* **`model/modeling.py` and `model/experimental.py` use relative imports on purpose.** Hugging
  Face's `custom_object_save` walks relative imports to copy dependent modules into a checkpoint
  directory, which is what makes a saved model loadable with `trust_remote_code=True` on a
  machine that does not have `msdelta` installed. `pyproject.toml` carries a `TID252` per-file
  ignore for exactly these two files, with a comment saying why. **Do not "fix" them.**
* Each subpackage's `__init__.py` re-exports its public names, so
  `from msdelta.eval import run_all_probes` works as well as the fully-qualified path.
  `msdelta/train/__init__.py` is the exception: it stays import-light so that
  `from msdelta.train.args import DataArguments` does not drag in matplotlib, wandb, and all of
  `msdelta.eval`.
* `msdelta/__init__.py` exports the ten public model/processor symbols and performs the Hugging
  Face auto-class registration. Its surface is unchanged by the restructure — `from msdelta
  import MSDeltaForPreTraining` still works.

## Entry points

```bash
msdelta-train --args_file configs/…/training.args      # → msdelta.train.cli:main
python -m msdelta.train --args_file …                  # → msdelta/train/__main__.py
```

Both resolve to the same `main()`. The `-m` form is what `pbs/polaris-pretrain.pbs` launches
under MPI.

## Where things live now

The package was flat until this restructure. Old → new:

| was | is now |
| --- | --- |
| `msdelta/configuration_msdelta.py` | `msdelta/model/configuration.py` |
| `msdelta/modeling_msdelta.py` | `msdelta/model/modeling.py` |
| `msdelta/modeling_msdelta_new.py` | `msdelta/model/experimental.py` |
| `msdelta/fourier.py` | `msdelta/model/fourier.py` |
| `msdelta/processing_msdelta.py` | `msdelta/data/processing.py` |
| `msdelta/data.py` | `msdelta/data/loading.py` |
| `msdelta/chemistry.py` | `msdelta/data/chemistry.py` |
| `msdelta/train.py` | `msdelta/train/cli.py` |
| `msdelta/training_args.py` | `msdelta/train/args.py` |
| `msdelta/callbacks.py` | `msdelta/train/callbacks.py` |
| `msdelta/wandb_distributed.py` | `msdelta/train/wandb_distributed.py` |
| `msdelta/{probe,alignment,retrieval,denoising,embedding,viz}.py` | `msdelta/eval/…` (unchanged names) |

The `_msdelta` suffixes were dropped — `msdelta.model.modeling` reads better than
`msdelta.model.modeling_msdelta`. `data.py` became `data/loading.py` rather than
`data/datasets.py` to avoid visually shadowing the `datasets` library it imports.

> **`.gitignore` footgun, now fixed.** The ignore rule for benchmark data was an unanchored
> `data/`, which silently matched the new `msdelta/data/` package — new files there would never
> have been committed. It is now `/data/`, anchored to the repo root. Keep root-level ignore
> patterns anchored.

**Old checkpoints are unaffected** (weights are keyed by parameter name, not module path), but any
notebook or script doing `from msdelta.modeling_msdelta import …` needs updating to
`from msdelta.model.modeling import …` — or just `from msdelta import …`, which never changed.
