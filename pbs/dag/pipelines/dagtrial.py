"""K84-I: the first live trial of the DAG scheduler (card I-DAG-1, approved 2026-09-28).

    smoke  one 25m arm of configs/sweep-hp-scale, MAX_STEPS=20, debug, 1 node, 60 min
           (only the 25m arm: the 400m smoke arm took 52 min in 8875194, too close to 60)
      -> score  eval_grouped_retrieval.pbs on two already-scored 50m models, validation
                split, debug / debug-scaling (~30 min); must match finals0927-validation

Every job gets SCRATCH_ROOT=<trial dir>, so its runs, code snapshots, notifications and
manifests stay out of the main scratch; HF_HOME and DATA point at the existing caches.
Run with DAG_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/dag-trial/home, whose
config.json sets scratch_root to the trial dir and is the only place dry_run is off.
"""

from dag.spec import Node, Pipeline, Sweep

CARD = "K84-I"
MAIN = "/lus/flare/projects/UIC-HPC/khuss/msdelta"
COMMON = {"SCRATCH_ROOT": "{scratch}", "HF_HOME": f"{MAIN}/huggingface"}

PIPELINE = Pipeline(name="dagtrial", budget_node_hours=3, card=CARD, nodes=[
    Node(id="smoke",
         sweep=Sweep("configs/sweep-hp-scale", "sweeps/arms/dagtrial_smoke.txt"),
         vars=dict(COMMON, MAX_STEPS="20"), nodes=1, runtime_min=45, queues=["debug"],
         approved=CARD, max_resumes=0,
         outputs=["{scratch}/runs/sweep-s025m_ck540k_lr4e-4_p128k2_seed0-{job}"]),
    Node(id="score", deps=["smoke"], script="pbs/eval_grouped_retrieval.pbs",
         vars=dict(COMMON, MODELS="sweeps/arms/dagtrial_models.txt", SPLIT="validation",
                   DATA=f"{MAIN}/eval-data/ms-contrastive-100k-validation-mp512",
                   OUT_DIR="{scratch}/eval/contrastive/dagtrial-validation"),   # $MSDELTA_EVAL (configs/homes.env)
         nodes=1, runtime_min=23, queues=["debug", "debug-scaling"], approved=CARD,
         max_retries=0,
         outputs=["{scratch}/eval/contrastive/dagtrial-validation/c8c19_mass_seed0.json",
                  "{scratch}/eval/contrastive/dagtrial-validation/c8c19_mass_seed1.json"]),
])
