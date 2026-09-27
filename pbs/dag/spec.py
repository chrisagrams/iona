"""Pipeline specs: what a DAG node is, and how pipelines are loaded and validated.

Specs are PYTHON, not YAML (decision recorded in pbs/dag/README.md):
  * real pipelines are generated -- K66-C is 4 scales x 5 scoring sets -- and a loop in
    Python is clearer than a YAML templating dialect;
  * some nodes need a small login-node step before submission (write the models file a
    scoring job reads from the training job's run directories); that is a Python callable;
  * the scheduler must run on the login node's stock interpreter without extra packages,
    and PyYAML is only in the venv. Everything here is the standard library.

A spec module under pbs/dag/pipelines/ defines ``PIPELINE = Pipeline(...)``. All modules
are loaded into ONE graph (one DAG for all jobs); node keys are ``<pipeline>/<node>``.

Templating in vars / outputs (resolved at submission time):
  {job}            this node's OUTPUT job id (the first job of a RESUME_JOB chain)
  {job:<node>}     a dependency's output job id (same pipeline, or ``<pipeline>/<node>``)
  {scratch}        SCRATCH_ROOT, {repo} the checkout, {pipeline}, {node}
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

SWEEP_SCRIPT = "pbs/aurora-finetune-sweep.pbs"
SECRET_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSW)", re.I)
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass
class Sweep:
    """A sweep node: aurora-finetune-sweep.pbs over the arms listed in ARMS_FILE."""
    root: str                      # SWEEP_ROOT, e.g. configs/sweep-hp-scale
    arms_file: str                 # ARMS_FILE, e.g. sweeps/arms/hp_scale_400m.txt
    module: str = "msdelta.finetune_contrastive"

    def arms(self, repo: Path) -> list[str]:
        path = repo / self.arms_file
        if not path.exists():
            return []
        return [a for a in re.split(r"[\s,]+", path.read_text()) if a]


@dataclass
class Calibration:
    """How to estimate an unknown runtime: a short debug run of the SAME command.

    estimate = overhead + (calibration runtime - overhead) * full_steps / steps.
    The calibration is its own job and needs its own approval (card ID).
    """
    vars: dict = field(default_factory=lambda: {"MAX_STEPS": "50"})
    steps: int = 50
    full_steps: int = 1000
    overhead_min: float = 8.0
    approved: Optional[str] = None


@dataclass
class Prepare:
    """A login-node step run just before submission (live mode only; plan describes it)."""
    describe: str
    run: Callable[[dict], list]    # ctx -> list of files written


@dataclass
class Node:
    id: str
    script: Optional[str] = None           # a pbs/*.pbs script, or ...
    sweep: Optional[Sweep] = None          # ... a sweep (script = aurora-finetune-sweep.pbs)
    vars: dict = field(default_factory=dict)
    deps: list = field(default_factory=list)
    nodes: int = 1
    runtime_min: Optional[float] = None    # expected runtime; None = unknown
    kind: Optional[str] = None             # history key; derived when omitted
    outputs: list = field(default_factory=list)   # declared outputs (globs, templated)
    queues: list = field(default_factory=lambda: ["debug", "debug-scaling", "capacity"])
    approved: Optional[str] = None         # experiment card ID; None = never submitted
    calibration: Optional[Calibration] = None
    prepare: Optional[Prepare] = None
    submit_with_parent: bool = False       # explicit smoke->run pair (afterok)
    rerun_on: tuple = ()                   # non-sweep: statuses that allow a plain rerun
    max_resumes: int = 2                   # sweeps: RESUME_JOB rounds after partial/walltime
    max_retries: int = 1                   # reruns / hung resubmissions
    stall_minutes: Optional[float] = None  # hung if progress files stop growing this long
    progress: list = field(default_factory=list)  # globs whose mtime shows progress
    adopt_job: Optional[str] = None        # a job submitted by hand, tracked not submitted
    legacy_log_success: bool = False       # adopted job without job_finish.sh: '=== done: N/N'
    note: str = ""

    # filled by load()
    pipeline: str = ""

    @property
    def key(self) -> str:
        return f"{self.pipeline}/{self.id}"

    @property
    def script_path(self) -> str:
        return SWEEP_SCRIPT if self.sweep else (self.script or "")

    @property
    def kind_key(self) -> str:
        if self.kind:
            return self.kind
        if self.sweep:
            return f"sweep:{Path(self.sweep.root).name}:{Path(self.sweep.arms_file).stem}"
        return Path(self.script or "?").stem

    def all_vars(self) -> dict:
        v = {}
        if self.sweep:
            v.update(SWEEP_ROOT=self.sweep.root, SWEEP_MODULE=self.sweep.module,
                     ARMS_FILE=self.sweep.arms_file)
        v.update(self.vars)
        return v

    def dep_keys(self) -> list[str]:
        return [d if "/" in d else f"{self.pipeline}/{d}" for d in self.deps]


@dataclass
class Pipeline:
    name: str
    nodes: list
    budget_node_hours: float
    card: str = ""                          # the card the pipeline was approved under
    description: str = ""


def calib_node(node: Node) -> Node:
    """The generated calibration node for a node with an unknown runtime."""
    c = node.calibration
    v = dict(node.vars)
    v.update(c.vars)
    n = Node(id=node.id + "~calib", script=node.script, sweep=node.sweep, vars=v,
             deps=list(node.deps), nodes=min(node.nodes, 2), runtime_min=45.0,
             kind=node.kind_key + ":calib", queues=["debug", "debug-scaling"],
             approved=c.approved, max_resumes=0, max_retries=0,
             note=f"calibration for {node.id}")
    n.pipeline = node.pipeline
    return n


def load(pipeline_dir: Path, only: Optional[list] = None) -> dict:
    """Load every pipelines/*.py into one {key: Node} graph (plus generated calib nodes)."""
    graph, pipelines = {}, {}
    for path in sorted(Path(pipeline_dir).glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"dag_pipeline_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pipe = getattr(mod, "PIPELINE", None)
        if pipe is None:
            continue
        if only and pipe.name not in only:
            continue
        add_pipeline(graph, pipelines, pipe)
    return {"nodes": graph, "pipelines": pipelines}


def add_pipeline(graph: dict, pipelines: dict, pipe: Pipeline) -> None:
    if pipe.name in pipelines:
        raise ValueError(f"duplicate pipeline {pipe.name}")
    pipelines[pipe.name] = pipe
    for n in pipe.nodes:
        n.pipeline = pipe.name
        if n.key in graph:
            raise ValueError(f"duplicate node {n.key}")
        graph[n.key] = n
        if n.calibration is not None and n.runtime_min is None:
            c = calib_node(n)
            graph[c.key] = c


def graph_from(pipes: list) -> dict:
    graph, pipelines = {}, {}
    for p in pipes:
        add_pipeline(graph, pipelines, p)
    return {"nodes": graph, "pipelines": pipelines}


def topo_order(nodes: dict) -> list[str]:
    """Dependency order; raises on a cycle or an unknown dependency."""
    order, mark = [], {}

    def visit(k, stack):
        if mark.get(k) == 2:
            return
        if mark.get(k) == 1:
            raise ValueError("dependency cycle: " + " -> ".join(stack + [k]))
        mark[k] = 1
        for d in nodes[k].dep_keys():
            if d not in nodes:
                raise ValueError(f"{k} depends on unknown node {d}")
            visit(d, stack + [k])
        mark[k] = 2
        order.append(k)

    for k in sorted(nodes):
        visit(k, [])
    return order


def children(nodes: dict) -> dict:
    ch = {k: [] for k in nodes}
    for k, n in nodes.items():
        for d in n.dep_keys():
            if d in ch:
                ch[d].append(k)
    return ch


def descendants(nodes: dict, key: str) -> list[str]:
    ch, out, todo = children(nodes), [], [key]
    while todo:
        for c in ch[todo.pop()]:
            if c not in out:
                out.append(c)
                todo.append(c)
    return out


def logged_cards(decisions_file: Optional[Path]) -> Optional[set]:
    """Card IDs in the notes/DECISIONS.md log table (None if no file is configured)."""
    if decisions_file is None or not Path(decisions_file).exists():
        return None
    ids = set()
    for line in Path(decisions_file).read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and re.match(r"\d{4}-\d{2}-\d{2}", cells[1]):
            for part in re.split(r"\s*/\s*", cells[2]):
                ids.add(part)
    return ids


def validate(graph: dict, repo: Path, queues: dict, decisions_file=None) -> tuple[list, list]:
    """(errors, warnings). Light: file existence, names, vars, queues, cycles."""
    errors, warnings = [], []
    nodes = graph["nodes"]
    try:
        topo_order(nodes)
    except ValueError as e:
        errors.append(str(e))
    cards = logged_cards(decisions_file)
    for k, n in nodes.items():
        if not NAME_RE.match(n.id.replace("~", "-")):
            errors.append(f"{k}: bad node id")
        if bool(n.script) == bool(n.sweep):
            errors.append(f"{k}: exactly one of script / sweep")
        sp = n.script_path
        if sp and not (repo / sp).exists():
            errors.append(f"{k}: script {sp} does not exist")
        if n.sweep:
            if not (repo / n.sweep.root).is_dir():
                errors.append(f"{k}: SWEEP_ROOT {n.sweep.root} does not exist")
            arms = n.sweep.arms(repo)
            if not arms:
                errors.append(f"{k}: ARMS_FILE {n.sweep.arms_file} missing or empty")
            for a in arms:
                if not (repo / n.sweep.root / a / "training.args").exists():
                    errors.append(f"{k}: unknown arm {a}")
        for var, val in n.all_vars().items():
            if SECRET_RE.search(var):
                errors.append(f"{k}: {var} looks like a secret; never pass secrets via qsub -v")
            if "," in str(val):
                errors.append(f"{k}: {var} contains a comma (qsub -v splits on it)")
        bad_q = [q for q in n.queues if q not in queues]
        if bad_q:
            errors.append(f"{k}: unknown queues {bad_q}")
        if n.nodes < 1 or n.nodes > max(q["max_nodes"] for q in queues.values()):
            errors.append(f"{k}: nodes={n.nodes} fits no queue")
        if n.submit_with_parent and len(n.dep_keys()) != 1:
            errors.append(f"{k}: submit_with_parent needs exactly one dependency")
        if not n.approved:
            warnings.append(f"{k}: not approved -- will never be submitted")
        elif cards is not None and n.approved not in cards:
            warnings.append(f"{k}: card {n.approved} is not in the DECISIONS.md log -- "
                            "the scheduler refuses it")
        if n.runtime_min is None and n.calibration is None and not n.id.endswith("~calib"):
            warnings.append(f"{k}: runtime unknown and no calibration -- needs history")
        if "capacity" in n.queues and not n.dep_keys() and not n.adopt_job \
                and not n.id.endswith("~calib"):
            warnings.append(f"{k}: capacity-eligible node without a (debug) parent -- "
                            "is there a smoke test?")
    for name, p in graph["pipelines"].items():
        if p.budget_node_hours <= 0:
            errors.append(f"{name}: budget_node_hours must be > 0")
    return errors, warnings
