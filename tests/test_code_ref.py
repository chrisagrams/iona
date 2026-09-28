"""Running a job's code from a commit (K90): pbs/lib/code_snapshot.sh with MSDELTA_CODE_REF,
and the pbs/qsub_ref submit helper. CPU only, login-node safe: temporary git repositories,
and qsub replaced by `echo` (the real qsub is never called)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SNAPSHOT_SH = REPO / "pbs/lib/code_snapshot.sh"
QSUB_REF = REPO / "pbs/qsub_ref"
# The last commit before K90: its code_snapshot.sh is the reference for "no ref, no change".
PRE_K90 = "86ed0a727f425e124a1d8ffc66fb18b28bc7d45a"


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


FIRST = {
    "msdelta/__init__.py": "VERSION = 1\n",
    "msdelta/eval/library_search.py": "x = 1\n",
    "sweeps/make_x.py": "print('gen v1')\n",
    "sweeps/arms/models.txt": "binned01 binned:0.1\n",
    "tests/test_x.py": "def test_x(): pass\n",
    "configs/sweep-x/a/training.args": "--lr 1\n",
    "pbs/job.pbs": "#!/bin/bash\n# v1\nsource \"$REPO_DIR/pbs/lib/code_snapshot.sh\"\n",
    "pbs/plain.pbs": "#!/bin/bash\necho no snapshot\n",
    "pbs/logs/old.log": "tracked log, never copied\n",
    "baselines_wip/tool/run.py": "pass\n",
    "baselines_wip/tool/out.json": "{}\n",
    "data/prep.py": "pass\n",
    "data/README.md": "docs\n",
    "data/sub/deep.py": "pass\n",
    "pyproject.toml": "[project]\nname = 'x'\n",
    "notes/PLAN.md": "not code\n",
}


@pytest.fixture
def repo(tmp_path):
    """Two commits; the second changes code and drops a file; then an uncommitted edit."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write(root, FIRST)
    git(root, "add", "-A", "-f")
    git(root, "commit", "-q", "-m", "one")
    first = git(root, "rev-parse", "HEAD")
    write(root, {"msdelta/__init__.py": "VERSION = 2\n",
                 "pbs/job.pbs": "#!/bin/bash\n# v2 MSDELTA_CODE_REF\nsource \"$REPO_DIR/pbs/lib/code_snapshot.sh\"\n"})
    (root / "msdelta/eval/library_search.py").unlink()
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "two")
    second = git(root, "rev-parse", "HEAD")
    git(root, "branch", "feature", first)
    # Working-tree state that a commit snapshot must NOT pick up.
    write(root, {"msdelta/__init__.py": "VERSION = 'dirty'\n", "msdelta/untracked.py": "u\n"})
    return root, first, second


def snapshot(repo: Path, scratch: Path, extra: dict[str, str] | None = None,
             script: Path = SNAPSHOT_SH, twice: bool = False):
    env = {**os.environ, "REPO_DIR": str(repo), "SCRATCH_ROOT": str(scratch),
           "PBS_JOBID": "123.test", "PYTHONPATH": f"{repo}:/keep/me"}
    for key in ("MSDELTA_CODE_REF", "MSDELTA_CODE_REF_NAME", "MSDELTA_NO_SNAPSHOT",
                "MSDELTA_CODE_DIR", "PYTHONSAFEPATH"):
        env.pop(key, None)
    env.update(extra or {})
    body = f'source "{script}"\n' * (2 if twice else 1)
    body += 'echo "PP=$PYTHONPATH"; echo "SAFE=${PYTHONSAFEPATH:-}"; echo "DIR=$MSDELTA_CODE_DIR"\n'
    return subprocess.run(["bash", "-c", "set -uo pipefail\n" + body], cwd=repo,
                          capture_output=True, text=True, env=env)


def tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


def fields(snapshot_txt: Path) -> dict[str, str]:
    out = {}
    for line in snapshot_txt.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


class TestCodeSnapshotRef:
    def test_snapshot_is_the_commit_not_the_working_tree(self, repo, tmp_path):
        root, first, _ = repo
        out = snapshot(root, tmp_path / "scratch",
                       {"MSDELTA_CODE_REF": first, "MSDELTA_CODE_REF_NAME": "feature"})
        assert out.returncode == 0, out.stderr
        snap = tmp_path / "scratch/code-snapshots/123"
        got = tree(snap)
        assert got.pop("SNAPSHOT.txt")
        expected = {name: text.encode() for name, text in FIRST.items()
                    if name not in {"pbs/logs/old.log", "baselines_wip/tool/out.json",
                                    "data/README.md", "data/sub/deep.py", "notes/PLAN.md"}}
        assert got == expected
        # ...and the snapshot is what Python sees.
        assert f"PP={snap}:/keep/me" in out.stdout
        assert "SAFE=1" in out.stdout

    def test_snapshot_txt_records_the_commit(self, repo, tmp_path):
        root, first, _ = repo
        out = snapshot(root, tmp_path / "s",
                       {"MSDELTA_CODE_REF": first[:10], "MSDELTA_CODE_REF_NAME": "feature"})
        assert out.returncode == 0, out.stderr
        f = fields(tmp_path / "s/code-snapshots/123/SNAPSHOT.txt")
        assert f["mode"] == "git-archive"
        assert f["commit"] == first          # full sha, even from a short one
        assert f["ref"] == "feature" and f["branch"] == "feature"
        assert f["dirty"].startswith("n/a")
        assert f["job"] == "123.test"

    @pytest.mark.parametrize("bad", ["0123456789abcdef0123456789abcdef01234567", "main",
                                     "not a sha"])
    def test_a_ref_that_is_not_a_known_sha_fails(self, repo, tmp_path, bad):
        root, _, _ = repo
        out = snapshot(root, tmp_path / "s", {"MSDELTA_CODE_REF": bad})
        assert out.returncode == 3
        assert "is not a commit sha" in out.stderr
        assert "PP=" not in out.stdout

    def test_ref_with_no_snapshot_is_refused(self, repo, tmp_path):
        root, first, _ = repo
        out = snapshot(root, tmp_path / "s",
                       {"MSDELTA_CODE_REF": first, "MSDELTA_NO_SNAPSHOT": "1"})
        assert out.returncode == 3

    def test_sourcing_twice_copies_once_and_keeps_the_checkout_off_pythonpath(self, repo, tmp_path):
        root, first, _ = repo
        out = snapshot(root, tmp_path / "s", {"MSDELTA_CODE_REF": first}, twice=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.count("=== code snapshot:") == 1
        snap = tmp_path / "s/code-snapshots/123"
        assert f"PP={snap}:/keep/me\n" in out.stdout

    def test_code_path_resolves_committed_files_only(self, repo, tmp_path):
        root, first, _ = repo
        (root / "sweeps/arms/adhoc.txt").write_text("x\n")
        env = {**os.environ, "REPO_DIR": str(root), "SCRATCH_ROOT": str(tmp_path / "s"),
               "PBS_JOBID": "9", "MSDELTA_CODE_REF": first}
        body = (f'source "{SNAPSHOT_SH}"\n'
                'msdelta_code_path sweeps/arms/models.txt\n'
                'msdelta_code_path sweeps/arms/adhoc.txt\n'
                'msdelta_code_path /abs/file\n'
                'echo "prefix=$MSDELTA_REF_PREFIX"\n')
        out = subprocess.run(["bash", "-c", body], cwd=root, capture_output=True, text=True,
                             env=env)
        snap = tmp_path / "s/code-snapshots/9"
        lines = out.stdout.splitlines()[1:]
        assert lines == [f"{snap}/sweeps/arms/models.txt", "sweeps/arms/adhoc.txt",
                         "/abs/file", f"prefix={snap}/"]


class TestNoRefUnchanged:
    def _old_script(self, tmp_path):
        try:
            old = subprocess.run(["git", "-C", str(REPO), "show",
                                  f"{PRE_K90}:pbs/lib/code_snapshot.sh"],
                                 check=True, capture_output=True, text=True).stdout
        except subprocess.CalledProcessError:
            pytest.skip("pre-K90 commit not in this clone")
        path = tmp_path / "old_code_snapshot.sh"
        path.write_text(old)
        return path

    def test_same_snapshot_and_output_as_before_k90(self, repo, tmp_path):
        """Without MSDELTA_CODE_REF the copy, the log and PYTHONPATH are exactly the pre-K90
        script's; SNAPSHOT.txt only GAINS `mode:` and `dirty:` lines, and a dirty tree adds
        uncommitted.diff."""
        root, _, _ = repo
        a = snapshot(root, tmp_path / "old", script=self._old_script(tmp_path))
        b = snapshot(root, tmp_path / "new")
        assert a.returncode == b.returncode == 0, a.stderr + b.stderr
        norm = lambda s, d: s.replace(str(tmp_path / d), "<SCRATCH>")
        assert norm(a.stdout, "old") == norm(b.stdout, "new")
        assert a.stderr == b.stderr
        ta = tree(tmp_path / "old/code-snapshots/123")
        tb = tree(tmp_path / "new/code-snapshots/123")
        keep = lambda t, extra=(): [l for l in t.decode().splitlines()
                                    if not l.startswith(("taken:",) + extra)]
        new_snapshot = tb.pop("SNAPSHOT.txt")
        assert keep(ta.pop("SNAPSHOT.txt")) == keep(new_snapshot, ("mode:", "dirty:"))
        diff = tb.pop("uncommitted.diff").decode()
        assert ta == tb
        # the working tree, dirty edit and untracked file included, as before
        assert tb["msdelta/__init__.py"] == b"VERSION = 'dirty'\n"
        assert "msdelta/untracked.py" in tb
        f = fields(tmp_path / "new/code-snapshots/123/SNAPSHOT.txt")
        assert f["dirty"].startswith("yes") and f["mode"].startswith("working-tree")
        assert len(f["commit"]) == 40
        assert "+VERSION = 'dirty'" in diff and "-VERSION = 2" in diff

    def test_dirty_diff_rebuilds_the_tracked_code(self, repo, tmp_path):
        """commit + uncommitted.diff == the tracked files of the dirty copy."""
        root, _, second = repo
        out = snapshot(root, tmp_path / "s")
        assert out.returncode == 0, out.stderr
        snap = tmp_path / "s/code-snapshots/123"
        rebuilt = tmp_path / "rebuilt"
        git(root, "worktree", "add", "-q", "--detach", str(rebuilt), second)
        subprocess.run(["git", "-C", str(rebuilt), "apply", str(snap / "uncommitted.diff")],
                       check=True)
        assert (rebuilt / "msdelta/__init__.py").read_bytes() == \
            (snap / "msdelta/__init__.py").read_bytes()

    def test_clean_tree_says_so(self, repo, tmp_path):
        root, _, _ = repo
        (root / "msdelta/untracked.py").unlink()
        git(root, "checkout", "-q", "--", "msdelta/__init__.py")
        out = snapshot(root, tmp_path / "s")
        assert out.returncode == 0, out.stderr
        snap = tmp_path / "s/code-snapshots/123"
        assert fields(snap / "SNAPSHOT.txt")["dirty"].startswith("no")
        assert fields(snap / "SNAPSHOT.txt")["mode"] == "git-archive (clean HEAD)"
        assert not (snap / "uncommitted.diff").exists()


@pytest.fixture
def qrepo(repo):
    """The temp repo with this checkout's qsub_ref and code_snapshot.sh committed in it
    (qsub_ref finds its checkout from its own location)."""
    root, first, second = repo
    (root / "pbs/lib").mkdir(parents=True, exist_ok=True)
    shutil.copy(QSUB_REF, root / "pbs/qsub_ref")
    shutil.copy(SNAPSHOT_SH, root / "pbs/lib/code_snapshot.sh")
    return root, first, second


def qsub_ref(root: Path, scratch: Path, *args: str, cwd: Path | None = None):
    env = {**os.environ, "QSUB": "echo", "SCRATCH_ROOT": str(scratch)}
    return subprocess.run(["bash", str(root / "pbs/qsub_ref"), *args], cwd=cwd or root,
                          capture_output=True, text=True, env=env)


class TestQsubRef:
    def test_builds_the_qsub_command(self, qrepo, tmp_path):
        root, first, _ = qrepo
        out = qsub_ref(root, tmp_path / "s", "feature", "-q", "debug", "-l", "select=1",
                       "-v", "MODELS=m.txt,SPLIT=validation", "-l", "walltime=00:30:00",
                       "-vOUT_DIR=results/x", "pbs/job.pbs")
        assert out.returncode == 0, out.stderr
        copy = tmp_path / "s/code-refs/submit" / f"{first[:12]}-job.pbs"
        assert out.stdout.split() == [
            "-q", "debug", "-l", "select=1", "-l", "walltime=00:30:00", "-v",
            f"MODELS=m.txt,SPLIT=validation,OUT_DIR=results/x,MSDELTA_CODE_REF={first},"
            f"MSDELTA_CODE_REF_NAME=feature,REPO_DIR={root}",
            str(copy)]
        # The job's own script is the COMMIT's (v1), not the checkout's (v2).
        assert copy.read_text() == FIRST["pbs/job.pbs"]
        assert "predates MSDELTA_CODE_REF" in out.stderr

    def test_no_user_v_and_a_sha_ref(self, qrepo, tmp_path):
        root, _, second = qrepo
        out = qsub_ref(root, tmp_path / "s", second[:8], "pbs/job.pbs")
        assert out.returncode == 0, out.stderr
        assert out.stdout.split()[:2] == [
            "-v", f"MSDELTA_CODE_REF={second},MSDELTA_CODE_REF_NAME={second[:8]},REPO_DIR={root}"]
        assert "predates" not in out.stderr

    def test_script_path_relative_to_cwd(self, qrepo, tmp_path):
        root, _, second = qrepo
        out = qsub_ref(root, tmp_path / "s", "main", "job.pbs", cwd=root / "pbs")
        assert out.returncode == 0, out.stderr
        assert out.stdout.split()[-1].endswith(f"{second[:12]}-job.pbs")

    def test_user_repo_dir_wins(self, qrepo, tmp_path):
        root, _, _ = qrepo
        out = qsub_ref(root, tmp_path / "s", "main", "-v", "REPO_DIR=/elsewhere", "pbs/job.pbs")
        assert out.returncode == 0, out.stderr
        assert out.stdout.count("REPO_DIR=") == 1 and "REPO_DIR=/elsewhere" in out.stdout

    @pytest.mark.parametrize("args,message", [
        (("nosuch", "pbs/job.pbs"), "is not a commit"),
        (("main", "pbs/missing.pbs"), "does not exist at"),
        (("main", "pbs/plain.pbs"), "takes no code snapshot"),
        (("main", "-v", "MSDELTA_CODE_REF=abc", "pbs/job.pbs"), "do not pass MSDELTA_CODE_REF"),
        (("main", "/tmp/outside.pbs"), "is not inside the checkout"),
    ])
    def test_refusals(self, qrepo, tmp_path, args, message):
        root, _, _ = qrepo
        out = qsub_ref(root, tmp_path / "s", *args)
        assert out.returncode == 2
        assert message in out.stderr
        assert out.stdout == ""       # qsub (echo) never ran


def test_sweep_runner_reads_the_grid_from_the_ref(tmp_path):
    """Dry run of the real sweep script under a ref: the snapshot is taken first, and the
    grid, the staleness check and the config snapshot all come from the commit."""
    nodes = tmp_path / "nodes"
    nodes.write_text("n0\n")
    sha = git(REPO, "rev-parse", "HEAD")
    env = {**os.environ, "DRY_RUN": "1", "PBS_NODEFILE": str(nodes), "PBS_JOBID": "dryref",
           "SCRATCH_ROOT": str(tmp_path / "s"), "TILES_PER_ARM": "12",
           "MSDELTA_CODE_REF": sha}
    out = subprocess.run(["bash", "pbs/aurora-finetune-sweep.pbs"], cwd=REPO,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.count("=== code snapshot:") == 1
    assert "git archive of" in out.stdout
    snap = tmp_path / "s/code-snapshots/dryref"
    assert fields(snap / "SNAPSHOT.txt")["commit"] == sha
    configs = tmp_path / "s/runs/.configs-dryref"
    assert (configs / "CODE_SNAPSHOT.txt").read_text() == (snap / "SNAPSHOT.txt").read_text()
    committed = git(REPO, "ls-tree", "-d", "--name-only", f"{sha}:configs/sweep-denoise").split()
    assert sorted(p.name for p in configs.iterdir() if p.is_dir()) == sorted(committed)


def test_code_provenance_reads_both_snapshot_kinds(tmp_path, monkeypatch):
    from msdelta.utils.provenance import code_provenance
    monkeypatch.delenv("MSDELTA_CODE_DIR", raising=False)
    assert code_provenance() == {}
    ref = tmp_path / "ref"; ref.mkdir()
    (ref / "SNAPSHOT.txt").write_text(
        "source:   /r\nmode:     git-archive\nref:      c25\nbranch:   c25\n"
        "commit:   abc123\ntaken:    t\njob:      1\ndirty:    n/a (built from the commit)\n")
    monkeypatch.setenv("MSDELTA_CODE_DIR", str(ref))
    assert code_provenance() == {"snapshot": str(ref), "mode": "git-archive", "commit": "abc123",
                                 "branch": "c25", "ref": "c25", "dirty": "n/a"}
    tree = tmp_path / "tree"; tree.mkdir()
    (tree / "SNAPSHOT.txt").write_text(
        "source:   /r\nbranch:   dev\ncommit:   def456\ntaken:    t\njob:      1\n"
        "uncommitted changes in the checkout at snapshot time (code dirs):\n     M msdelta/x.py\n")
    monkeypatch.setenv("MSDELTA_CODE_DIR", str(tree))
    assert code_provenance() == {"snapshot": str(tree), "mode": "working-tree",
                                 "commit": "def456", "branch": "dev", "dirty": True}
    (tree / "SNAPSHOT.txt").write_text("mode:     working-tree (rsync)\ncommit:   d\n"
                                       "dirty:    no (clean)\nuncommitted changes (code dirs):\n")
    assert code_provenance()["dirty"] is False


class TestSweepResume:
    """RESUME_JOB continues the ORIGINAL job: its config snapshot and its code (the commit
    it recorded, or its own snapshot directory when that was dirty). Dry runs of the real
    sweep script against fake old-job directories."""

    ARMS = ["lr1e5_es01_ep2_h128", "lr2e4_es10_ep4_h512"]

    def _old_job(self, scratch: Path, snapshot_txt: str | None, configs: bool = True):
        if configs:
            cfg = scratch / "runs/.configs-OLD"
            for arm in self.ARMS:
                (cfg / arm).mkdir(parents=True)
                shutil.copy(REPO / "configs/sweep-denoise" / arm / "training.args",
                            cfg / arm / "training.args")
            (cfg / "CODE_SNAPSHOT.txt").write_text("old record\n")
        if snapshot_txt is not None:
            snap = scratch / "code-snapshots/OLD"
            (snap / "msdelta").mkdir(parents=True)
            (snap / "SNAPSHOT.txt").write_text(snapshot_txt)

    def _run(self, tmp_path, **extra):
        nodes = tmp_path / "nodes"
        nodes.write_text("n0\n")
        env = {**os.environ, "DRY_RUN": "1", "PBS_NODEFILE": str(nodes), "PBS_JOBID": "NEW.x",
               "SCRATCH_ROOT": str(tmp_path / "s"), "TILES_PER_ARM": "12", "RESUME_JOB": "OLD"}
        for key in ("MSDELTA_CODE_REF", "MSDELTA_CODE_REUSE", "MSDELTA_CODE_DIR"):
            env.pop(key, None)
        env.update(extra)
        out = subprocess.run(["bash", "pbs/aurora-finetune-sweep.pbs"], cwd=REPO,
                             capture_output=True, text=True, env=env)
        return out, out.stdout + out.stderr

    @staticmethod
    def _slots(stdout: str) -> list[str]:
        arms = []
        for line in stdout.splitlines():
            if line.startswith("  slot"):
                arms += line.split(":", 1)[1].split()
        return sorted(arms)

    def test_clean_original_resumes_from_its_commit_and_configs(self, tmp_path):
        sha = git(REPO, "rev-parse", "HEAD")
        self._old_job(tmp_path / "s", f"source:   /x\nbranch:   dev\ncommit:   {sha}\n"
                      "dirty:    no\nuncommitted changes in the checkout at snapshot time:\n")
        out, text = self._run(tmp_path)
        assert out.returncode == 0, text
        assert self._slots(out.stdout) == self.ARMS          # the old job's arms, not 216
        assert "grid check skipped" in text
        assert f"code from its commit {sha[:12]}" in text
        new = fields(tmp_path / "s/code-snapshots/NEW/SNAPSHOT.txt")
        assert new["mode"] == "git-archive" and new["commit"] == sha and new["branch"] == "dev"
        cfg = tmp_path / "s/runs/.configs-NEW"
        assert sorted(p.name for p in cfg.iterdir() if p.is_dir()) == self.ARMS
        assert "git-archive" in (cfg / "CODE_SNAPSHOT.txt").read_text()

    @pytest.mark.parametrize("dirty", [
        "uncommitted changes in the checkout at snapshot time (code dirs):\n     M msdelta/x.py\n",
        "dirty:    yes\nuncommitted changes in the checkout at snapshot time (code dirs):\n"
        "     M msdelta/x.py\n"])
    def test_dirty_original_reuses_its_snapshot_directory(self, tmp_path, dirty):
        sha = git(REPO, "rev-parse", "HEAD")
        old_txt = f"source:   /x\nbranch:   dev\ncommit:   {sha}\njob:      OLD.x\n" + dirty
        self._old_job(tmp_path / "s", old_txt)
        out, text = self._run(tmp_path)
        assert out.returncode == 0, text
        assert "DIRTY working tree" in out.stderr
        assert "REUSING" in out.stdout
        assert not (tmp_path / "s/code-snapshots/NEW").exists()   # nothing new copied
        assert (tmp_path / "s/runs/.configs-NEW/CODE_SNAPSHOT.txt").read_text() == old_txt
        assert self._slots(out.stdout) == self.ARMS

    def test_unknown_commit_reuses_the_snapshot_directory(self, tmp_path):
        self._old_job(tmp_path / "s", "branch:   dev\ncommit:   " + "e" * 40 + "\ndirty:    no\n")
        out, text = self._run(tmp_path)
        assert out.returncode == 0, text
        assert "is not in" in out.stderr and "REUSING" in out.stdout

    def test_missing_config_snapshot_is_refused(self, tmp_path):
        self._old_job(tmp_path / "s", "commit:   abc\n", configs=False)
        out, text = self._run(tmp_path)
        assert out.returncode == 2 and "config snapshot" in text

    def test_missing_code_snapshot_is_refused(self, tmp_path):
        self._old_job(tmp_path / "s", None)
        out, text = self._run(tmp_path)
        assert out.returncode == 2 and "no code snapshot" in text

    def test_a_different_ref_is_refused(self, tmp_path):
        sha = git(REPO, "rev-parse", "HEAD")
        self._old_job(tmp_path / "s", f"commit:   {sha}\ndirty:    no\n")
        other = git(REPO, "rev-parse", "HEAD~1")
        out, text = self._run(tmp_path, MSDELTA_CODE_REF=other)
        assert out.returncode == 2 and "RESUME_CODE=current" in text

    def test_current_overrides_restore_the_old_behaviour(self, tmp_path):
        self._old_job(tmp_path / "s", None, configs=False)
        out, text = self._run(tmp_path, RESUME_CODE="current", RESUME_CONFIGS="current",
                              SKIP_GRID_CHECK="1")
        assert out.returncode == 0, text
        assert "resume OLD" not in text
        new = fields(tmp_path / "s/code-snapshots/NEW/SNAPSHOT.txt")
        # the checkout's code (rsync of a dirty tree, or git archive of a clean HEAD), not a ref
        assert new["mode"] in ("working-tree (rsync)", "git-archive (clean HEAD)")
        assert "ref" not in new
        assert len(self._slots(out.stdout)) > len(self.ARMS)


# ---------------------------------------------------------------------------------------------
# K111: race-free snapshot of the checkout (no ref). A clean tree is `git archive` of the HEAD
# sha resolved first; a dirty tree is rsynced under a SHARED flock that pbs/checkout_ff takes
# EXCLUSIVELY while it fast-forwards.

CHECKOUT_FF = REPO / "pbs/checkout_ff"
# The last commit before K111: its code_snapshot.sh is the rsync-only reference.
PRE_K111 = "a797659158588f099df9fab696dc9f55bff36ede"
NEXT = {  # a fast-forward of FIRST: changes, adds and deletes code files
    "msdelta/__init__.py": "VERSION = 'next'\n",
    "msdelta/new_module.py": "NEW = 1\n",
    "tests/test_x.py": "def test_x(): assert True\n",
}
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


@pytest.fixture
def clean(tmp_path):
    """A clean checkout on `main` (= commit A) with branch `next` (= B) one fast-forward
    ahead; .gitignore'd __pycache__ and pbs/logs files lying around, as in the real one."""
    root = tmp_path / "clean"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write(root, {**FIRST, ".gitignore": "__pycache__/\n*.py[oc]\npbs/logs/\n"})
    git(root, "add", "-A", "-f")
    git(root, "commit", "-q", "-m", "A")
    a = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "-b", "next")
    write(root, NEXT)
    (root / "sweeps/make_x.py").unlink()
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "B")
    b = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "main")
    write(root, {"msdelta/__pycache__/x.cpython-312.pyc": "bytecode",
                 "pbs/logs/new.log": "log\n"})
    assert git(root, "status", "--porcelain") == ""
    return root, a, b


def commit_code(root: Path, sha: str, tmp: Path) -> dict[str, bytes]:
    """The code files of commit `sha`, as the snapshot must hold them."""
    out = tmp / f"expect-{sha[:8]}"
    wt = tmp / f"wt-{sha[:8]}"
    if not out.exists():
        git(root, "worktree", "add", "-q", "--detach", str(wt), sha)
        r = snapshot(wt, out, script=_pre_k111(tmp))       # rsync of a clean checkout of sha
        assert r.returncode == 0, r.stderr
        git(root, "worktree", "remove", "--force", str(wt))
    t = tree(out / "code-snapshots/123")
    t.pop("SNAPSHOT.txt")
    return t


def _pre_k111(tmp: Path) -> Path:
    path = tmp / "pre_k111_code_snapshot.sh"
    if not path.exists():
        try:
            text = subprocess.run(["git", "-C", str(REPO), "show",
                                   f"{PRE_K111}:pbs/lib/code_snapshot.sh"],
                                  check=True, capture_output=True, text=True).stdout
        except subprocess.CalledProcessError:
            pytest.skip("pre-K111 commit not in this clone")
        path.write_text(text)
    return path


def lock_of(root: Path) -> Path:
    return root / ".git/msdelta-checkout.lock"


def code_of(snap: Path, pycache: bool = True) -> dict[str, bytes]:
    t = tree(snap)
    t.pop("SNAPSHOT.txt")
    return t if pycache else {k: v for k, v in t.items() if "__pycache__" not in k}


class TestAtomicSnapshot:
    def test_clean_tree_is_git_archive_of_head_with_the_rsync_file_set(self, clean, tmp_path):
        root, a, _ = clean
        old = snapshot(root, tmp_path / "old", script=_pre_k111(tmp_path))
        new = snapshot(root, tmp_path / "new")
        assert old.returncode == new.returncode == 0, old.stderr + new.stderr
        f = fields(tmp_path / "new/code-snapshots/123/SNAPSHOT.txt")
        assert f["mode"] == "git-archive (clean HEAD)"
        assert f["commit"] == a and f["branch"] == "main" and f["dirty"].startswith("no")
        # exactly the files (and bytes) the pre-K111 rsync copied from the same clean tree,
        # except __pycache__: its exclude rule never matched before K111 (fixed; see below)
        assert "msdelta/__pycache__/x.cpython-312.pyc" in code_of(tmp_path / "old/code-snapshots/123")
        assert code_of(tmp_path / "new/code-snapshots/123") == \
            code_of(tmp_path / "old/code-snapshots/123", pycache=False)
        # the log line and PYTHONPATH are unchanged
        norm = lambda s, d: s.replace(str(tmp_path / d), "<SCRATCH>")
        assert norm(old.stdout, "old") == norm(new.stdout, "new")
        # SNAPSHOT.txt: the same lines, only `mode:` (and `taken:`) differ
        keep = lambda p: [l for l in p.read_text().splitlines()
                          if not l.startswith(("taken:", "mode:"))]
        assert keep(tmp_path / "old/code-snapshots/123/SNAPSHOT.txt") == \
            keep(tmp_path / "new/code-snapshots/123/SNAPSHOT.txt")
        assert not lock_of(root).exists()          # a clean snapshot takes no lock

    def test_clean_tree_file_set_matches_rsync_on_the_bigger_fixture(self, repo, tmp_path):
        """The FIRST/second fixture with its dirty edits undone: every include/exclude rule
        (pbs/logs, baselines_wip non-code, data/ subdirs and docs, notes/) agrees."""
        root, _, second = repo
        (root / "msdelta/untracked.py").unlink()
        git(root, "checkout", "-q", "--", "msdelta/__init__.py")
        old = snapshot(root, tmp_path / "old", script=_pre_k111(tmp_path))
        new = snapshot(root, tmp_path / "new")
        assert old.returncode == new.returncode == 0
        assert fields(tmp_path / "new/code-snapshots/123/SNAPSHOT.txt")["commit"] == second
        assert code_of(tmp_path / "new/code-snapshots/123") == \
            code_of(tmp_path / "old/code-snapshots/123")

    def test_dirty_tree_is_rsynced_under_the_shared_lock(self, clean, tmp_path):
        root, a, _ = clean
        (root / "msdelta/__init__.py").write_text("VERSION = 'edited'\n")
        lock = lock_of(root)
        report = tmp_path / "lockstate"
        hook = (f'[ "$1" = rsync-copied ] || exit 0\n'
                f'flock -n -x {lock} true && echo x-free >> {report} || echo x-held >> {report}\n'
                f'flock -n -s {lock} true && echo s-free >> {report} || echo s-held >> {report}\n')
        out = snapshot(root, tmp_path / "s", {"MSDELTA_SNAPSHOT_TEST_HOOK": hook})
        assert out.returncode == 0, out.stderr
        assert report.read_text().split() == ["x-held", "s-free"]   # shared, held during copy
        snap = tmp_path / "s/code-snapshots/123"
        f = fields(snap / "SNAPSHOT.txt")
        assert f["mode"] == "working-tree (rsync)" and f["commit"] == a
        assert f["dirty"].startswith("yes")
        assert (snap / "msdelta/__init__.py").read_text() == "VERSION = 'edited'\n"
        assert "+VERSION = 'edited'" in (snap / "uncommitted.diff").read_text()
        assert not (snap / "msdelta/__pycache__").exists()      # rsync now drops it too
        # released when the snapshot is done
        assert subprocess.run(["flock", "-n", "-x", str(lock), "true"]).returncode == 0

    def test_untracked_code_file_counts_as_dirty(self, clean, tmp_path):
        root, _, _ = clean
        (root / "msdelta/adhoc.py").write_text("x\n")
        out = snapshot(root, tmp_path / "s")
        assert out.returncode == 0, out.stderr
        snap = tmp_path / "s/code-snapshots/123"
        assert fields(snap / "SNAPSHOT.txt")["mode"] == "working-tree (rsync)"
        assert (snap / "msdelta/adhoc.py").exists()

    def test_fast_forward_during_a_clean_snapshot_gives_exactly_one_commit(self, clean, tmp_path):
        """The checkout is fast-forwarded A -> B after HEAD was resolved and before the copy:
        the snapshot is exactly A's code, recorded as A."""
        root, a, b = clean
        hook = f'[ "$1" = clean-resolved ] && git -C {root} merge -q --ff-only next'
        out = snapshot(root, tmp_path / "s", {"MSDELTA_SNAPSHOT_TEST_HOOK": hook, **GIT_ENV})
        assert out.returncode == 0, out.stderr
        assert git(root, "rev-parse", "HEAD") == b             # the race really happened
        snap = tmp_path / "s/code-snapshots/123"
        f = fields(snap / "SNAPSHOT.txt")
        assert f["commit"] == a and f["mode"] == "git-archive (clean HEAD)"
        assert code_of(snap) == commit_code(root, a, tmp_path)
        assert code_of(snap) != commit_code(root, b, tmp_path)

    def test_concurrent_locked_checkout_changes_never_mix(self, clean, tmp_path):
        """Stress: the checkout flips A <-> B under the exclusive lock (as pbs/checkout_ff
        does) while snapshots run; every snapshot is exactly the commit it records."""
        import threading
        import time
        root, a, b = clean
        expect = {a: commit_code(root, a, tmp_path), b: commit_code(root, b, tmp_path)}
        stop = threading.Event()

        def flip():
            target = [b, a]
            i = 0
            while not stop.is_set():
                subprocess.run(["flock", "-x", str(lock_of(root)), "git", "-C", str(root),
                                "reset", "-q", "--hard", target[i % 2]], capture_output=True)
                i += 1
                time.sleep(0.03)

        t = threading.Thread(target=flip)
        t.start()
        try:
            results = []
            for k in range(12):
                out = snapshot(root, tmp_path / "s", {"PBS_JOBID": f"j{k}"})
                results.append(out)
        finally:
            stop.set()
            t.join()
        for k, out in enumerate(results):
            assert out.returncode == 0, out.stderr
            snap = tmp_path / f"s/code-snapshots/j{k}"
            f = fields(snap / "SNAPSHOT.txt")
            assert f["commit"] in expect
            assert code_of(snap) == expect[f["commit"]], (k, f)

    def test_dirty_head_moving_during_the_copy_is_retried_once(self, clean, tmp_path):
        root, _, _ = clean
        (root / "msdelta/__init__.py").write_text("VERSION = 'edited'\n")
        marker = tmp_path / "moved"
        hook = (f'[ "$1" = rsync-copied ] && [ ! -e {marker} ] || exit 0\n'
                f'touch {marker}; git -C {root} commit -q --allow-empty -m moved')
        out = snapshot(root, tmp_path / "s", {"MSDELTA_SNAPSHOT_TEST_HOOK": hook, **GIT_ENV})
        assert out.returncode == 0, out.stderr
        assert "moved from" in out.stderr and "copying again" in out.stderr
        f = fields(tmp_path / "s/code-snapshots/123/SNAPSHOT.txt")
        assert f["commit"] == git(root, "rev-parse", "HEAD")    # the second, stable copy

    def test_dirty_head_moving_twice_fails_loudly(self, clean, tmp_path):
        root, _, _ = clean
        (root / "msdelta/__init__.py").write_text("VERSION = 'edited'\n")
        hook = f'[ "$1" = rsync-copied ] && git -C {root} commit -q --allow-empty -m moved'
        out = snapshot(root, tmp_path / "s", {"MSDELTA_SNAPSHOT_TEST_HOOK": hook, **GIT_ENV})
        assert out.returncode == 3
        assert "HEAD moved twice" in out.stderr
        assert "PP=" not in out.stdout

    def test_dirty_snapshot_waits_for_the_exclusive_lock_then_gives_up(self, clean, tmp_path):
        root, _, _ = clean
        lock_of(root).touch()
        holder = subprocess.Popen(["flock", "-x", str(lock_of(root)), "sleep", "20"])
        try:
            import time
            for _ in range(100):      # wait until the holder has the lock
                if subprocess.run(["flock", "-n", "-s", str(lock_of(root)), "true"]).returncode:
                    break
                time.sleep(0.05)
            # a clean tree needs no lock
            ok = snapshot(root, tmp_path / "c", {"MSDELTA_SNAPSHOT_LOCK_TIMEOUT": "1"})
            assert ok.returncode == 0, ok.stderr
            (root / "msdelta/__init__.py").write_text("VERSION = 'edited'\n")
            out = snapshot(root, tmp_path / "d", {"MSDELTA_SNAPSHOT_LOCK_TIMEOUT": "1"})
            assert out.returncode == 3
            assert "no shared lock" in out.stderr
        finally:
            holder.kill()
            holder.wait()


def checkout_ff(*args: str, script: Path = CHECKOUT_FF, env: dict | None = None, cwd=None):
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          env={**os.environ, **GIT_ENV, **(env or {})}, cwd=cwd)


class TestCheckoutFF:
    def test_fast_forwards_holding_the_exclusive_lock(self, clean, tmp_path):
        root, a, b = clean
        seen = tmp_path / "seen"
        hook = root / ".git/hooks/post-merge"
        hook.write_text(f"#!/bin/bash\nflock -n -s {lock_of(root)} true; echo $? > {seen}\n")
        hook.chmod(0o755)
        out = checkout_ff("-C", str(root), "next")
        assert out.returncode == 0, out.stderr
        assert git(root, "rev-parse", "HEAD") == b
        assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        assert seen.read_text().strip() == "1"        # no shared lock possible during the ff
        assert f"{a[:12]} -> {b[:12]}" in out.stdout
        assert subprocess.run(["flock", "-n", "-x", str(lock_of(root)), "true"]).returncode == 0

    def test_refuses_a_dirty_checkout(self, clean, tmp_path):
        root, a, _ = clean
        (root / "msdelta/__init__.py").write_text("VERSION = 'edited'\n")
        out = checkout_ff("-C", str(root), "next")
        assert out.returncode == 2
        assert "uncommitted changes" in out.stderr and "msdelta/__init__.py" in out.stderr
        assert git(root, "rev-parse", "HEAD") == a

    def test_waits_for_a_shared_holder_then_gives_up(self, clean, tmp_path):
        import time
        root, a, _ = clean
        lock_of(root).touch()
        holder = subprocess.Popen(["flock", "-s", str(lock_of(root)), "sleep", "20"])
        try:
            for _ in range(100):
                if subprocess.run(["flock", "-n", "-x", str(lock_of(root)), "true"]).returncode:
                    break
                time.sleep(0.05)
            out = checkout_ff("-C", str(root), "next", env={"CHECKOUT_FF_TIMEOUT": "1"})
            assert out.returncode == 3 and "no exclusive lock" in out.stderr
            assert git(root, "rev-parse", "HEAD") == a
        finally:
            holder.kill()
            holder.wait()

    def test_not_a_fast_forward_and_unknown_branch(self, clean, tmp_path):
        root, a, _ = clean
        git(root, "checkout", "-q", "-b", "side", a)
        write(root, {"msdelta/side.py": "s\n"})
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "side")
        git(root, "checkout", "-q", "main")
        out = checkout_ff("-C", str(root), "side")
        assert out.returncode == 0                      # main is behind side: a real ff
        out = checkout_ff("-C", str(root), "next")      # diverged now
        assert out.returncode == 1 and "not a fast-forward" in out.stderr
        out = checkout_ff("-C", str(root), "nosuch")
        assert out.returncode == 2 and "is not a commit" in out.stderr

    def test_default_target_is_the_main_worktree(self, clean, tmp_path):
        """Run from a scratch worktree (the K111 procedure) with no -C: the MAIN checkout
        is fast-forwarded, not the worktree the script lives in."""
        root, a, b = clean
        scratch = tmp_path / "merge-wt"
        git(root, "worktree", "add", "-q", "--detach", str(scratch), a)
        (scratch / "pbs").mkdir(exist_ok=True)
        shutil.copy(CHECKOUT_FF, scratch / "pbs/checkout_ff")
        out = checkout_ff("next", script=scratch / "pbs/checkout_ff", cwd=scratch)
        assert out.returncode == 0, out.stderr
        assert git(root, "rev-parse", "HEAD") == b
        assert git(scratch, "rev-parse", "HEAD") == a
