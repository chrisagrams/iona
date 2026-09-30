"""K150-P: StopAtStepCallback stops at the step while the schedule is defined over max_steps."""
from types import SimpleNamespace

import pytest

from msdelta.utils.callbacks import StopAtStepCallback


def _control():
    return SimpleNamespace(should_save=False, should_training_stop=False)


def test_stops_and_saves_at_step():
    cb = StopAtStepCallback(5)
    c = cb.on_step_end(None, SimpleNamespace(global_step=4), _control())
    assert not c.should_training_stop and not c.should_save
    c = cb.on_step_end(None, SimpleNamespace(global_step=5), _control())
    assert c.should_training_stop and c.should_save


def test_rejects_non_positive():
    with pytest.raises(ValueError):
        StopAtStepCallback(0)


def test_off_by_default(monkeypatch):
    import msdelta.utils.callbacks as cbm
    monkeypatch.delenv("MSDELTA_STOP_AT_STEP", raising=False)
    src = open(cbm.__file__).read()
    assert 'os.environ.get("MSDELTA_STOP_AT_STEP")' in src
