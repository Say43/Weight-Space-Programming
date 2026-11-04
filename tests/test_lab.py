import pytest

from editor import lab
from eval import config as cfg


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "EXPERIMENT_LOG", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(cfg, "LOG_DIR", tmp_path)


def test_numbers_count_up_per_track():
    assert lab.start("editor", "h", "c", 10) == 1
    assert lab.start("editor", "h", "c", 10) == 2
    assert lab.start("baseline", "h", "c", 10) == 1
    assert lab.used("editor") == 2


def test_finish_does_not_consume_budget():
    number = lab.start("editor", "h", "c", 10)
    lab.finish("editor", number, {"x": 1})
    assert lab.used("editor") == 1


def test_item_limit():
    with pytest.raises(RuntimeError):
        lab.start("editor", "h", "c", 513)


def test_budget_exhausted():
    for _ in range(10):
        lab.start("baseline", "h", "c", 1)
    with pytest.raises(RuntimeError):
        lab.start("baseline", "h", "c", 1)
