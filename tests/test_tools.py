import torch
from torch import nn

from baselines.run_b1 import choose_layer, split_validation
from editor.tools import projection_delta, with_deltas


def test_projection_removes_direction_from_outputs():
    torch.manual_seed(0)
    weight = torch.randn(8, 20)
    direction = torch.nn.functional.normalize(torch.randn(8), dim=0)
    delta = projection_delta(weight, direction)
    assert (direction @ (weight + delta)).abs().max() < 1e-5
    assert torch.linalg.matrix_rank(delta) == 1


def test_with_deltas_restores_bit_identically():
    model = nn.Linear(4, 4, bias=False)
    original = model.weight.data.clone()
    with with_deltas(model, {"weight": torch.full((4, 4), 0.1)}):
        assert not torch.equal(model.weight.data, original)
    assert torch.equal(model.weight.data, original)


def test_with_deltas_restores_after_exception():
    model = nn.Linear(4, 4, bias=False)
    original = model.weight.data.clone()
    try:
        with with_deltas(model, {"weight": torch.ones(4, 4)}):
            raise RuntimeError
    except RuntimeError:
        pass
    assert torch.equal(model.weight.data, original)


def test_validation_split_is_disjoint_and_complete():
    fit, validation = split_validation(999)
    assert len(validation) == 199 and not set(fit) & set(validation) and len(fit) + len(validation) == 999


def test_choose_layer_respects_c2_tolerance():
    rows = [{"layer": 1, "C1_flip": {"rate": 0.1}, "C2_acc": {"rate": 0.5}},
            {"layer": 2, "C1_flip": {"rate": 0.4}, "C2_acc": {"rate": 0.99}}]
    assert choose_layer(rows, 1.0)["layer"] == 2
    assert choose_layer(rows[:1], 1.0)["layer"] == 1
