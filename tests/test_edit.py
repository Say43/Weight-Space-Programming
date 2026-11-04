import pytest
import torch
from torch import nn

from edits.apply_edit import apply_edit, edit_stats, save_edit, weights_hash


def fresh():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 4, bias=False))


def test_reapply_is_bit_identical(tmp_path):
    path = tmp_path / "edit.safetensors"
    torch.manual_seed(1)
    save_edit({"0.weight": torch.randn(16, 16) * 1e-3}, path)
    first, second = fresh(), fresh()
    apply_edit(first, path)
    apply_edit(second, path)
    assert weights_hash(first, ["0.weight"]) == weights_hash(second, ["0.weight"])
    assert weights_hash(first, ["0.weight"]) != weights_hash(fresh(), ["0.weight"])


def test_edit_leaves_other_parameters_alone(tmp_path):
    path = tmp_path / "edit.safetensors"
    save_edit({"0.weight": torch.ones(16, 16)}, path)
    model = fresh()
    apply_edit(model, path)
    assert torch.equal(model[1].weight, fresh()[1].weight)


def test_embeddings_and_head_are_rejected(tmp_path):
    with pytest.raises(AssertionError):
        save_edit({"model.embed_tokens.weight": torch.ones(2, 2)}, tmp_path / "x.safetensors")
    with pytest.raises(AssertionError):
        save_edit({"lm_head.weight": torch.ones(2, 2)}, tmp_path / "y.safetensors")


def test_edit_stats_counts_rank_and_share(tmp_path):
    path = tmp_path / "edit.safetensors"
    u = torch.zeros(4, 1)
    u[0] = 1.0
    delta = u @ torch.ones(1, 8)
    save_edit({"0.weight": delta}, path)
    stats = edit_stats(path, total_params=800)
    assert stats == {"matrices": 1, "changed_params": 8, "changed_share": 0.01, "max_delta_rank": 1}
