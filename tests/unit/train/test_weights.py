from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import prime_rl.trainer.ckpt as ckpt_module
from prime_rl.trainer.ckpt import WeightCheckpointManager
from prime_rl.trainer.lora_merge import merge_lora_state_dict
from prime_rl.trainer.models.layers.lora.base import set_lora_num_tokens, set_multilora_scaling
from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear
from prime_rl.trainer.models.layers.lora.multi_moe import (
    MultiLoRAGptOssGroupedExperts,
    MultiLoRAGroupedExperts,
    MultiLoRANonGatedGroupedExperts,
)
from prime_rl.trainer.models.layers.moe import GptOssGroupedExperts, GroupedExperts, NonGatedGroupedExperts


def test_merge_lora_state_dict_merges_every_production_schema_without_mutating_input():
    set_lora_num_tokens(torch.zeros(1, dtype=torch.long), reset_reference=True)
    set_multilora_scaling(torch.ones(1), reset_reference=True)
    cases = [
        (
            MultiLoRALinear(
                nn.Linear(3, 2, bias=False),
                rank=2,
                n_adapters=1,
                use_grouped_mm=False,
            ),
            "model.proj.",
            {"model.proj.weight"},
        ),
        (
            MultiLoRAGroupedExperts(
                GroupedExperts(dim=4, hidden_dim=6, num_experts=2, use_grouped_mm=False),
                rank=2,
                n_adapters=1,
                use_grouped_mm=False,
            ),
            "model.experts.",
            {"model.experts.w1", "model.experts.w2", "model.experts.w3"},
        ),
        (
            MultiLoRANonGatedGroupedExperts(
                NonGatedGroupedExperts(input_dim=4, intermediate_dim=6, num_experts=2, use_grouped_mm=False),
                rank=2,
                n_adapters=1,
                use_grouped_mm=False,
            ),
            "model.experts.",
            {"model.experts.w1", "model.experts.w2"},
        ),
        (
            MultiLoRAGptOssGroupedExperts(
                GptOssGroupedExperts(hidden_size=4, intermediate_size=6, num_experts=2, use_grouped_mm=False),
                rank=2,
                n_adapters=1,
                use_grouped_mm=False,
            ),
            "model.experts.",
            {"model.experts.gate_up_proj", "model.experts.down_proj"},
        ),
    ]

    for module, prefix, adapted_keys in cases:
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                parameter.fill_(1.0 if "lora_" in name else 0.0)
        state_dict = module.state_dict(prefix=prefix)
        before = {key: value.clone() for key, value in state_dict.items()}
        base_schema = module.base_layer.state_dict(prefix=prefix)

        merged = merge_lora_state_dict(state_dict, scaling=0.5)

        if set(merged) != set(base_schema):
            pytest.fail(f"merged state changed the base schema: {sorted(merged)}")
        for key in adapted_keys:
            torch.testing.assert_close(merged[key], torch.ones_like(merged[key]))
        for key, value in state_dict.items():
            torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize(
    ("state_dict", "message"),
    [
        (
            {
                "model.proj.weight": torch.zeros(2, 3),
                "model.proj.lora_A.0": torch.zeros(1, 3),
            },
            "missing LoRA B",
        ),
        (
            {
                "model.proj.weight": torch.zeros(2, 3),
                "model.proj.lora_A.1": torch.zeros(1, 3),
                "model.proj.lora_B.1": torch.zeros(2, 1),
            },
            "unsupported LoRA key",
        ),
        (
            {
                "model.proj.weight": torch.zeros(2, 3),
                "model.proj.lora_A.0": torch.zeros(1, 4),
                "model.proj.lora_B.0": torch.zeros(2, 1),
            },
            "incompatible LoRA shapes",
        ),
        (
            {
                "model.proj.weight": torch.zeros(2, 3),
                "model.proj.lora_A.0": torch.zeros(1, 3, dtype=torch.float64),
                "model.proj.lora_B.0": torch.zeros(2, 1),
            },
            "incompatible LoRA dtypes",
        ),
    ],
)
def test_merge_lora_state_dict_rejects_incompatible_schemas(state_dict, message):
    with pytest.raises(ValueError, match=message):
        merge_lora_state_dict(state_dict, scaling=1.0)


@pytest.mark.parametrize("save_adapter_separately", [False, True])
def test_weight_checkpoint_always_exports_merged_lora_weights(monkeypatch, tmp_path, save_adapter_separately):
    set_lora_num_tokens(torch.zeros(1, dtype=torch.long), reset_reference=True)
    set_multilora_scaling(torch.ones(1), reset_reference=True)
    model = MultiLoRALinear(nn.Linear(3, 2, bias=False), rank=2, n_adapters=1, use_grouped_mm=False)
    manager = object.__new__(WeightCheckpointManager)
    manager.config = SimpleNamespace(save_adapter_separately=save_adapter_separately)
    manager.logger = MagicMock()
    manager.world = SimpleNamespace(is_master=True)
    manager.weights_dir = tmp_path / "weights"
    captured = {}

    monkeypatch.setattr(
        ckpt_module,
        "get_multi_run_manager",
        lambda: SimpleNamespace(max_runs=1, scaling_factors=torch.tensor([0.5], dtype=torch.bfloat16)),
    )

    def gather(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("gather complete")

    monkeypatch.setattr(ckpt_module, "gather_merged_lora_weights_on_master", gather)
    monkeypatch.setattr(
        ckpt_module,
        "gather_weights_on_master",
        lambda *args, **kwargs: pytest.fail("LoRA export used the unmerged gather"),
    )

    with pytest.raises(RuntimeError, match="gather complete"):
        manager.save(3, model, object())

    if captured.get("scaling") != 0.5:
        pytest.fail(f"export scaling was {captured.get('scaling')!r}")
    if manager.get_step_path(3).exists():
        pytest.fail("failed LoRA merge created a checkpoint directory")
