import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import prime_rl.trainer.ckpt as ckpt_module
import prime_rl.trainer.weights as weights_module
from prime_rl.configs.orchestrator import LoRAConfig as RunLoRAConfig
from prime_rl.configs.orchestrator import ModelConfig as RunModelConfig
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.trainer.ckpt import WeightCheckpointManager
from prime_rl.trainer.lora_merge import merge_lora_state_dict
from prime_rl.trainer.runs import MultiRunManager


def _load_linear_lora_layer(monkeypatch):
    models_root = Path(__file__).parents[3] / "src/prime_rl/trainer/models"
    packages = {
        "prime_rl.trainer.models": models_root,
        "prime_rl.trainer.models.layers": models_root / "layers",
        "prime_rl.trainer.models.layers.lora": models_root / "layers/lora",
    }
    for name in (
        "prime_rl.trainer.models.layers.lora.base",
        "prime_rl.trainer.models.layers.lora.multi_linear",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, path in packages.items():
        package = ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    base = importlib.import_module("prime_rl.trainer.models.layers.lora.base")
    linear = importlib.import_module("prime_rl.trainer.models.layers.lora.multi_linear")
    return linear.MultiLoRALinear, base


def test_merge_lora_state_dict_merges_selected_linear_adapter_without_mutating_input(
    monkeypatch,
):
    MultiLoRALinear, lora_base = _load_linear_lora_layer(monkeypatch)
    base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)
    adapter_a = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]], dtype=torch.bfloat16)
    adapter_b = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    lora_base.set_lora_num_tokens(torch.zeros(1, dtype=torch.long), reset_reference=True)
    lora_base.set_multilora_scaling(torch.full((1,), 0.5), reset_reference=True)
    module = MultiLoRALinear(
        nn.Linear(3, 2, bias=False, dtype=torch.bfloat16),
        rank=2,
        n_adapters=1,
        alpha=1.0,
        use_grouped_mm=False,
    )
    with torch.no_grad():
        module.weight.copy_(base)
        module.lora_A[0].copy_(adapter_a)
        module.lora_B[0].copy_(adapter_b)
    state_dict = module.state_dict(prefix="model.proj.")
    before = {key: value.clone() for key, value in state_dict.items()}

    expected_keys = {
        "model.proj.weight",
        "model.proj.lora_A.0",
        "model.proj.lora_B.0",
    }
    if set(state_dict) != expected_keys:
        pytest.fail(f"unexpected emitted MultiLoRA keys: {sorted(state_dict)}")

    merged = merge_lora_state_dict(
        state_dict,
        expected_rank=2,
        scaling=0.5,
    )

    torch.testing.assert_close(
        merged["model.proj.weight"],
        (base.float() + 0.5 * (adapter_b.float() @ adapter_a.float())).bfloat16(),
    )
    if set(merged) != {"model.proj.weight"}:
        pytest.fail(f"unexpected merged keys: {sorted(merged)}")
    merged_weight = merged["model.proj.weight"]
    if merged_weight.dtype != module.weight.dtype:
        pytest.fail(f"merged dtype changed to {merged_weight.dtype}")
    if merged_weight.device != module.weight.device:
        pytest.fail(f"merged device changed to {merged_weight.device}")
    if merged_weight.shape != module.weight.shape:
        pytest.fail(f"merged shape changed to {tuple(merged_weight.shape)}")
    for key, value in state_dict.items():
        torch.testing.assert_close(value, before[key])
    torch.testing.assert_close(module.weight, base)


@pytest.mark.parametrize(
    ("state_dict", "match"),
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
                "model.proj.w1_lora_A.0": torch.zeros(1, 3),
                "model.proj.w1_lora_B.0": torch.zeros(2, 1),
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
                "model.proj.lora_A.0": torch.zeros(1, 3),
                "model.proj.lora_B.0": torch.zeros(2, 1),
                "model.other.weight": torch.zeros(2, 3),
                "model.other.lora_A.0": torch.zeros(1, 3),
                "model.other.lora_B.0": torch.zeros(2, 1),
                "model.other.lora_A.1": torch.zeros(1, 3),
                "model.other.lora_B.1": torch.zeros(2, 1),
            },
            "unsupported LoRA key",
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
def test_merge_lora_state_dict_rejects_incomplete_or_incompatible_linear_schema(
    state_dict,
    match,
):
    with pytest.raises(ValueError, match=match):
        merge_lora_state_dict(state_dict, expected_rank=1, scaling=1.0)


def test_merge_lora_state_dict_rejects_a_rank_different_from_live_metadata():
    state_dict = {
        "model.proj.weight": torch.zeros(2, 3),
        "model.proj.lora_A.0": torch.zeros(1, 3),
        "model.proj.lora_B.0": torch.zeros(2, 1),
    }

    with pytest.raises(ValueError, match="incompatible LoRA rank.*expected 2"):
        merge_lora_state_dict(state_dict, expected_rank=2, scaling=1.0)


@pytest.mark.parametrize(
    "extra",
    [
        {
            "model.proj.lora_A.1": torch.zeros(1, 3),
            "model.proj.lora_B.1": torch.zeros(2, 1),
        },
        {"model.proj.lora_C.0": torch.zeros(2, 1)},
    ],
)
def test_merge_lora_state_dict_rejects_every_non_run_zero_lora_key(extra):
    state_dict = {
        "model.proj.weight": torch.zeros(2, 3),
        "model.proj.lora_A.0": torch.zeros(1, 3),
        "model.proj.lora_B.0": torch.zeros(2, 1),
        **extra,
    }

    with pytest.raises(ValueError, match="unsupported LoRA key"):
        merge_lora_state_dict(state_dict, expected_rank=1, scaling=1.0)


class _ExportModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def is_prime_state_dict(self, state_dict):
        return True

    def convert_to_hf(self, state_dict):
        return None


def _weight_manager(tmp_path) -> WeightCheckpointManager:
    manager = object.__new__(WeightCheckpointManager)
    manager.weights_dir = tmp_path / "weights"
    manager.config = SimpleNamespace(save_adapter_separately=False)
    manager.lora_config = SimpleNamespace(rank=8, alpha=80.0)
    manager.logger = MagicMock()
    manager.world = SimpleNamespace(is_master=True)
    manager.ckpt_steps = []
    manager.save_to_path = MagicMock()
    manager.mark_stable = MagicMock()
    return manager


def _run_manager(max_runs=1):
    manager = object.__new__(MultiRunManager)
    manager.max_runs = max_runs
    manager.config = {
        0: OrchestratorConfig.model_construct(
            model=RunModelConfig.model_construct(lora=RunLoRAConfig(rank=2, alpha=6.0))
        )
    }
    return manager


def test_weight_checkpoint_uses_live_run_zero_lora_metadata(monkeypatch, tmp_path):
    manager = _weight_manager(tmp_path)
    captured = {}

    def gather(model, is_master, **kwargs):
        captured.update(kwargs)
        return {"model.weight": torch.zeros(1)}

    monkeypatch.setattr(ckpt_module, "PreTrainedModelPrimeRL", _ExportModel)
    monkeypatch.setattr(ckpt_module, "get_multi_run_manager", lambda: _run_manager())
    monkeypatch.setattr(ckpt_module, "has_lora_layers", lambda model: True)
    monkeypatch.setattr(ckpt_module, "gather_merged_lora_weights_on_master", gather)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    manager.save(3, _ExportModel(), object())

    if captured.get("expected_rank") != 2:
        pytest.fail(f"export rank was {captured.get('expected_rank')!r}")
    if captured.get("scaling") != 3.0:
        pytest.fail(f"export scaling was {captured.get('scaling')!r}")


def test_weight_checkpoint_rejects_multiple_adapter_slots_before_outputs(
    monkeypatch,
    tmp_path,
):
    manager = _weight_manager(tmp_path)
    monkeypatch.setattr(ckpt_module, "PreTrainedModelPrimeRL", _ExportModel)
    monkeypatch.setattr(ckpt_module, "get_multi_run_manager", lambda: _run_manager(max_runs=2))
    monkeypatch.setattr(ckpt_module, "has_lora_layers", lambda model: True)
    monkeypatch.setattr(
        ckpt_module,
        "gather_merged_lora_weights_on_master",
        lambda *args, **kwargs: pytest.fail("gather ran for multiple adapter slots"),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    with pytest.raises(ValueError, match="exactly one run slot"):
        manager.save(3, _ExportModel(), object())

    if manager.get_step_path(3).exists():
        pytest.fail(f"invalid export created {manager.get_step_path(3)}")


def test_weight_checkpoint_rejects_merge_schema_before_outputs(monkeypatch, tmp_path):
    manager = _weight_manager(tmp_path)
    monkeypatch.setattr(ckpt_module, "get_multi_run_manager", lambda: _run_manager())
    monkeypatch.setattr(ckpt_module, "has_lora_layers", lambda model: True)

    def reject_schema(*args, **kwargs):
        raise ValueError("unsupported LoRA key")

    monkeypatch.setattr(ckpt_module, "gather_merged_lora_weights_on_master", reject_schema)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    with pytest.raises(ValueError, match="unsupported LoRA key"):
        manager.save(3, _ExportModel(), object())

    if manager.get_step_path(3).exists():
        pytest.fail(f"invalid schema created {manager.get_step_path(3)}")


def test_lora_merge_schema_failure_is_broadcast_to_every_rank(monkeypatch):
    bad_state = {
        "model.proj.weight": torch.zeros(2, 3),
        "model.proj.lora_A.0": torch.zeros(1, 3),
    }
    shared_verdict = []
    current_is_master = [False]

    def gather(model, is_master, dtype):
        return bad_state if is_master else {}

    def broadcast(verdict, src):
        if current_is_master[0]:
            shared_verdict[:] = verdict
        else:
            verdict[:] = shared_verdict

    monkeypatch.setattr(weights_module, "_gather_state_dict_on_master", gather)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    messages = []
    for is_master in (True, False):
        current_is_master[0] = is_master
        with pytest.raises(ValueError, match="missing LoRA B") as error:
            weights_module.gather_merged_lora_weights_on_master(
                nn.Linear(1, 1),
                is_master,
                expected_rank=1,
                scaling=1.0,
            )
        messages.append(str(error.value))

    if messages[0] != messages[1]:
        pytest.fail(f"ranks received different merge errors: {messages!r}")
