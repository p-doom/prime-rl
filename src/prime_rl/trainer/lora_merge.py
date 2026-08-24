from __future__ import annotations

import math
import re
from collections.abc import Mapping

import torch
from torch import Tensor

_LINEAR_LORA_KEY = re.compile(r"^(?P<prefix>.+)\.lora_(?P<factor>[AB])\.0$")


def merge_lora_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    expected_rank: int,
    scaling: float,
) -> dict[str, Tensor]:
    """Merge run zero from PrimeRL's linear MultiLoRA state-dict schema."""
    if isinstance(expected_rank, bool) or not isinstance(expected_rank, int) or expected_rank < 1:
        raise ValueError("expected LoRA rank must be a positive integer")
    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        raise ValueError("LoRA scaling must be a real number")
    if not math.isfinite(scaling) or scaling < 0:
        raise ValueError("LoRA scaling must be finite and non-negative")

    factors: dict[str, dict[str, str]] = {}
    factor_keys: set[str] = set()
    for key in state_dict:
        if "lora_" not in key:
            continue
        match = _LINEAR_LORA_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"unsupported LoRA key in weight export: {key!r}")
        prefix = match.group("prefix")
        factor = match.group("factor")
        factors.setdefault(prefix, {})[factor] = key
        factor_keys.add(key)

    if not factors:
        raise ValueError("LoRA merge requested for a state dict without linear adapters")

    for prefix, module_factors in factors.items():
        missing = {"A", "B"} - module_factors.keys()
        if missing:
            factor = next(iter(sorted(missing)))
            raise ValueError(f"missing LoRA {factor} for {prefix!r} adapter 0")

    merged = dict(state_dict)
    for prefix, selected in factors.items():
        base_key = f"{prefix}.weight"
        if base_key not in state_dict:
            raise ValueError(f"missing base weight for LoRA module {prefix!r}")
        base = state_dict[base_key]
        lora_a = state_dict[selected["A"]]
        lora_b = state_dict[selected["B"]]
        if base.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(
                f"incompatible LoRA shapes for {prefix!r}: "
                f"base={tuple(base.shape)}, A={tuple(lora_a.shape)}, "
                f"B={tuple(lora_b.shape)}"
            )
        if lora_a.shape[0] != expected_rank or lora_b.shape[1] != expected_rank:
            raise ValueError(
                f"incompatible LoRA rank for {prefix!r}: expected {expected_rank}, "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            )
        if lora_b.shape[1] != lora_a.shape[0] or base.shape != (lora_b.shape[0], lora_a.shape[1]):
            raise ValueError(
                f"incompatible LoRA shapes for {prefix!r}: "
                f"base={tuple(base.shape)}, A={tuple(lora_a.shape)}, "
                f"B={tuple(lora_b.shape)}"
            )
        if not all(torch.is_floating_point(tensor) for tensor in (base, lora_a, lora_b)):
            raise ValueError(f"LoRA weights for {prefix!r} must be floating point")
        if lora_a.dtype != base.dtype or lora_b.dtype != base.dtype:
            raise ValueError(
                f"incompatible LoRA dtypes for {prefix!r}: base={base.dtype}, A={lora_a.dtype}, B={lora_b.dtype}"
            )
        if lora_a.device != base.device or lora_b.device != base.device:
            raise ValueError(
                f"incompatible LoRA devices for {prefix!r}: base={base.device}, A={lora_a.device}, B={lora_b.device}"
            )
        delta = lora_b.to(torch.float32) @ lora_a.to(torch.float32)
        merged[base_key] = (base.to(torch.float32) + float(scaling) * delta).to(base.dtype)

    return {key: value for key, value in merged.items() if key not in factor_keys}
