from __future__ import annotations

import math
import re
from collections.abc import Mapping

import torch
from torch import Tensor

_LORA_KEY = re.compile(
    r"^(?P<prefix>.+)\.(?:(?P<projection>w1|w2|w3|gate_up|down)_)?lora_(?P<factor>[AB])\.0$"
)


def _base_weight(prefix: str, projection: str | None) -> tuple[str, int, bool]:
    if projection is None:
        return f"{prefix}.weight", 2, False
    if projection in {"w1", "w2", "w3"}:
        return f"{prefix}.{projection}", 3, False
    if projection == "gate_up":
        return f"{prefix}.gate_up_proj", 3, True
    if projection == "down":
        return f"{prefix}.down_proj", 3, True
    raise ValueError(f"unsupported LoRA projection {projection!r}")


def merge_lora_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    expected_rank: int,
    scaling: float,
) -> dict[str, Tensor]:
    """Merge run zero from PrimeRL's MultiLoRA state-dict schemas."""
    if isinstance(expected_rank, bool) or not isinstance(expected_rank, int) or expected_rank < 1:
        raise ValueError("expected LoRA rank must be a positive integer")
    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        raise ValueError("LoRA scaling must be a real number")
    if not math.isfinite(scaling) or scaling < 0:
        raise ValueError("LoRA scaling must be finite and non-negative")

    factors: dict[str, tuple[int, bool, dict[str, str]]] = {}
    factor_keys: set[str] = set()
    for key in state_dict:
        if "lora_" not in key:
            continue
        match = _LORA_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"unsupported LoRA key in weight export: {key!r}")
        base_key, ndim, transposed = _base_weight(match.group("prefix"), match.group("projection"))
        factor = match.group("factor")
        stored_ndim, stored_transposed, selected = factors.setdefault(base_key, (ndim, transposed, {}))
        if stored_ndim != ndim or stored_transposed != transposed or factor in selected:
            raise ValueError(f"ambiguous LoRA {factor} for base weight {base_key!r}")
        selected[factor] = key
        factor_keys.add(key)

    if not factors:
        raise ValueError("LoRA merge requested for a state dict without supported adapters")

    for base_key, (_, _, selected) in factors.items():
        missing = {"A", "B"} - selected.keys()
        if missing:
            factor = next(iter(sorted(missing)))
            raise ValueError(f"missing LoRA {factor} for {base_key!r} adapter 0")

    merged = dict(state_dict)
    for base_key, (ndim, transposed, selected) in factors.items():
        if base_key not in state_dict:
            raise ValueError(f"missing base weight for LoRA target {base_key!r}")
        base = state_dict[base_key]
        lora_a = state_dict[selected["A"]]
        lora_b = state_dict[selected["B"]]
        if base.ndim != ndim or lora_a.ndim != ndim or lora_b.ndim != ndim:
            raise ValueError(
                f"incompatible LoRA shapes for {base_key!r}: "
                f"base={tuple(base.shape)}, A={tuple(lora_a.shape)}, "
                f"B={tuple(lora_b.shape)}"
            )
        if lora_a.shape[-2] != expected_rank or lora_b.shape[-1] != expected_rank:
            raise ValueError(
                f"incompatible LoRA rank for {base_key!r}: expected {expected_rank}, "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            )
        delta_shape = (*lora_b.shape[:-2], lora_b.shape[-2], lora_a.shape[-1])
        if lora_b.shape[:-2] != lora_a.shape[:-2] or lora_b.shape[-1] != lora_a.shape[-2]:
            raise ValueError(
                f"incompatible LoRA shapes for {base_key!r}: "
                f"base={tuple(base.shape)}, A={tuple(lora_a.shape)}, "
                f"B={tuple(lora_b.shape)}"
            )
        if transposed:
            delta_shape = (*delta_shape[:-2], delta_shape[-1], delta_shape[-2])
        if base.shape != delta_shape:
            raise ValueError(
                f"incompatible LoRA shapes for {base_key!r}: "
                f"base={tuple(base.shape)}, A={tuple(lora_a.shape)}, "
                f"B={tuple(lora_b.shape)}"
            )
        if not all(torch.is_floating_point(tensor) for tensor in (base, lora_a, lora_b)):
            raise ValueError(f"LoRA weights for {base_key!r} must be floating point")
        if lora_a.dtype != base.dtype or lora_b.dtype != base.dtype:
            raise ValueError(
                f"incompatible LoRA dtypes for {base_key!r}: "
                f"base={base.dtype}, A={lora_a.dtype}, B={lora_b.dtype}"
            )
        if lora_a.device != base.device or lora_b.device != base.device:
            raise ValueError(
                f"incompatible LoRA devices for {base_key!r}: "
                f"base={base.device}, A={lora_a.device}, B={lora_b.device}"
            )
        delta = torch.matmul(lora_b.to(torch.float32), lora_a.to(torch.float32))
        if transposed:
            delta = delta.transpose(-2, -1)
        merged[base_key] = (base.to(torch.float32) + float(scaling) * delta).to(base.dtype)

    return {key: value for key, value in merged.items() if key not in factor_keys}
