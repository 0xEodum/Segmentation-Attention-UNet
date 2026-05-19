from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .models import AttentionGate


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def iter_attention_gates(model: nn.Module) -> Iterable[tuple[str, AttentionGate]]:
    unwrapped = _unwrap_model(model)
    for name, module in unwrapped.named_modules():
        if isinstance(module, AttentionGate):
            yield name, module


def set_attention_capture(model: nn.Module, enabled: bool, *, clear: bool = True) -> None:
    for _, gate in iter_attention_gates(model):
        gate.capture_attention = enabled
        if clear:
            gate.last_attention = None


def collect_attention_maps(
    model: nn.Module,
    *,
    upsample_to: tuple[int, int] | None = None,
    detach_cpu: bool = True,
) -> dict[str, torch.Tensor]:
    maps: dict[str, torch.Tensor] = {}
    for name, gate in iter_attention_gates(model):
        if gate.last_attention is None:
            continue
        attention = gate.last_attention
        if upsample_to is not None and attention.shape[-2:] != upsample_to:
            attention = F.interpolate(attention.float(), size=upsample_to, mode="bilinear", align_corners=False)
        if detach_cpu:
            attention = attention.detach().float().cpu()
        maps[name] = attention
    return maps


def forward_with_attention(
    model: nn.Module,
    images: torch.Tensor,
    *,
    upsample_to: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    set_attention_capture(model, True, clear=True)
    logits = model(images)
    maps = collect_attention_maps(model, upsample_to=upsample_to)
    set_attention_capture(model, False, clear=True)
    return logits, maps
