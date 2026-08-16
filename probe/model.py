"""The probe: a frozen backbone with lightweight per-task heads.

Following the benchmark protocol, the backbone is frozen and each task is decoded
by a small classifier on top of the backbone's scene embedding:

    audio -> backbone.embed -> LayerNorm -> per-task head -> logits

The default head is a single linear layer (a linear probe). An MLP head is
available as an option for exploring beyond the linear protocol.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from models.base import BackboneWrapper
from tasks import TaskSpec, resolve_tasks


class MLPHead(nn.Module):
    """Two-layer probe head: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else min(512, max(2 * in_dim, 256))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Probe(nn.Module):
    """A frozen backbone plus one trainable head per task.

    Args:
        backbone: The frozen encoder to probe.
        tasks: Task names to attach heads for (must share one family).
        use_mlp_head: Use an :class:`MLPHead` instead of a linear head.
        head_hidden_dim: Hidden width for MLP heads (default: sized to the input).
        head_dropout: Dropout for MLP heads.
    """

    def __init__(self, backbone: BackboneWrapper, tasks: Sequence[str],
                 use_mlp_head: bool = False, head_hidden_dim: Optional[int] = None,
                 head_dropout: float = 0.1):
        super().__init__()
        self.specs: list[TaskSpec] = resolve_tasks(tasks)
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        dim = int(backbone.output_dim)
        # Non-affine normalization of the frozen embedding before the probe.
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.heads = nn.ModuleDict()
        for spec in self.specs:
            if use_mlp_head:
                head: nn.Module = MLPHead(dim, spec.n_classes, head_hidden_dim, head_dropout)
            else:
                head = nn.Linear(dim, spec.n_classes)
            self.heads[spec.name] = head
        self._init_heads()

    def _init_heads(self, std: float = 0.02) -> None:
        for module in self.heads.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @torch.no_grad()
    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Frozen scene embedding ``[B, D]`` (normalization applied)."""
        return self.norm(self.backbone.embed(audio))

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return ``{task_name: logits [B, n_classes]}`` for every attached head."""
        emb = self.norm(self.backbone.embed(audio))
        return {name: head(emb) for name, head in self.heads.items()}

    def head_parameters(self):
        """Trainable parameters (the heads; the backbone is frozen)."""
        return self.heads.parameters()


class NaivePredictor(nn.Module):
    """A training-free reference that ignores the audio entirely.

    It emits random per-task logits, giving the floor a real probe must beat -- this
    is the baseline the aggregate scores are normalized against.

    Exposes the same ``forward(audio) -> {task: logits}`` interface as :class:`Probe`
    but has no trainable parameters, so the runner skips training for it.
    """

    trainable = False

    def __init__(self, tasks: Sequence[str], seed: int = 42):
        super().__init__()
        self.specs: list[TaskSpec] = resolve_tasks(tasks)
        self._gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = audio.shape[0]
        return {s.name: torch.randn(B, s.n_classes, generator=self._gen,
                                    dtype=torch.float32).to(audio.device)
                for s in self.specs}


# Probes that train have trainable heads; naive predictors do not.
Probe.trainable = True
