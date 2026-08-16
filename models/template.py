"""Copy this file to add your own backbone to the benchmark.

Steps:
    1. Copy this file to ``models/my_backbone.py`` (or anywhere importable).
    2. Load your pretrained encoder in ``__init__`` and set the three attributes.
    3. Implement ``forward_features`` to return ``[B, T, D]`` (or ``[B, D]``).
    4. Register one entry per audio format your model supports.
    5. Make sure the module is imported before you run (e.g. add it to
       ``models/__init__.py``), then evaluate with ``--backbone my_encoder``.

The probe freezes the backbone, mean-pools the features (override ``aggregate`` to
change that), normalizes, and trains a linear head. You only provide features.
"""
from __future__ import annotations

import torch

from models.base import BackboneWrapper
from models.registry import register


class MyBackbone(BackboneWrapper):
    def __init__(self, checkpoint: str | None = None, audio_format: str = "binaural"):
        super().__init__()
        # 1. Declare the audio your encoder expects.
        self.sample_rate = 24000        # Hz
        self.audio_format = audio_format  # "stereo" | "binaural" | "foa"
        self.output_dim = 768           # embedding dimensionality D

        # 2. Build/load your frozen encoder. self.in_channels is derived from
        #    audio_format (2 for stereo/binaural, 4 for foa).
        # self.encoder = load_my_encoder(checkpoint, in_channels=self.in_channels)
        raise NotImplementedError("Instantiate your encoder here, then delete this line.")

    def forward_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode ``audio`` [B, C, T] into features [B, T, D] (or [B, D])."""
        # return self.encoder(audio)
        raise NotImplementedError


# 3. Register one factory per supported format. The name is what you pass to
#    --backbone on the command line.
@register("my_encoder")
def _build_my_encoder() -> BackboneWrapper:
    return MyBackbone(checkpoint="/path/to/weights.pt", audio_format="binaural")
