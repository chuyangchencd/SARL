"""The backbone interface every model plugs into.

To evaluate your own encoder, subclass :class:`BackboneWrapper`, declare the audio
it expects, and implement :meth:`~BackboneWrapper.forward_features`. The probe
handles everything downstream (normalization, the linear head, training, scoring).

Contract
--------
A wrapper declares three things and implements one method:

* ``sample_rate`` -- the rate, in Hz, the backbone expects its input at.
* ``audio_format`` -- ``"stereo"``, ``"binaural"`` or ``"foa"``; this fixes the
  channel count of the input (see :data:`FORMAT_CHANNELS`).
* ``output_dim`` -- the embedding dimensionality ``D``.
* ``forward_features(audio)`` -- map input audio ``[B, C, T]`` to features.

The benchmark feeds the backbone audio already at ``sample_rate`` with the channel
count implied by ``audio_format``, so a wrapper never has to resample or re-channel.

Aggregation
-----------
:meth:`forward_features` may return either frame-level features ``[B, T, D]`` or a
single embedding ``[B, D]``. :meth:`aggregate` reduces features to one ``[B, D]``
embedding per scene and **defaults to mean pooling over time**, matching the
benchmark's protocol. Override it only if your model needs a different reduction
(for example attention pooling); whether the features are temporal never concerns
the probe.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# Channel count implied by each audio format.
FORMAT_CHANNELS = {"stereo": 2, "binaural": 2, "foa": 4}


class BackboneWrapper(nn.Module, ABC):
    """Base class adapting a pretrained encoder to the benchmark.

    Subclasses must set :attr:`sample_rate`, :attr:`audio_format` and
    :attr:`output_dim`, and implement :meth:`forward_features`.
    """

    #: Sample rate, in Hz, the backbone expects its input at.
    sample_rate: int
    #: Input format: ``"stereo"``, ``"binaural"`` or ``"foa"``.
    audio_format: str
    #: Embedding dimensionality ``D`` produced per scene.
    output_dim: int

    @property
    def in_channels(self) -> int:
        """Number of input channels implied by :attr:`audio_format`."""
        try:
            return FORMAT_CHANNELS[self.audio_format]
        except KeyError:
            raise ValueError(
                f"{type(self).__name__}.audio_format is {self.audio_format!r}; "
                f"expected one of {list(FORMAT_CHANNELS)}."
            ) from None

    @abstractmethod
    def forward_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode input audio into features.

        Args:
            audio: ``[B, C, T]`` at :attr:`sample_rate`, with ``C`` equal to
                :attr:`in_channels`.

        Returns:
            Frame-level features ``[B, T, D]`` or a single embedding ``[B, D]``,
            where ``D`` is :attr:`output_dim`.
        """
        raise NotImplementedError

    def aggregate(self, features: torch.Tensor) -> torch.Tensor:
        """Reduce features to one embedding per scene ``[B, D]``.

        Defaults to mean pooling over the time axis of ``[B, T, D]`` features;
        ``[B, D]`` features are returned unchanged. Override for a different
        reduction.
        """
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(
            f"{type(self).__name__}.forward_features returned shape "
            f"{tuple(features.shape)}; expected [B, T, D] or [B, D]."
        )

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode and aggregate ``audio`` into a scene embedding ``[B, D]``."""
        emb = self.aggregate(self.forward_features(audio))
        if emb.ndim != 2:
            raise ValueError(
                f"{type(self).__name__}.aggregate returned shape {tuple(emb.shape)}; "
                f"expected [B, D]."
            )
        return emb
