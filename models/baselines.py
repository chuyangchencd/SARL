"""A weight-free reference backbone.

A worked example of the :class:`~models.base.BackboneWrapper` contract, not a result
reported in the paper. It needs no pretrained weights, so it runs anywhere and shows
how a real wrapper produces ``[B, T, D]`` features:

* :class:`RawFeatures` -- deterministic signal features (log-mel, optionally with
  interaural phase differences), a "no learning" reference.

It registers format-specific variants (e.g. ``rawfeat_logmel_binaural``) in the
backbone registry.
"""
from __future__ import annotations

import torch
import torchaudio

from models.base import BackboneWrapper
from models.registry import register


class RawFeatures(BackboneWrapper):
    """Deterministic time-frequency features with no learned parameters.

    Feature types:
        ``logmel``     -- per-channel log-mel spectrogram (any format).
        ``logmel_ipd`` -- log-mel plus cos/sin interaural phase difference
                          (two-channel stereo or binaural only).
    """

    def __init__(self, audio_format: str, feature_type: str = "logmel",
                 sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 300,
                 win_length: int = 1024, n_mels: int = 128, f_min: float = 20.0,
                 f_max: float | None = None):
        super().__init__()
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.n_mels = n_mels

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            power=None)  # complex STFT
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft // 2 + 1,
            f_min=f_min, f_max=f_max or sample_rate / 2, norm="slaney")
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        if feature_type == "logmel":
            self.output_dim = self.in_channels * n_mels
        elif feature_type == "logmel_ipd":
            if self.in_channels != 2:
                raise ValueError("logmel_ipd requires a 2-channel format")
            self.output_dim = (self.in_channels + 2) * n_mels
        else:
            raise ValueError(f"Unknown feature_type {feature_type!r}")

    def _logmel(self, stft: torch.Tensor) -> torch.Tensor:
        # stft [B, C, F, T] -> [B, C, T, M]
        mel = self.mel_scale(stft.abs() ** 2)
        return self.to_db(mel).transpose(-1, -2)

    def forward_features(self, audio: torch.Tensor) -> torch.Tensor:
        stft = self.spectrogram(audio)                 # [B, C, F, T]
        feats = self._logmel(stft)                     # [B, C, T, M]
        if self.feature_type == "logmel_ipd":
            phase = torch.atan2(stft.imag, stft.real).transpose(-1, -2)  # [B, C, T, F]
            ipd = phase[:, 1] - phase[:, 0]                              # [B, T, F]
            ipd = torch.stack([ipd.cos(), ipd.sin()], dim=1)            # [B, 2, T, F]
            ipd = torch.matmul(ipd, self.mel_scale.fb)                  # [B, 2, T, M]
            feats = torch.cat([feats, ipd], dim=1)                      # [B, C+2, T, M]
        # [B, C', T, M] -> [B, T, C'*M]
        return feats.permute(0, 2, 1, 3).flatten(2).contiguous()


# --- registry entries ------------------------------------------------------
# One variant per audio format, so a backbone name fully specifies its input.

for _fmt in ("stereo", "binaural", "foa"):
    register(f"rawfeat_logmel_{_fmt}")(
        lambda fmt=_fmt: RawFeatures(fmt, feature_type="logmel"))

for _fmt in ("stereo", "binaural"):
    register(f"rawfeat_logmel_ipd_{_fmt}")(
        lambda fmt=_fmt: RawFeatures(fmt, feature_type="logmel_ipd"))

del _fmt
