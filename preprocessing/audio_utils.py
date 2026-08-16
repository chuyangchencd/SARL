"""Audio helpers for building the source-clip pool.

These use only soundfile + numpy so preprocessing runs without PyTorch. Clips are
made mono, resampled, length-fitted to the target duration, and RMS-normalized,
matching how the benchmark expects its ``audio/`` pool.
"""
from __future__ import annotations

import math

import numpy as np
import soundfile as sf


def load_mono(path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono float32; returns ``(samples, sample_rate)``."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav, int(sr)


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample via linear interpolation."""
    if sr == target_sr:
        return wav
    n = int(len(wav) * target_sr / sr)
    x_old = np.linspace(0, len(wav) - 1, len(wav), dtype=np.float64)
    x_new = np.linspace(0, len(wav) - 1, n, dtype=np.float64)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def pad_or_trim(wav: np.ndarray, num_samples: int) -> np.ndarray:
    if len(wav) == num_samples:
        return wav
    if len(wav) > num_samples:
        return wav[:num_samples].copy()
    return np.pad(wav, (0, num_samples - len(wav)))


def repeat_to_fill(wav: np.ndarray, num_samples: int) -> np.ndarray:
    """Tile a short clip to at least ``num_samples``, then trim exactly."""
    if len(wav) >= num_samples:
        return wav[:num_samples].copy()
    reps = int(np.ceil(num_samples / max(len(wav), 1)))
    return np.tile(wav, reps)[:num_samples].copy()


def circular_shift(wav: np.ndarray, shift_samples: int) -> np.ndarray:
    """Circularly roll the waveform (positive shifts right)."""
    return wav if shift_samples == 0 else np.roll(wav, int(shift_samples))


def normalize_dbfs(wav: np.ndarray, target_dbfs: float = -24.0, eps: float = 1e-8) -> np.ndarray:
    """Scale to a target RMS level in dBFS."""
    rms = np.sqrt(np.mean(wav * wav) + eps)
    if rms <= eps:
        return wav
    gain = 10.0 ** ((target_dbfs - 20.0 * math.log10(rms)) / 20.0)
    return (wav * gain).astype(np.float32)


def is_silent(wav: np.ndarray, threshold_rms: float = 1e-5) -> bool:
    if wav.size == 0:
        return True
    return float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) < threshold_rms


def write_wav(path: str, wav: np.ndarray, sr: int) -> None:
    sf.write(path, wav, sr)
