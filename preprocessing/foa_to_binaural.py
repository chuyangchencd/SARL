"""Decode first-order Ambisonics (FOA) to binaural with an HRTF.

Used to render the binaural ambient noise from the FOA ambient (the original paper
rendered this step with a proprietary VST, which can't be redistributed; this is an
open equivalent built on a measured HRTF, e.g. FABIAN).

Method: a virtual-loudspeaker binaural decode. The soundfield is decoded to a set of
virtual loudspeakers spread over the sphere, each convolved with the HRTF for its
direction, and summed at the two ears. Because the ear signals are linear in the four
FOA channels, the whole loudspeaker sum collapses into eight fixed FIR filters
(left/right x W/X/Y/Z), so decoding a clip is just eight convolutions.

FOA channel convention assumed: ``[W, X, Y, Z]`` (W omni; X front, Y left, Z up).
If your FOA uses ACN ordering ``[W, Y, Z, X]`` (as TAU-SNoise does), reorder before
calling — see ``ACN_TO_WXYZ`` and its use in ``preprocess_ambient``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import fftconvolve, resample

SPEED_OF_SOUND = 343.0

# TAU-SNoise FOA is ACN-ordered [W, Y, Z, X]; index it with this to get [W, X, Y, Z].
# (Verified empirically: ACN reordering matches the original VST binaural at corr ~1.00.)
ACN_TO_WXYZ = [0, 3, 1, 2]


# --- HRTF loading (SOFA SimpleFreeFieldHRIR, e.g. FABIAN) ------------------

def load_hrtf(path: str) -> Dict:
    """Load a SOFA HRIR file into arrays of left/right IRs and their directions."""
    if not path.lower().endswith(".sofa"):
        raise ValueError(f"HRTF must be a .sofa file, got {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HRTF file not found: {path}")
    import sofar
    obj = sofar.read_sofa(path)
    ir = np.asarray(obj.Data_IR)                 # (M, 2, N)
    pos = np.asarray(obj.SourcePosition)         # (M, 3): az deg, el deg, radius m
    return {
        "left_ir": ir[:, 0, :],
        "right_ir": ir[:, 1, :],
        "azimuth": pos[:, 0].flatten(),
        "elevation": pos[:, 1].flatten(),
        "sample_rate": int(np.asarray(obj.Data_SamplingRate).flat[0]),
        "distance": float(np.median(pos[:, 2].flatten())),
    }


def _dedistance(ir: np.ndarray, sr: int, distance: float) -> np.ndarray:
    """Remove free-field propagation (delay + 1/R) so only the head filter remains."""
    n = len(ir)
    if n == 0:
        return ir
    H = np.fft.rfft(ir)
    k = np.arange(len(H), dtype=np.float64)
    H = H * distance * np.exp(2j * np.pi * k * (distance / SPEED_OF_SOUND * sr) / n)
    return np.fft.irfft(H, n=n).real.astype(np.float64)


def _hrir(hrtf: Dict, az_deg: float, el_deg: float, target_sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-measurement HRIR pair for a direction, resampled + distance-normalized."""
    a = np.degrees(np.arctan2(np.sin(np.radians(az_deg)), np.cos(np.radians(az_deg))))
    idx = int(np.argmin((hrtf["azimuth"] - a) ** 2 + (hrtf["elevation"] - el_deg) ** 2))
    h_l = hrtf["left_ir"][idx].astype(np.float64)
    h_r = hrtf["right_ir"][idx].astype(np.float64)
    sr = hrtf["sample_rate"]
    if sr != target_sr:
        n_new = int(round(len(h_l) * target_sr / sr))
        h_l, h_r = resample(h_l, n_new), resample(h_r, n_new)
    d = hrtf["distance"]
    return _dedistance(h_l, target_sr, d), _dedistance(h_r, target_sr, d)


# --- virtual-loudspeaker decode --------------------------------------------

def _speaker_directions() -> List[Tuple[float, float]]:
    """A spread of virtual loudspeaker directions (lat/long grid + poles)."""
    dirs = [(az, el) for el in (-60, -30, 0, 30, 60) for az in range(0, 360, 30)]
    return dirs + [(0.0, 90.0), (0.0, -90.0)]


def build_decode_filters(hrtf: Dict, target_sr: int, w_gain: float = 1.0):
    """Build the 8 FIR filters that turn ``[W,X,Y,Z]`` into binaural ``[L,R]``.

    Returns ``(left_filters, right_filters)``, each ``[4, N_taps]`` (rows W,X,Y,Z).
    """
    dirs = _speaker_directions()
    n = len(dirs)
    # Filter length = resampled HRIR length; probe one.
    h0_l, _ = _hrir(hrtf, 0.0, 0.0, target_sr)
    taps = len(h0_l)
    left = np.zeros((4, taps)); right = np.zeros((4, taps))
    for az, el in dirs:
        r = np.radians([az, el])
        x = np.cos(r[1]) * np.cos(r[0])   # front
        y = np.cos(r[1]) * np.sin(r[0])   # left
        z = np.sin(r[1])                  # up
        coeff = np.array([w_gain, x, y, z]) / n   # basic first-order decode
        h_l, h_r = _hrir(hrtf, az, el, target_sr)
        left += coeff[:, None] * h_l[None, :]
        right += coeff[:, None] * h_r[None, :]
    return left, right


def foa_to_binaural(foa: np.ndarray, left_filters: np.ndarray,
                    right_filters: np.ndarray) -> np.ndarray:
    """Decode FOA ``[4, T]`` (W,X,Y,Z) to binaural ``[2, T]`` with prebuilt filters."""
    if foa.shape[0] != 4:
        raise ValueError(f"FOA must be [4, T], got {foa.shape}")
    T = foa.shape[1]
    left = sum(fftconvolve(foa[c], left_filters[c])[:T] for c in range(4))
    right = sum(fftconvolve(foa[c], right_filters[c])[:T] for c in range(4))
    return np.stack([left, right], axis=0).astype(np.float32)
