"""On-the-fly synthesis of spatial audio scenes for probing.

A *scene* is one dry source clip convolved with a room impulse response (RIR) and
mixed with an ambient noise segment. Scenes are generated lazily from pools of
source clips, RIRs, and ambient recordings using a seeded random number generator,
so a given ``(seed, epoch, index)`` always yields the same scene without any
scene list being stored on disk.

Expected data layout under ``data_root``::

    audio/<split>/<class>/*.wav          dry mono source clips (class = event label)
    rir_source/<split>/<format>/*.npy    source-task RIRs ([C, N, T])
    rir_source/<split>/metadata.json     one per split, shared by all formats
    rir_room/<split>/<format>/*.wav      room-task RIRs   ([C, T])
    rir_room/<split>/metadata.json       one per split, shared by all formats
    ambient/<split>/<format>/*.wav       ambient noise segments

where ``<format>`` names the stored capture format:

    mic       tetrahedral mic array (4ch), downmixed to stereo for stereo models
    binaural  binaural (2ch)
    foa       first-order Ambisonics (4ch)

A backbone's ``audio_format`` maps to these dirs: stereo -> mic, binaural -> binaural,
foa -> foa (see ``FORMAT_DIR``).

Scenes are single-source, matching the benchmark's controlled setting. Each item
carries one raw label per requested task, keyed by task name (e.g. ``"azimuth"``);
binning, soft labels, and one-hot encoding are the probe's responsibility, not the
loader's.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

from tasks import Family, get_task, resolve_tasks

# Audio formats a backbone can request.
AUDIO_FORMATS = ("stereo", "binaural", "foa")
# Audio format -> on-disk subdirectory name, shared by rir_source/, rir_room/, and
# ambient/. The stereo format is stored as the 4-channel mic (tetrahedral) capture
# and downmixed to stereo on load.
FORMAT_DIR = {"stereo": "mic", "binaural": "binaural", "foa": "foa"}

RIR_ROOT_SOURCE = "rir_source"   # source tasks: ray-traced RIRs stored as .npy
RIR_ROOT_ROOM = "rir_room"       # room tasks: shoebox RIRs stored as .wav
_RIR_LOAD_SR = 24000             # sample rate of the .npy source RIRs

TARGET_DBFS = -24.0           # RMS normalization target for dry source clips


# =============================================================================
# Audio loading helpers
# =============================================================================

def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x if torch.is_floating_point(x) else x.float()


def normalize_dbfs(wav: torch.Tensor, target_dbfs: float = TARGET_DBFS,
                   eps: float = 1e-8) -> torch.Tensor:
    """Scale a waveform to a target RMS level in dBFS."""
    wav = _to_float(wav)
    rms = torch.sqrt(torch.mean(wav * wav) + eps)
    if rms <= eps:
        return wav.contiguous()
    gain_db = target_dbfs - 20.0 * torch.log10(rms)
    gain = torch.pow(torch.tensor(10.0, dtype=wav.dtype), gain_db / 20.0)
    return (wav * gain).contiguous()


def load_mono(path: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file as a normalized mono waveform ``[T]``."""
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.dim() != 2:
        raise ValueError(f"Expected [C, T], got {tuple(wav.shape)} from {path}")
    if wav.size(0) != 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = normalize_dbfs(_to_float(wav.squeeze(0)))
    return wav.contiguous(), int(sr)


def resample(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return AF.resample(wav, sr, target_sr).contiguous()


def pad_or_trim(x: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Pad with zeros or trim ``x`` (last dim) to exactly ``num_samples``."""
    T = x.shape[-1]
    if T == num_samples:
        return x.contiguous()
    if T > num_samples:
        return x[..., :num_samples].contiguous()
    return F.pad(x, (0, num_samples - T)).contiguous()


def load_rir_npy(path: str, index: int) -> Tuple[torch.Tensor, int]:
    """Load one position from a source RIR file ``[C, N, T]`` -> ``[C, T]``."""
    rir = np.load(path, allow_pickle=False)
    if rir.ndim != 3:
        raise ValueError(f"Expected RIR [C, N, T], got {rir.shape} from {path}")
    N = rir.shape[1]
    if not 0 <= index < N:
        raise IndexError(f"RIR position {index} out of range (N={N}) in {path}")
    return torch.from_numpy(rir[:, index, :]).float().contiguous(), _RIR_LOAD_SR


def load_rir_wav(path: str) -> Tuple[torch.Tensor, int]:
    """Load a room RIR stored as a ``[C, T]`` wav."""
    rir, sr = torchaudio.load(path)
    return _to_float(rir).contiguous(), int(sr)


# =============================================================================
# Signal processing
# =============================================================================

def convolve_rir(src: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Convolve each source with its RIR via FFT.

    Args:
        src: ``[B, T]`` dry source waveforms.
        rir: ``[B, C, T_rir]`` room impulse responses.

    Returns:
        ``[B, C, T]`` reverberant multichannel audio, trimmed to the source length.
    """
    B, T = src.shape
    _, C, T_rir = rir.shape
    n_fft = 1 << (T + T_rir - 2).bit_length()
    X = torch.fft.rfft(F.pad(src, (0, n_fft - T)), n_fft, dim=-1)       # [B, F]
    H = torch.fft.rfft(F.pad(rir, (0, n_fft - T_rir)), n_fft, dim=-1)   # [B, C, F]
    y = torch.fft.irfft(X.unsqueeze(1) * H, n_fft, dim=-1)[..., :T]      # [B, C, T]
    return y.contiguous()


def mic_to_stereo(mic: torch.Tensor) -> torch.Tensor:
    """Downmix a tetrahedral mic array ``[B, 4, T]`` to stereo ``[B, 2, T]``."""
    if mic.ndim != 3 or mic.shape[1] != 4:
        raise ValueError(f"mic must be [B, 4, T], got {tuple(mic.shape)}")
    w = mic.mean(dim=1, keepdim=True)                                    # omni
    x = 0.5 * (mic[:, 0:1] + mic[:, 1:2] - mic[:, 2:3] - mic[:, 3:4])    # L-R dipole
    return torch.cat([w + x, w - x], dim=1).contiguous()


def load_ambient(ambient_root: str, filename: str, audio_format: str) -> Tuple[torch.Tensor, int]:
    """Load an ambient noise segment ``[C, T]`` for the given format."""
    subdir = FORMAT_DIR[audio_format]
    path = os.path.join(ambient_root, subdir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ambient file not found: {path}")
    noise, sr = torchaudio.load(path)
    noise = _to_float(noise)
    if audio_format == "stereo":
        if noise.shape[0] != 4:
            raise ValueError(f"Expected 4-channel mic ambient, got {noise.shape[0]} in {path}")
        noise = mic_to_stereo(noise.unsqueeze(0)).squeeze(0)
    return noise.contiguous(), int(sr)


def batch_process(src: torch.Tensor, rir: torch.Tensor, audio_format: str,
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Render a batch of scenes to model-ready audio ``[B, C, T]``.

    Convolves each source with its RIR, downmixes to stereo if requested, and adds
    the ambient segment when provided.
    """
    y = convolve_rir(src, rir)
    if audio_format == "stereo":
        y = mic_to_stereo(y)
    if noise is not None:
        y = y + noise.to(y.device, y.dtype)
    return y


# =============================================================================
# Pools and metadata
# =============================================================================

def _build_source_pool(audio_root: Path, split: str,
                       class_names: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    """Scan ``audio/<split>/<class>/*.wav``; return clip paths and their class ids.

    Clips are visited class-by-class in ``class_names`` order, then by sorted
    filename, so the pool order is deterministic across machines.
    """
    split_dir = audio_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Audio split not found: {split_dir}")
    paths: List[str] = []
    path_to_class: Dict[str, int] = {}
    for class_id, name in enumerate(class_names):
        class_dir = split_dir / name
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.rglob("*.wav")):
            sp = str(p.resolve())
            paths.append(sp)
            path_to_class[sp] = class_id
    if not paths:
        raise FileNotFoundError(f"No source clips found under {split_dir}")
    return paths, path_to_class


def _find_rir_metadata(rir_split_dir: Path) -> Path:
    """Locate a split's single RIR metadata file (shared across all formats)."""
    path = rir_split_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"RIR metadata not found: {path}")
    return path


def _load_rir_entries(meta_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(meta_path.read_text())
    return data.get("simulations", data) if isinstance(data, dict) else data


def _list_ambient(ambient_root: Path, split: str) -> List[str]:
    # List by the foa/ names; the same filename must exist under whichever format
    # dir the backbone requests (the builder writes shared stems across formats).
    foa_dir = ambient_root / split / "foa"
    if not foa_dir.exists():
        return []
    return [f.name for f in sorted(foa_dir.glob("*.wav"))]


# =============================================================================
# Scene generation (seeded, reproducible)
# =============================================================================
#
# The random draw sequence below is fixed so that a given (seed, epoch, index)
# reproduces an identical scene: source clip, RIR, source position, ambient.

def _generate_scene(rng: random.Random, source_paths: List[str],
                    rir_entries: List[Dict[str, Any]], ambient_files: List[str],
                    task_names: Sequence[str], path_to_class: Dict[str, int],
                    family: Family) -> Dict[str, Any]:
    """Draw one scene's ingredients and its raw per-task labels."""
    _ = rng.randint(1, 1)                       # source count (fixed to 1)
    src_path = rng.sample(source_paths, 1)[0]
    rir_entry = rng.choice(rir_entries)
    coords = rir_entry.get("coordinates", [])
    if not coords:
        raise ValueError("RIR entry has no coordinates")
    pos = rng.sample(range(len(coords)), 1)[0]
    ambient = rng.choice(ambient_files) if ambient_files else None

    labels: Dict[str, float] = {}
    if family is Family.SOURCE:
        coord = coords[pos]
        if "azimuth" in task_names:
            labels["azimuth"] = float(coord.get("azimuth", 0.0))
        if "elevation" in task_names:
            labels["elevation"] = float(coord.get("elevation", 0.0))
        if "distance" in task_names:
            labels["distance"] = float(coord.get("distance", 1.0))
        if "event" in task_names:
            labels["event"] = float(path_to_class[src_path])
    else:
        if "rt60" in task_names:
            labels["rt60"] = float(rir_entry["rt60"])
        if "volume" in task_names:
            labels["volume"] = float(rir_entry["volume"])
        if "shape" in task_names:
            labels["shape"] = float(int(rir_entry["shape"]))

    return {
        "src_path": src_path,
        "rir_fname": rir_entry["fname"],
        "rir_pos": pos,
        "ambient": ambient,
        "labels": labels,
    }


# =============================================================================
# Dataset
# =============================================================================

class SpatialSceneDataset(Dataset):
    """Lazily synthesized single-source spatial scenes.

    Each item is ``(src, rir, labels, noise)`` where ``src`` is ``[T]``, ``rir``
    is ``[C, T_rir]``, ``labels`` maps each requested task name to a scalar tensor
    holding its aligned raw value, and ``noise`` is ``[C, T]`` or ``None``.

    Reproducibility: ``(seed, epoch, index)`` fully determines a scene. Call
    :meth:`set_epoch` at the start of each epoch to vary scenes across epochs.
    """

    def __init__(self, data_root: str, split: str, tasks: Sequence[str],
                 audio_format: str, duration: float, sample_rate: int,
                 seed: int, size: int):
        if audio_format not in AUDIO_FORMATS:
            raise ValueError(f"Unknown audio_format {audio_format!r}; "
                             f"expected one of {list(AUDIO_FORMATS)}")
        self.specs = resolve_tasks(tasks)             # validates no family mixing
        self.task_names = [s.name for s in self.specs]
        self.family = self.specs[0].family
        self.data_root = Path(data_root)
        self.split = split
        self.audio_format = audio_format
        self.sample_rate = int(sample_rate)
        self.num_samples = int(round(float(duration) * self.sample_rate))
        self.seed = int(seed)
        self.size = int(size)
        self._epoch = 0

        rir_root_name = RIR_ROOT_ROOM if self.family is Family.ROOM else RIR_ROOT_SOURCE
        self.rir_dir = self.data_root / rir_root_name / split
        self.ambient_dir = self.data_root / "ambient" / split

        class_names = get_task("event").class_names
        self.source_paths, self.path_to_class = _build_source_pool(
            self.data_root / "audio", split, class_names)
        entries = _load_rir_entries(_find_rir_metadata(self.rir_dir))
        self.rir_entries = self._filter_rir_entries(entries)
        if not self.rir_entries:
            raise ValueError(f"No usable RIR entries under {self.rir_dir}")
        self.ambient_files = _list_ambient(self.data_root / "ambient", split)

    def _filter_rir_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep entries that carry the metadata every requested task needs."""
        keep = []
        for e in entries:
            if not e.get("coordinates"):
                continue
            if self.family is Family.ROOM:
                if e.get("rt60") is None or e.get("volume") is None:
                    continue
                if "shape" in self.task_names and e.get("shape") is None:
                    continue
            keep.append(e)
        return keep

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.size

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self._epoch * 1_000_000 + index)

    def __getitem__(self, index: int):
        scene = _generate_scene(
            self._rng(index), self.source_paths, self.rir_entries,
            self.ambient_files, self.task_names, self.path_to_class, self.family)

        # Source waveform.
        src, sr = load_mono(scene["src_path"])
        src = pad_or_trim(resample(src, sr, self.sample_rate), self.num_samples)

        # RIR (format is the subdirectory name).
        fmt_dir = self.rir_dir / FORMAT_DIR[self.audio_format]
        if self.family is Family.ROOM:
            rir, rir_sr = load_rir_wav(str(fmt_dir / scene["rir_fname"]))
        else:
            rir, rir_sr = load_rir_npy(str(fmt_dir / scene["rir_fname"]),
                                       scene["rir_pos"])
        rir = resample(rir, rir_sr, self.sample_rate)

        # Ambient noise (optional; failures degrade to silence).
        noise = None
        if scene["ambient"]:
            try:
                noise, nsr = load_ambient(str(self.ambient_dir), scene["ambient"],
                                          self.audio_format)
                noise = pad_or_trim(resample(noise, nsr, self.sample_rate), self.num_samples)
            except Exception as err:  # noqa: BLE001 - never fail a batch over ambient
                print(f"Warning: ambient load failed for scene {index}: {err}")
                noise = None

        # Aligned raw labels.
        labels = {
            name: torch.tensor(get_task(name).align(scene["labels"][name]),
                               dtype=torch.float32)
            for name in self.task_names if name in scene["labels"]
        }
        return src, rir, labels, noise


# =============================================================================
# Collate + loader
# =============================================================================

def collate_scenes(batch):
    """Stack scenes into ``(src [B,T], rir [B,C,T_rir], labels {name:[B]}, noise [B,C,T]|None)``."""
    srcs, rirs, label_dicts, noises = zip(*batch)
    B = len(batch)
    T = srcs[0].shape[-1]
    C = rirs[0].shape[0]
    T_rir = max(r.shape[-1] for r in rirs)

    src_b = torch.stack(srcs, dim=0).contiguous()
    rir_b = torch.zeros(B, C, T_rir, dtype=rirs[0].dtype)
    for i, r in enumerate(rirs):
        rir_b[i, :, : r.shape[-1]] = r

    labels_b = {k: torch.stack([d[k] for d in label_dicts], dim=0).contiguous()
                for k in label_dicts[0]}

    noise_b = None
    if any(n is not None for n in noises):
        ref = next(n for n in noises if n is not None)
        noise_b = torch.zeros(B, ref.shape[0], T, dtype=ref.dtype)
        for i, n in enumerate(noises):
            if n is not None:
                noise_b[i] = n
        noise_b = noise_b.contiguous()

    return src_b, rir_b.contiguous(), labels_b, noise_b


def make_loader(data_root: str, split: str, tasks: Sequence[str], audio_format: str,
                duration: float, sample_rate: int, batch_size: int, seed: int, size: int,
                num_workers: int = 4, shuffle: bool = True, pin_memory: bool = True,
                persistent_workers: bool = True) -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` of spatial scenes.

    The shuffle order is seeded so every backbone sees identical scenes in an
    identical order. Call ``loader.dataset.set_epoch(epoch)`` each epoch for
    per-epoch variation.
    """
    ds = SpatialSceneDataset(data_root, split, tasks, audio_format, duration,
                             sample_rate, seed, size)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        collate_fn=collate_scenes,
    )


# =============================================================================
# Self-test: python -m data.dataset   (uses SARL_DATA_ROOT or the shared copy)
# =============================================================================

if __name__ == "__main__":
    data_root = os.environ.get("SARL_DATA_ROOT", "/scratch/cc6946/shared/SARL/data")
    fmt, dur, sr = "foa", 10.0, 24000
    passed = []

    def check(name, fn):
        try:
            fn()
            passed.append((name, True))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            passed.append((name, False))

    def source_family():
        ds = SpatialSceneDataset(data_root, "train", ["azimuth", "elevation", "distance", "event"],
                                 fmt, dur, sr, seed=42, size=4)
        src, rir, labels, noise = ds[0]
        assert src.shape == (ds.num_samples,), src.shape
        assert rir.dim() == 2
        assert set(labels) == {"azimuth", "elevation", "distance", "event"}
        assert 0.0 <= float(labels["azimuth"]) < 360.0
        assert -60.0 <= float(labels["elevation"]) <= 60.0
        # Reproducibility: same (seed, index) -> same label.
        assert float(ds[0][2]["azimuth"]) == float(ds[0][2]["azimuth"])
        src_b, rir_b, lb, nb = collate_scenes([ds[0], ds[1]])
        assert src_b.shape == (2, ds.num_samples)
        assert lb["azimuth"].shape == (2,)
        y = batch_process(src_b, rir_b, fmt, nb)
        assert y.shape[0] == 2 and y.dim() == 3

    def room_family():
        ds = SpatialSceneDataset(data_root, "train", ["rt60", "volume", "shape"],
                                 fmt, dur, sr, seed=43, size=4)
        _, _, labels, _ = ds[0]
        assert set(labels) == {"rt60", "volume", "shape"}
        assert 0.1 <= float(labels["rt60"]) <= 3.0

    def no_mixing():
        try:
            SpatialSceneDataset(data_root, "train", ["azimuth", "rt60"], fmt, dur, sr, 42, 2)
        except ValueError:
            return
        raise AssertionError("mixing families should raise")

    def loader_smoke():
        loader = make_loader(data_root, "train", ["azimuth", "distance"], fmt, dur, sr,
                             batch_size=2, seed=42, size=4, num_workers=0)
        src_b, rir_b, lb, nb = next(iter(loader))
        assert src_b.shape[0] == 2 and "azimuth" in lb

    check("source family", source_family)
    check("room family", room_family)
    check("no family mixing", no_mixing)
    check("loader smoke", loader_smoke)

    for name, ok in passed:
        print(f"{name} [ok]" if ok else f"{name} [FAILED]")
    n_fail = sum(1 for _, ok in passed if not ok)
    if n_fail:
        print(f"{len(passed) - n_fail} passed, {n_fail} failed.")
        raise SystemExit(1)
    print(f"All {len(passed)} checks passed.")
