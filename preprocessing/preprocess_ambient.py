"""Preprocess TAU-SNoise ambient noise into ``ambient/<split>/{foa,mic,binaural}/``.

Download TAU-SNoise from https://zenodo.org/records/6408611, then:

    python -m preprocessing.preprocess_ambient --source /path/to/TAU-SNoise_DB \\
        --out data_root --hrtf /path/to/hrir.sofa

Each source recording is cut into non-overlapping 10 s segments in three formats:

    foa       4ch first-order Ambisonics (ACN), from ``ambience_foa*`` recordings
    mic       4ch tetrahedral,                    from ``ambience_tetra*`` recordings
    binaural  2ch, decoded from the FOA with an HRTF (see ``foa_to_binaural``)

Rooms are split alphabetically: first 2 -> val, next 2 -> test, the rest -> train.
Binaural rendering needs a SOFA HRTF (``--hrtf``), e.g. FABIAN; it reproduces the
original (VST-rendered) binaural ambient to within correlation ~1.0.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from preprocessing import audio_utils as au
from preprocessing.foa_to_binaural import (
    ACN_TO_WXYZ, build_decode_filters, foa_to_binaural, load_hrtf,
)


def split_rooms(rooms: List[Path]) -> Dict[str, List[Path]]:
    """Alphabetical split: first 2 -> val, next 2 -> test, the rest -> train."""
    rooms = sorted(rooms, key=lambda p: p.name)
    return {"val": rooms[:2], "test": rooms[2:4], "train": rooms[4:]}


def find_recordings(room: Path) -> Tuple[List[Path], List[Path]]:
    """Return (foa recordings, tetrahedral recordings) in a room directory."""
    wavs = sorted(room.rglob("*.wav"))
    foa = [w for w in wavs if "ambience_foa" in w.name.lower()]
    tetra = [w for w in wavs if "ambience_tetra" in w.name.lower()]
    return foa, tetra


def segment(path: Path, sr: int, n_samples: int) -> List[np.ndarray]:
    """Load a recording, resample, and cut into non-overlapping ``[C, n_samples]`` clips."""
    wav, file_sr = au.sf.read(str(path), dtype="float32")   # [T, C]
    if wav.ndim == 1:
        wav = wav[:, None]
    if file_sr != sr:
        wav = np.stack([au.resample(wav[:, c], file_sr, sr) for c in range(wav.shape[1])], axis=1)
    clips = []
    for start in range(0, len(wav) - n_samples + 1, n_samples):
        clips.append(wav[start:start + n_samples].T.copy())   # [C, n_samples]
    return clips


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="TAU-SNoise_DB root.")
    p.add_argument("--out", required=True, type=Path, help="Output data root (gets ambient/).")
    p.add_argument("--hrtf", required=True, type=Path, help="SOFA HRTF for binaural decode.")
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--duration", type=float, default=10.0)
    args = p.parse_args()

    n_samples = int(args.duration * args.sample_rate)
    rooms = [d for d in args.source.iterdir() if d.is_dir()]
    if not rooms:
        raise SystemExit(f"No room directories under {args.source}")
    splits = split_rooms(rooms)

    hrtf = load_hrtf(str(args.hrtf))
    left_filters, right_filters = build_decode_filters(hrtf, args.sample_rate)

    for split, split_rooms_ in splits.items():
        counts = {"foa": 0, "mic": 0, "binaural": 0}
        for room in split_rooms_:
            foa_recs, tetra_recs = find_recordings(room)
            # Segment names are shared across formats ({room}_{rec}_{i}) so the loader
            # can look up the same filename in foa/, mic/, and binaural/.
            # FOA -> foa/ and (decoded) binaural/
            for k, rec in enumerate(foa_recs):
                for i, clip in enumerate(segment(rec, args.sample_rate, n_samples)):
                    stem = f"{room.name}_{k}_{i:04d}"
                    _write(args.out, "foa", split, stem, clip, args.sample_rate)
                    binaural = foa_to_binaural(clip[ACN_TO_WXYZ], left_filters, right_filters)
                    _write(args.out, "binaural", split, stem, binaural, args.sample_rate)
                    counts["foa"] += 1
                    counts["binaural"] += 1
            # Tetrahedral -> mic/ (same {room}_{rec}_{i} names as the FOA segments)
            for k, rec in enumerate(tetra_recs):
                for i, clip in enumerate(segment(rec, args.sample_rate, n_samples)):
                    stem = f"{room.name}_{k}_{i:04d}"
                    _write(args.out, "mic", split, stem, clip, args.sample_rate)
                    counts["mic"] += 1
        print(f"[{split}] {counts} from {len(split_rooms_)} rooms")


def _write(out: Path, fmt: str, split: str, stem: str, clip: np.ndarray, sr: int) -> None:
    d = out / "ambient" / split / fmt   # loader reads ambient/<split>/<format>/
    d.mkdir(parents=True, exist_ok=True)
    au.write_wav(str(d / f"{stem}.wav"), clip.T, sr)   # soundfile wants [T, C]


if __name__ == "__main__":
    main()
