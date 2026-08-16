"""Build the benchmark's source-clip pool from public datasets.

Turns ESC-50, MUSAN, and UrbanSound8K into the ``audio/<split>/<class>/*.wav``
layout the loader expects, mapping each dataset's categories onto the seven event
classes and producing balanced, RMS-normalized 10-second clips.

    python -m preprocessing.build esc50        --source /path/to/ESC-50        --out /path/to/data
    python -m preprocessing.build musan        --source /path/to/musan         --out /path/to/data
    python -m preprocessing.build urbansound8k --source /path/to/UrbanSound8K  --out /path/to/data

Run all three into the same ``--out`` to assemble the full 7-class pool:

    animals, human_nonspeech, domestic   (ESC-50)
    music, speech                        (MUSAN)
    construction, motor                  (UrbanSound8K)

Two strategies balance the classes to ``--per_class`` clips per class (split across
train/val/test by the split ratios):
short clips (ESC-50, UrbanSound8K) are repeated to 10s and circularly shifted;
long recordings (MUSAN) are cut into non-overlapping 10s windows. Given a seed,
the split and sampling are deterministic.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from preprocessing import audio_utils as au

# A pool entry is a source file plus a variant: ("shift", n) or ("window", start).
Variant = Tuple[str, int]
PoolEntry = Tuple[Path, Variant]
Item = Tuple[Path, str]  # (source path, class name)


# =============================================================================
# Dataset category -> event class mappings and discovery
# =============================================================================

ESC50_TO_CLASS = {
    **{c: "animals" for c in
       ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"]},
    **{c: "human_nonspeech" for c in
       ["crying_baby", "sneezing", "clapping", "breathing", "coughing",
        "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping"]},
    **{c: "domestic" for c in
       ["door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
        "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
        "clock_tick", "glass_breaking"]},
}
URBANSOUND8K_TO_CLASS = {
    "drilling": "construction", "jackhammer": "construction",
    "engine_idling": "motor", "car_horn": "motor", "siren": "motor",
}


def discover_esc50(source: Path) -> List[Item]:
    rows = csv.DictReader((source / "meta" / "esc50.csv").open())
    file_to_class = {r["filename"]: ESC50_TO_CLASS[r["category"]]
                     for r in rows if r["category"] in ESC50_TO_CLASS}
    return [(p, file_to_class[p.name])
            for p in sorted((source / "audio").glob("*.wav")) if p.name in file_to_class]


def discover_musan(source: Path) -> List[Item]:
    items: List[Item] = []
    for top in ("music", "speech"):
        items += [(p, top) for p in sorted((source / top).rglob("*.wav"))]
    return items


def discover_urbansound8k(source: Path) -> List[Item]:
    rows = csv.DictReader((source / "metadata" / "UrbanSound8K.csv").open())
    file_to_class = {r["slice_file_name"]: URBANSOUND8K_TO_CLASS[r["class"]]
                     for r in rows if r["class"] in URBANSOUND8K_TO_CLASS}
    items: List[Item] = []
    for p in sorted((source / "audio").rglob("*.wav")):
        cls = file_to_class.get(p.name)
        if cls:
            items.append((p, cls))
    return items


# name -> (classes, discovery fn, strategy, shifts-per-file for the shift strategy)
DATASETS: Dict[str, Tuple[List[str], Callable[[Path], List[Item]], str, int]] = {
    "esc50": (["animals", "human_nonspeech", "domestic"], discover_esc50, "shift", 10),
    "musan": (["music", "speech"], discover_musan, "window", 0),
    "urbansound8k": (["construction", "motor"], discover_urbansound8k, "shift", 2),
}


# =============================================================================
# Split, pool, render
# =============================================================================

def split_by_ratio(items: List[Item], ratios: Tuple[float, float, float],
                   seed: int) -> Dict[str, Dict[str, List[Path]]]:
    """Shuffle each class's files and split into disjoint train/val/test lists."""
    by_class: Dict[str, List[Path]] = {}
    for path, cls in items:
        by_class.setdefault(cls, []).append(path)
    train_r, val_r, _ = ratios
    out = {s: {} for s in ("train", "val", "test")}
    rng = random.Random(seed)
    for cls, paths in by_class.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        t, v = int(n * train_r), int(n * val_r)
        out["train"][cls] = shuffled[:t]
        out["val"][cls] = shuffled[t:t + v]
        out["test"][cls] = shuffled[t + v:]
    return out


def build_pool(files_by_class: Dict[str, List[Path]], strategy: str, num_shifts: int,
               sr: int, n_samples: int) -> Dict[str, List[PoolEntry]]:
    """Enumerate (path, variant) candidates per class for the chosen strategy."""
    pool: Dict[str, List[PoolEntry]] = {c: [] for c in files_by_class}
    for cls, paths in files_by_class.items():
        for path in paths:
            try:
                wav = au.resample(*au.load_mono(str(path)), sr)
            except Exception:
                continue
            if strategy == "shift":
                filled = au.repeat_to_fill(wav, n_samples)
                if au.is_silent(filled):
                    continue
                step = max(1, n_samples // num_shifts)
                for i in range(num_shifts):
                    pool[cls].append((path, ("shift", (i * step) % n_samples)))
            else:  # window: non-overlapping 10s crops of a long recording
                start = 0
                while start + n_samples <= len(wav):
                    if not au.is_silent(wav[start:start + n_samples]):
                        pool[cls].append((path, ("window", start)))
                    start += n_samples
    return pool


def render(path: Path, variant: Variant, sr: int, n_samples: int):
    kind, value = variant
    wav = au.resample(*au.load_mono(str(path)), sr)
    if kind == "shift":
        wav = au.circular_shift(au.repeat_to_fill(wav, n_samples), value)
    else:  # window
        start = min(value, max(0, len(wav) - n_samples))
        wav = au.pad_or_trim(wav[start:start + n_samples], n_samples)
    return au.normalize_dbfs(wav)


def sample_and_write(pool: Dict[str, List[PoolEntry]], per_class: int, out_audio: Path,
                     split: str, sr: int, n_samples: int, seed: int) -> int:
    """Sample up to ``per_class`` variants per class, render, and write."""
    rng = random.Random(seed)
    written = 0
    for cls, entries in pool.items():
        if not entries:
            continue
        out_dir = out_audio / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for path, variant in rng.sample(entries, min(per_class, len(entries))):
            tag = f"s{variant[1]}" if variant[0] == "shift" else f"st{variant[1]}"
            au.write_wav(str(out_dir / f"{path.stem}_{tag}.wav"),
                         render(path, variant, sr, n_samples), sr)
            written += 1
    return written


def run(dataset: str, source: Path, out: Path, per_class: int,
        ratios: Tuple[float, float, float], sr: int, duration: float, seed: int) -> None:
    classes, discover, strategy, num_shifts = DATASETS[dataset]
    n_samples = int(duration * sr)
    out_audio = out / "audio"

    items = discover(source)
    if not items:
        raise SystemExit(f"No source clips found for {dataset} under {source}.")
    print(f"[{dataset}] {len(items)} source files -> classes {classes}")

    splits = split_by_ratio(items, ratios, seed)
    per_split = {"train": int(per_class * ratios[0]),
                 "val": int(per_class * ratios[1])}
    per_split["test"] = per_class - per_split["train"] - per_split["val"]

    for split in ("train", "val", "test"):
        pool = build_pool(splits[split], strategy, num_shifts, sr, n_samples)
        n = sample_and_write(pool, per_split[split], out_audio, split, sr, n_samples, seed)
        sizes = ", ".join(f"{c}={len(pool.get(c, []))}" for c in classes)
        print(f"  [{split}] wrote {n} clips (pool: {sizes})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(DATASETS))
    p.add_argument("--source", required=True, type=Path, help="Dataset root directory.")
    p.add_argument("--out", required=True, type=Path, help="Output data root (gets audio/).")
    p.add_argument("--per_class", type=int, default=4000, help="Clips per class (train+val+test).")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    run(args.dataset, args.source, args.out, args.per_class, ratios,
        args.sample_rate, args.duration, args.seed)


if __name__ == "__main__":
    main()
