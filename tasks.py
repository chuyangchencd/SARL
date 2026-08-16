"""Task registry for the SARL benchmark.

SARL probes seven spatial attributes, grouped into two families:

    source : azimuth, elevation, distance, event
    room   : rt60, volume, shape

Because source and room scenes are rendered with different room-impulse-response
pipelines and carry different metadata, the two families cannot be mixed within a
single evaluation run.

Each task is described by one :class:`TaskSpec`, which defines how the attribute's
label is derived from scene metadata, how many logits the probe head produces, how
a prediction is decoded, and how the task is scored. Data loading, probe heads, and
metrics all read their task-specific settings from :data:`TASKS` here, so a task is
defined in exactly one place.

Scoring
-------
Continuous attributes (azimuth, elevation, distance, rt60) are discretized into
bins, decoded to the predicted bin center, and scored by normalized mean absolute
error::

    score = 1 - MAE / score_range

Categorical attributes (event, volume, shape) are scored by macro-averaged F1.
``volume`` is a continuous attribute discretized into log-spaced bins but treated
as categorical for scoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class Family(str, Enum):
    """Task family. Source and room tasks cannot be evaluated in the same run."""

    SOURCE = "source"
    ROOM = "room"


class Metric(str, Enum):
    """How a task is scored."""

    NORM_MAE = "norm_mae"   # score = 1 - MAE / score_range
    MACRO_F1 = "macro_f1"   # macro-averaged F1 over classes


class Binning(str, Enum):
    """How a target class index is derived from a scene's metadata value."""

    LINEAR = "linear"      # nearest of ``n`` linearly spaced bin centers
    CIRCULAR = "circular"  # angle binned over [0, 360) with wrap-around (azimuth)
    LOG = "log"            # uniform-width buckets in log space (volume)
    NATIVE = "native"      # metadata already provides the class index (event, shape)


@dataclass(frozen=True)
class TaskSpec:
    """Complete description of one probing task.

    Attributes:
        name: Task identifier used on the command line and in result keys.
        family: :class:`Family` the task belongs to.
        metric: How the task is scored (see :class:`Metric`).
        binning: How a metadata value becomes a target class index.
        n_classes: Number of probe-head output logits (bins or classes).
        metadata_key: Key holding the raw value (or class index) in scene metadata.
        value_min, value_max: Bin range in physical units (binned tasks only).
        score_range: ``R`` in ``score = 1 - MAE / R`` (regression tasks only).
        tolerance: Error band, in physical units, for reporting the fraction of
            predictions within tolerance of the true value (regression tasks).
        soft_labels: Whether to train with Gaussian soft targets around the true bin.
        unit: Physical unit of the attribute, for display.
        class_names: Class labels, in index order (categorical tasks only).
    """

    name: str
    family: Family
    metric: Metric
    binning: Binning
    n_classes: int
    metadata_key: str

    # Binned / regression tasks:
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    score_range: Optional[float] = None
    tolerance: Optional[float] = None
    soft_labels: bool = False
    unit: str = ""

    # Categorical tasks:
    class_names: Optional[Tuple[str, ...]] = None

    @property
    def is_regression(self) -> bool:
        return self.metric is Metric.NORM_MAE

    @property
    def is_classification(self) -> bool:
        return self.metric is Metric.MACRO_F1

    def bin_centers(self) -> torch.Tensor:
        """Decoded value for each output class, shape ``[n_classes]``.

        For ``NATIVE`` tasks the class index is its own decoded value, so this
        returns ``arange(n_classes)``.
        """
        if self.binning is Binning.CIRCULAR:
            # Drop the 360-degree endpoint so it does not duplicate 0 degrees.
            return torch.linspace(0.0, 360.0, self.n_classes + 1)[:-1]
        if self.binning is Binning.LINEAR:
            return torch.linspace(self.value_min, self.value_max, self.n_classes)
        if self.binning is Binning.LOG:
            lo, hi = math.log(self.value_min), math.log(self.value_max)
            return torch.exp(torch.linspace(lo, hi, self.n_classes))
        return torch.arange(self.n_classes, dtype=torch.float32)

    def align(self, value: float) -> float:
        """Map a raw metadata value into the space the bin centers live in.

        Only azimuth is remapped, from the ``[-180, 180)`` degrees stored in
        metadata to the ``[0, 360)`` degrees used by its circular bins; every
        other task returns the value unchanged. Callers store this aligned value
        as the ground truth so predictions and targets share one coordinate space.
        """
        if self.binning is Binning.CIRCULAR:
            return (float(value) + 360.0) % 360.0
        return float(value)

    def value_to_index(self, value: float) -> int:
        """Map a raw metadata value to its target class index."""
        if self.binning is Binning.NATIVE:
            return int(value)
        if self.binning is Binning.CIRCULAR:
            v = (float(value) + 360.0) % 360.0
            centers = self.bin_centers()
            d = (centers - v).abs()
            d = torch.minimum(d, 360.0 - d)  # shortest angular distance
            return int(d.argmin().item())
        if self.binning is Binning.LOG:
            v = max(float(value), 1e-12)
            lo, hi = math.log(self.value_min), math.log(self.value_max)
            step = (hi - lo) / self.n_classes
            idx = int((math.log(v) - lo) / step)
            return min(max(idx, 0), self.n_classes - 1)
        centers = self.bin_centers()  # LINEAR
        return int((centers - float(value)).abs().argmin().item())


# =============================================================================
# Task definitions
# =============================================================================
# Azimuth is scored against a range of 180 degrees (the largest possible circular
# error), not the full 360-degree sampling span.

EVENT_CLASSES: Tuple[str, ...] = (
    "animals", "domestic", "human_nonspeech", "music", "speech",
    "construction", "motor",
)
SHAPE_CLASSES: Tuple[str, ...] = ("cube", "flat", "corridor", "rectangular")

_SPECS: List[TaskSpec] = [
    # --- source family ----------------------------------------------------
    TaskSpec(
        name="azimuth", family=Family.SOURCE, metric=Metric.NORM_MAE,
        binning=Binning.CIRCULAR, n_classes=36, metadata_key="azimuth",
        value_min=0.0, value_max=360.0, score_range=180.0,
        tolerance=20.0, soft_labels=True, unit="deg",
    ),
    TaskSpec(
        name="elevation", family=Family.SOURCE, metric=Metric.NORM_MAE,
        binning=Binning.LINEAR, n_classes=12, metadata_key="elevation",
        value_min=-60.0, value_max=60.0, score_range=120.0,
        tolerance=20.0, soft_labels=True, unit="deg",
    ),
    TaskSpec(
        name="distance", family=Family.SOURCE, metric=Metric.NORM_MAE,
        binning=Binning.LINEAR, n_classes=20, metadata_key="distance",
        value_min=0.5, value_max=2.5, score_range=2.0,
        tolerance=0.25, soft_labels=True, unit="m",
    ),
    TaskSpec(
        name="event", family=Family.SOURCE, metric=Metric.MACRO_F1,
        binning=Binning.NATIVE, n_classes=7, metadata_key="event",
        class_names=EVENT_CLASSES,
    ),
    # --- room family ------------------------------------------------------
    TaskSpec(
        name="rt60", family=Family.ROOM, metric=Metric.NORM_MAE,
        binning=Binning.LINEAR, n_classes=29, metadata_key="rt60",
        value_min=0.1, value_max=3.0, score_range=2.9,
        tolerance=0.5, soft_labels=True, unit="s",
    ),
    TaskSpec(
        name="volume", family=Family.ROOM, metric=Metric.MACRO_F1,
        binning=Binning.LOG, n_classes=5, metadata_key="volume",
        value_min=60.0, value_max=2500.0, unit="m^3",
    ),
    TaskSpec(
        name="shape", family=Family.ROOM, metric=Metric.MACRO_F1,
        binning=Binning.NATIVE, n_classes=4, metadata_key="shape",
        class_names=SHAPE_CLASSES,
    ),
]

TASKS: Dict[str, TaskSpec] = {spec.name: spec for spec in _SPECS}

SOURCE_TASKS: Tuple[str, ...] = tuple(s.name for s in _SPECS if s.family is Family.SOURCE)
ROOM_TASKS: Tuple[str, ...] = tuple(s.name for s in _SPECS if s.family is Family.ROOM)


def get_task(name: str) -> TaskSpec:
    """Look up a task by name."""
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(
            f"Unknown task {name!r}. Available tasks: {', '.join(TASKS)}."
        ) from None


def resolve_tasks(names: Sequence[str]) -> List[TaskSpec]:
    """Resolve task names for a run, rejecting sets that span both families."""
    specs = [get_task(n) for n in names]
    families = {s.family for s in specs}
    if len(families) > 1:
        raise ValueError(
            "Source and room tasks cannot be evaluated in the same run "
            f"(requested: {', '.join(s.name for s in specs)}).\n"
            f"  source tasks: {', '.join(SOURCE_TASKS)}\n"
            f"  room tasks:   {', '.join(ROOM_TASKS)}"
        )
    return specs


def family_of(names: Sequence[str]) -> Family:
    """Return the family shared by a set of tasks (validates they don't mix)."""
    return resolve_tasks(names)[0].family


if __name__ == "__main__":
    # Print the registry as a table and run a few sanity checks.
    hdr = f"{'task':<10} {'family':<7} {'metric':<9} {'binning':<9} {'n':>3}  decoding"
    print(hdr)
    print("-" * len(hdr))
    for spec in _SPECS:
        centers = spec.bin_centers()
        if spec.binning is Binning.NATIVE:
            decode = ", ".join(spec.class_names)
        else:
            decode = f"[{centers[0]:.3g} .. {centers[-1]:.3g}] {spec.unit}"
        print(
            f"{spec.name:<10} {spec.family.value:<7} {spec.metric.value:<9} "
            f"{spec.binning.value:<9} {spec.n_classes:>3}  {decode}"
        )

    assert TASKS["azimuth"].value_to_index(-179.0) == TASKS["azimuth"].value_to_index(181.0)
    assert TASKS["elevation"].value_to_index(-60.0) == 0
    assert TASKS["elevation"].value_to_index(60.0) == 11
    assert TASKS["distance"].value_to_index(0.5) == 0
    assert TASKS["volume"].value_to_index(60.0) == 0
    assert TASKS["volume"].value_to_index(2500.0) == 4
    assert TASKS["event"].value_to_index(3) == 3
    assert family_of(["azimuth", "distance"]) is Family.SOURCE
    try:
        resolve_tasks(["azimuth", "rt60"])
    except ValueError:
        pass
    else:
        raise AssertionError("mixing families should raise")
    print("\nAll registry checks passed.")
