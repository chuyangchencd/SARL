"""Scoring for the probing tasks.

Each task accumulates predictions over an evaluation set and reports a task score
in ``[0, 1]`` (higher is better):

* Regression tasks decode the predicted bin center and report normalized MAE,
  ``score = 1 - MAE / score_range`` (azimuth error is circular).
* Classification tasks report macro-averaged F1.

To compare models on the same footing, task scores are mapped through baseline
normalization against a random predictor, ``phi(x; b) = (x - b) / (1 - b)``, which
sends the random baseline to 0 and a perfect score to 1. Source and room family
scores are the mean of their tasks' normalized scores.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch

from tasks import Binning, Family, Metric, TaskSpec, get_task


def _circular_abs_diff(a: torch.Tensor, b: torch.Tensor, wrap: float = 360.0) -> torch.Tensor:
    d = (a - b).abs() % wrap
    return torch.minimum(d, wrap - d)


class TaskMetric:
    """Streaming metric for one task. Feed ``(logits, raw_labels)`` per batch."""

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self._centers = spec.bin_centers()
        self.reset()

    def reset(self) -> None:
        if self.spec.metric is Metric.NORM_MAE:
            self._err_sum = 0.0
            self._within_tol = 0.0
            self._n = 0.0
        else:
            n = self.spec.n_classes
            self._tp = torch.zeros(n)
            self._fp = torch.zeros(n)
            self._fn = torch.zeros(n)
            self._correct = 0.0
            self._n = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, raw: torch.Tensor) -> None:
        """Accumulate a batch.

        Args:
            logits: ``[B, n_classes]`` head outputs.
            raw: ``[B]`` aligned ground-truth values (as produced by the loader).
        """
        logits = logits.detach().cpu()
        raw = raw.detach().cpu().flatten()
        valid = torch.isfinite(raw)
        if valid.sum() == 0:
            return
        logits, raw = logits[valid], raw[valid]
        pred_idx = logits.argmax(dim=-1)
        centers = self._centers.to(logits.dtype)

        if self.spec.metric is Metric.NORM_MAE:
            pred_val = centers[pred_idx]
            if self.spec.binning is Binning.CIRCULAR:
                err = _circular_abs_diff(pred_val, raw)
            else:
                err = (pred_val - raw).abs()
            self._err_sum += float(err.sum())
            self._within_tol += float((err <= self.spec.tolerance).sum())
            self._n += float(err.numel())
        else:
            target = torch.tensor([self.spec.value_to_index(float(v)) for v in raw],
                                  dtype=torch.long)
            self._correct += float((pred_idx == target).sum())
            self._n += float(target.numel())
            for c in range(self.spec.n_classes):
                self._tp[c] += float(((pred_idx == c) & (target == c)).sum())
                self._fp[c] += float(((pred_idx == c) & (target != c)).sum())
                self._fn[c] += float(((pred_idx != c) & (target == c)).sum())

    def compute(self) -> Dict[str, float]:
        """Return metric fields including the unit-interval ``score``."""
        if self._n == 0:
            return {"score": 0.0}
        if self.spec.metric is Metric.NORM_MAE:
            mae = self._err_sum / self._n
            return {
                "mae": mae,
                "within_tol_f1": self._within_tol / self._n,
                "score": 1.0 - mae / self.spec.score_range,
            }
        eps = 1e-8
        prec = self._tp / (self._tp + self._fp + eps)
        rec = self._tp / (self._tp + self._fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        support = (self._tp + self._fn) > 0
        macro_f1 = float(f1[support].mean()) if support.any() else 0.0
        return {
            "macro_f1": macro_f1,
            "acc": self._correct / self._n,
            "score": macro_f1,
        }


def build_metrics(tasks: Iterable[str]) -> Dict[str, TaskMetric]:
    """One :class:`TaskMetric` per task name."""
    return {name: TaskMetric(get_task(name)) for name in tasks}


# =============================================================================
# Aggregation across tasks
# =============================================================================

def normalize_score(score: float, baseline: float) -> float:
    """Baseline normalization ``phi(x; b) = (x - b) / (1 - b)``."""
    denom = 1.0 - baseline
    if abs(denom) < 1e-12:
        return 0.0
    return (score - baseline) / denom


def family_score(task_scores: Dict[str, float], baselines: Dict[str, float],
                 family: Family) -> float:
    """Mean baseline-normalized score over the tasks of one family."""
    vals: List[float] = [
        normalize_score(task_scores[name], baselines.get(name, 0.0))
        for name in task_scores if get_task(name).family is family
    ]
    return sum(vals) / len(vals) if vals else math.nan
