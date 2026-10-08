"""Benchmark Metrics Generator for IRN Pipeline (NFR-REL-02).

Computes:
- De-duplication Precision, Recall, F1-score against ground-truth locations.
- Duplicate detection rate (< 2% target).
- Vectorized execution throughput (target >= 20,000 detections/sec).
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, NamedTuple, Tuple
import numpy as np


class AnomalyMetrics(NamedTuple):
    true_positives: int
    false_positives: int
    false_negatives: int
    duplicate_count: int
    precision: float
    recall: float
    f1_score: float
    duplicate_rate: float


def evaluate_anomaly_benchmarks(
    ground_truth_list: List[Dict[str, Any]],
    recovered_anomalies_list: List[Dict[str, Any]],
    spatial_tolerance_m: float = 3.0
) -> AnomalyMetrics:
    """Evaluates recovered anomalies against ground-truth benchmarks."""
    matched_gt = set()
    matched_rec = set()
    duplicate_count = 0

    for i, rec in enumerate(recovered_anomalies_list):
        rec_s = float(rec.get("s", 0.0))
        rec_t = float(rec.get("t", 0.0))
        rec_cls = str(rec.get("class_name", ""))

        matched_any_gt = False
        for j, gt in enumerate(ground_truth_list):
            gt_s = float(gt.get("s", 0.0))
            gt_t = float(gt.get("t", 0.0))
            gt_cls = str(gt.get("class_name", ""))

            if rec_cls == gt_cls:
                dist = math.sqrt((rec_s - gt_s)**2 + (rec_t - gt_t)**2)
                if dist <= spatial_tolerance_m:
                    if j in matched_gt:
                        duplicate_count += 1
                    else:
                        matched_gt.add(j)
                    matched_rec.add(i)
                    matched_any_gt = True
                    break

    tp = len(matched_gt)
    fp = len(recovered_anomalies_list) - len(matched_rec)
    fn = len(ground_truth_list) - tp

    precision = round(tp / max(1, tp + fp), 4)
    recall = round(tp / max(1, tp + fn), 4)
    f1 = round(2 * precision * recall / max(1e-4, precision + recall), 4)
    dup_rate = round(duplicate_count / max(1, len(recovered_anomalies_list)), 4)

    return AnomalyMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        duplicate_count=duplicate_count,
        precision=precision,
        recall=recall,
        f1_score=f1,
        duplicate_rate=dup_rate
    )


def measure_fusion_throughput(num_detections: int, elapsed_seconds: float) -> float:
    """Returns detections processed per second."""
    if elapsed_seconds <= 0:
        return float(num_detections)
    return round(num_detections / elapsed_seconds, 1)
