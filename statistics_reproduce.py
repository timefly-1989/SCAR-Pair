#!/usr/bin/env python3
"""Paired uncertainty analysis over independently reproduced predictions."""

from __future__ import annotations

import json
from collections import defaultdict

import joblib
import numpy as np

import scar_pair_reproduce as base
from stage2_evaluate import shared_source_groups


N_BOOT = 10_000
N_RANDOM = 50_000
COMPARATORS = ("Query-Fuse-strict", "CE-Calibrated-Broad", "SCAR-Fuse")
METRICS = ("recall", "complete", "ndcg")


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for position, key in enumerate(ordered):
        running = max(running, (total - position) * values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def main() -> None:
    prediction_path = base.OUTPUT / "retrieval_predictions.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing {prediction_path}; run stage2_evaluate.py evaluate")
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    records = joblib.load(base.CACHE / "stage2_records.joblib")
    qa_order = [record["qa_id"] for record in records]
    qa_pos = {qa_id: i for i, qa_id in enumerate(qa_order)}
    component = shared_source_groups(records)
    components: dict[int, list[int]] = defaultdict(list)
    for idx, group in enumerate(component):
        components[int(group)].append(idx)
    component_members = list(components.values())
    lookup = {
        (row["protocol"], row["method"], row["qa_id"]): row
        for row in rows
    }
    rng = np.random.default_rng(base.SEED)
    output = []
    for baseline in COMPARATORS:
        raw_p: dict[str, float] = {}
        pending = []
        for metric in METRICS:
            differences = np.asarray(
                [
                    lookup[("stratified", "SCAR-Pair", qa_id)][metric]
                    - lookup[("stratified", baseline, qa_id)][metric]
                    for qa_id in qa_order
                ],
                dtype=np.float64,
            )
            boot = np.empty(N_BOOT, dtype=np.float64)
            for draw in range(N_BOOT):
                sampled = rng.integers(0, len(component_members), len(component_members))
                indices = np.concatenate([component_members[x] for x in sampled])
                boot[draw] = differences[indices].mean()
            signs = rng.choice((-1.0, 1.0), size=(N_RANDOM, len(differences)))
            randomized = (signs * differences).mean(axis=1)
            observed = float(differences.mean())
            p_value = float((np.count_nonzero(np.abs(randomized) >= abs(observed)) + 1) / (N_RANDOM + 1))
            raw_p[metric] = p_value
            pending.append(
                {"baseline": baseline, "metric": metric, "difference": observed,
                 "cluster_bootstrap_percentile_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                 "task_sign_randomization_p_raw": p_value}
            )
        adjusted = holm(raw_p)
        for row in pending:
            row["task_sign_randomization_p_holm"] = adjusted[row["metric"]]
            output.append(row)
    report = {
        "seed": base.SEED,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_unit": "shared-gold-source connected component",
        "bootstrap_statistic": "task-weighted mean after resampled-component concatenation",
        "bootstrap_interval": "percentile",
        "randomization_resamples": N_RANDOM,
        "randomization_unit": "task-level independent sign flip",
        "randomization_plus_one_correction": True,
        "holm_family": "three metrics separately for each comparator",
        "results": output,
    }
    base.write_json(base.OUTPUT / "statistics_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
