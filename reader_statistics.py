#!/usr/bin/env python3
"""Legacy task-level reader analysis retained for historical comparison.

The submission tables do not use this result. Run ``publication_analysis.py``
for the shared-source-component bootstrap and joint component sign-randomization
analysis used by the manuscript.
"""

from __future__ import annotations

import json
from collections import defaultdict

import joblib
import numpy as np

import scar_pair_reproduce as base
from reader_reproduce import gold_consistency
from stage2_evaluate import shared_source_groups
from statistics_reproduce import N_BOOT, N_RANDOM, holm


COMPARATORS = (
    "No-memory", "Query-Fuse-strict", "CE-Calibrated-Broad",
    "SCAR-Fuse", "Oracle-Gold", "Hybrid-Q", "CE-Calibrated",
)
METRICS = ("memory_argument_em", "memory_call_em", "full_call_em")


def main() -> None:
    rows = [json.loads(line) for line in (base.OUTPUT / "reader_predictions.jsonl").read_text().splitlines() if line.strip()]
    lookup = {(row["condition"], row["qa_id"]): row for row in rows}
    corpus = base.load_corpus()
    valid_ids, _ = gold_consistency(corpus)
    records = joblib.load(base.CACHE / "stage2_records.joblib")
    qa_order = [record["qa_id"] for record in records if record["qa_id"] in valid_ids]
    missing = [(condition, qa_id) for condition in ("SCAR-Pair", *COMPARATORS)
               for qa_id in qa_order if (condition, qa_id) not in lookup]
    if missing:
        raise ValueError(f"Reader predictions incomplete, first missing keys: {missing[:5]}")
    all_groups = shared_source_groups(records)
    members: dict[int, list[int]] = defaultdict(list)
    valid_position = {qa_id: pos for pos, qa_id in enumerate(qa_order)}
    for record, group in zip(records, all_groups):
        if record["qa_id"] in valid_position:
            members[int(group)].append(valid_position[record["qa_id"]])
    components = list(members.values())
    rng = np.random.default_rng(base.SEED + 41)
    results = []
    for comparator in COMPARATORS:
        pending = []
        raw_p = {}
        for metric in METRICS:
            differences = np.asarray([
                float(lookup[("SCAR-Pair", qa_id)][metric])
                - float(lookup[(comparator, qa_id)][metric])
                for qa_id in qa_order
            ])
            boot = np.empty(N_BOOT)
            for draw in range(N_BOOT):
                sampled = rng.integers(0, len(components), len(components))
                indices = np.concatenate([components[idx] for idx in sampled])
                boot[draw] = differences[indices].mean()
            observed = float(differences.mean())
            random_values = np.empty(N_RANDOM)
            # Chunk sign matrices so the procedure remains memory-bounded.
            for start in range(0, N_RANDOM, 5000):
                end = min(start + 5000, N_RANDOM)
                signs = rng.choice((-1.0, 1.0), size=(end - start, len(differences)))
                random_values[start:end] = (signs * differences).mean(axis=1)
            p = float((np.count_nonzero(np.abs(random_values) >= abs(observed)) + 1) / (N_RANDOM + 1))
            raw_p[metric] = p
            pending.append({
                "comparator": comparator, "metric": metric, "difference": observed,
                "cluster_bootstrap_percentile_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "task_sign_randomization_p_raw": p,
            })
        adjusted = holm(raw_p)
        for row in pending:
            row["task_sign_randomization_p_holm"] = adjusted[row["metric"]]
            results.append(row)
    report = {
        "status": "legacy sensitivity analysis; not used by the manuscript",
        "tasks": len(qa_order), "seed": base.SEED + 41,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_unit": "shared-gold-source connected component",
        "bootstrap_statistic": "task-weighted mean after resampled-component concatenation",
        "bootstrap_interval": "percentile",
        "randomization_resamples": N_RANDOM,
        "randomization_unit": "task-level independent sign flip",
        "randomization_plus_one_correction": True,
        "holm_family": "three exact-match metrics separately for each comparator",
        "results": results,
    }
    base.write_json(base.OUTPUT / "legacy_reader_statistics_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
