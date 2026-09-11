#!/usr/bin/env python3
"""Recompute the manuscript's descriptive stratified-protocol slices."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

import numpy as np

import scar_pair_reproduce as base


METHODS = ("SCAR-Fuse", "CE-Schema")


def main() -> None:
    corpus = base.load_corpus()
    tasks = {task["qa_id"]: task for task in corpus.tasks}
    raw = [json.loads(line) for line in (base.OUTPUT / "retrieval_predictions.jsonl").read_text().splitlines()]
    rows = {
        (row["qa_id"], row["method"]): row
        for row in raw if row["protocol"] == "stratified" and row["method"] in METHODS
    }

    def level(task: dict[str, Any]) -> str:
        value = task.get("complexity_metadata", {}).get("level", "")
        return value if value in {"L1", "L2"} else "L3-L4"

    def source(task: dict[str, Any]) -> str:
        return base.source_name(task["source_conversation_ids"][0])

    slice_specs: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("L1", lambda t: level(t) == "L1"),
        ("L2", lambda t: level(t) == "L2"),
        ("L3-L4", lambda t: level(t) == "L3-L4"),
        ("One gold source", lambda t: len(t["source_conversation_ids"]) == 1),
        ("Multiple gold sources", lambda t: len(t["source_conversation_ids"]) > 1),
        ("No temporal conflict", lambda t: not t.get("complexity_metadata", {}).get("has_temporal_conflict", False)),
        ("Temporal conflict", lambda t: bool(t.get("complexity_metadata", {}).get("has_temporal_conflict", False))),
        ("ToolACE", lambda t: source(t) == "ToolACE"),
        ("OASST1", lambda t: source(t) == "OASST1"),
        ("BFCL-live", lambda t: source(t) == "BFCL-live"),
    ]
    results = []
    for label, predicate in slice_specs:
        ids = [qa_id for qa_id, task in tasks.items() if predicate(task)]
        method_metrics = {}
        for method in METHODS:
            subset = [rows[(qa_id, method)] for qa_id in ids]
            method_metrics[method] = {
                metric: float(np.mean([row[metric] for row in subset]))
                for metric in ("recall", "complete", "ndcg")
            }
        results.append({
            "slice": label, "tasks": len(ids), "methods": method_metrics,
            "delta_scar_minus_ce": {
                metric: method_metrics["SCAR-Fuse"][metric] - method_metrics["CE-Schema"][metric]
                for metric in ("recall", "complete", "ndcg")
            },
        })
    report = {
        "protocol": "stratified five-fold out-of-fold",
        "comparison": "SCAR-Fuse minus CE-Schema",
        "source_slice_rule": "source corpus of the first annotated source ID",
        "results": results,
    }
    base.write_json(base.OUTPUT / "slice_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
