#!/usr/bin/env python3
"""Strict surface coverage of explicitly grounded argument values."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from typing import Any

import numpy as np

import scar_pair_reproduce as base


METHODS = ("Hybrid-Q", "Calibrated-Q", "SCAR-Light", "CE-Schema", "CE-Calibrated", "SCAR-Fuse")


def scalar_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(scalar_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {scalar_text(item)}" for key, item in value.items())
    return str(value)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", scalar_text(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def contains(haystack: str, needle: str) -> bool:
    return bool(needle and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def main() -> None:
    corpus = base.load_corpus()
    rows = [json.loads(line) for line in (base.OUTPUT / "retrieval_predictions.jsonl").read_text().splitlines()]
    rankings = {
        (row["qa_id"], row["method"]): row["top10_document_indices"][:5]
        for row in rows if row["protocol"] == "stratified" and row["method"] in METHODS
    }
    values_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_by_id = {task["qa_id"]: task for task in corpus.tasks}
    for task in corpus.tasks:
        arguments = task.get("tool_call", {}).get("arguments", {}) or {}
        grounding = task.get("tool_call", {}).get("grounding_info", {}) or {}
        for name, info in grounding.items():
            if info.get("type") != "explicit" or name not in arguments:
                continue
            value = arguments[name]
            normalized = normalize(value)
            if value is None or isinstance(value, bool) or len(normalized) <= 1:
                continue
            values_by_task[task["qa_id"]].append(
                {"argument": name, "value": value, "normalized": normalized}
            )
    if sum(map(len, values_by_task.values())) != 360 or len(values_by_task) != 301:
        raise ValueError("Explicit-value audit does not match 360 values in 301 tasks")
    predictions = []
    for qa_id, values in values_by_task.items():
        task = task_by_id[qa_id]
        gold = [corpus.id_to_index[item] for item in task["source_conversation_ids"]]
        method_rankings = {method: rankings[(qa_id, method)] for method in METHODS}
        method_rankings["Oracle-Gold"] = gold[:5]
        for method, ids in method_rankings.items():
            text = normalize("\n".join(corpus.documents[int(idx)]["text"] for idx in ids))
            found = [contains(text, item["normalized"]) for item in values]
            predictions.append(
                {"qa_id": qa_id, "method": method, "value_count": len(values),
                 "found_count": int(sum(found)), "coverage": float(np.mean(found)),
                 "values": [{**item, "found": hit} for item, hit in zip(values, found)]}
            )
    with (base.OUTPUT / "coverage_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_rows = []
    for method in (*METHODS, "Oracle-Gold"):
        subset = [row for row in predictions if row["method"] == method]
        found = sum(row["found_count"] for row in subset)
        total = sum(row["value_count"] for row in subset)
        report_rows.append(
            {"method": method, "tasks": len(subset), "values": total,
             "values_found": found, "macro": float(np.mean([row["coverage"] for row in subset])),
             "micro": found / total}
        )
    report = {
        "normalization": "NFKC, casefold, non-word to spaces, collapse whitespace, token boundaries",
        "structured_values": "recursive key/value flattening",
        "excluded": "boolean, null, and normalized one-character values",
        "results": report_rows,
    }
    base.write_json(base.OUTPUT / "coverage_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
