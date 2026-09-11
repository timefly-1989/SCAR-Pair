#!/usr/bin/env python3
"""Fail-fast integrity checks for the complete reproduction artifact set."""

from __future__ import annotations

import json
import math
import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import scar_pair_reproduce as base


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            finite(item, f"{path}[{idx}]")


def unique(rows: list[dict[str, Any]], keys: tuple[str, ...], label: str) -> None:
    values = [tuple(row[key] for key in keys) for row in rows]
    if len(values) != len(set(values)):
        duplicates = [key for key, count in Counter(values).items() if count > 1]
        raise ValueError(f"Duplicate {label} keys: {duplicates[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial-reader", action="store_true")
    args = parser.parse_args()
    checks = []
    retrieval = jsonl(base.OUTPUT / "retrieval_predictions.jsonl")
    if len(retrieval) != 26452:
        raise ValueError(f"retrieval rows: {len(retrieval)}")
    unique(retrieval, ("protocol", "qa_id", "method"), "retrieval")
    finite(retrieval)
    checks.append({"artifact": "retrieval_predictions.jsonl", "rows": len(retrieval)})

    expected = {
        "capacity_predictions.jsonl": 3495,
        "coverage_predictions.jsonl": 2107,
        "depth_grid_predictions.jsonl": 6256,
        "graph_probe_predictions.jsonl": 94916,
        "query_ce_predictions.jsonl": 1556,
    }
    keys = {
        "capacity_predictions.jsonl": ("protocol", "qa_id", "method"),
        "coverage_predictions.jsonl": ("qa_id", "method"),
        "depth_grid_predictions.jsonl": ("d_light", "d_ce", "qa_id"),
        "graph_probe_predictions.jsonl": ("protocol", "qa_id", "graph", "k_graph", "alpha"),
        "query_ce_predictions.jsonl": ("protocol", "qa_id", "method"),
    }
    for name, count in expected.items():
        rows = jsonl(base.OUTPUT / name)
        if len(rows) != count:
            raise ValueError(f"{name} rows: {len(rows)} != {count}")
        unique(rows, keys[name], name)
        finite(rows)
        checks.append({"artifact": name, "rows": len(rows)})

    reader_path = base.OUTPUT / "reader_predictions.jsonl"
    if not reader_path.exists() and not args.allow_partial_reader:
        raise FileNotFoundError(reader_path)
    if reader_path.exists():
        reader = jsonl(reader_path)
        unique(reader, ("condition", "qa_id"), "reader")
        finite(reader)
        conditions = Counter(row["condition"] for row in reader)
        from reader_reproduce import CONDITIONS
        expected_keys = {(c, t['qa_id']) for c in CONDITIONS for t in base.load_corpus().tasks}
        complete = {(r['condition'], r['qa_id']) for r in reader} == expected_keys
        if not complete and not args.allow_partial_reader:
            raise ValueError(f"reader output incomplete: {len(reader)}/3128; {dict(conditions)}")
        checks.append({
            "artifact": reader_path.name, "rows": len(reader),
            "complete": complete, "conditions": dict(conditions),
        })
        if complete:
            report = json.loads((base.OUTPUT / 'reader_report.json').read_text())
            if report.get('prediction_file_sha256') != base.sha256(reader_path):
                raise ValueError('Reader report is stale; run reader_reproduce.py summarize')
            if report.get('verified_prompt_and_metric_rows') != len(reader):
                raise ValueError('Reader full prompt/metric verification missing')
            audit = json.loads((base.OUTPUT / 'reader_parser_audit.json').read_text())
            if (audit.get('prediction_file_sha256') != base.sha256(reader_path)
                    or audit.get('audited_rows') != len(reader)):
                raise ValueError('Reader parser audit is stale or incomplete')
            finite(audit, 'reader_parser_audit.json')
            checks.append({'artifact': 'reader_parser_audit.json',
                           'rows': audit['audited_rows'],
                           'additional_rejections': audit['additional_rejections']})
    for path in sorted(base.OUTPUT.glob("*report*.json")):
        finite(json.loads(path.read_text(encoding="utf-8")), path.name)
    for model in ('bge-small', 'mgte', 'bge-m3'):
        path = base.OUTPUT / f'locomo_predictions_{model}_timestamped_unicode.jsonl'
        rows = jsonl(path)
        if len(rows) != 1982 * 3:
            raise ValueError(f'Incomplete LoCoMo predictions: {model}')
        unique(rows, ('sample_id', 'question_idx', 'method'), model)
        finite(rows)
        checks.append({'artifact': path.name, 'rows': len(rows)})
    report = {"status": "passed", "checks": checks}
    base.write_json(base.OUTPUT / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
