#!/usr/bin/env python3
"""Standalone mGTE and BGE-M3 baselines on reconstructed Mem2ActBench."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import scar_pair_reproduce as base
from locomo_reproduce import MODEL_SPECS
from stage2_evaluate import task_metrics


def load_model(name: str) -> SentenceTransformer:
    local, remote, revision = MODEL_SPECS[name]
    source = str(local) if local.exists() else remote
    model = SentenceTransformer(source, device="mps", trust_remote_code=(name == "mgte"),
                                revision=None if local.exists() else revision)
    model.max_seq_length = 512
    return model


def cached_encode(
    model: SentenceTransformer, path: Path, texts: list[str], description: str
) -> tuple[np.ndarray, float]:
    if path.exists():
        values = np.load(path)
        if len(values) != len(texts) or not np.isfinite(values).all():
            raise ValueError(f"Invalid embedding cache {path}")
        return values, 0.0
    start = time.perf_counter()
    values = base.encode(model, texts, description, batch_size=64)
    elapsed = time.perf_counter() - start
    np.save(path, values)
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("mgte", "bge-m3"))
    parser.add_argument("--schema-style", choices=("full", "tool"), default="full")
    args = parser.parse_args()
    corpus = base.load_corpus()
    model = load_model(args.model)
    documents = [row["text"] for row in corpus.documents]
    requests = [base.query_text(task) for task in corpus.tasks]
    schemas = [
        (base.schema_view(task) if args.schema_style == "full" else base.tool_view(task))
        for task in corpus.tasks
    ]
    doc, doc_time = cached_encode(
        model, base.CACHE / f"{args.model}_mem2act_documents.npy", documents,
        f"{args.model} Mem2Act documents"
    )
    q, q_time = cached_encode(
        model, base.CACHE / f"{args.model}_mem2act_requests.npy", requests,
        f"{args.model} Mem2Act requests"
    )
    schema, schema_time = cached_encode(
        model, base.CACHE / f"{args.model}_mem2act_schemas_{args.schema_style}.npy", schemas,
        f"{args.model} Mem2Act schema views"
    )
    predictions = []
    aggregate_rows = {"request": [], "schema": []}
    for idx, task in enumerate(tqdm(corpus.tasks, desc=f"{args.model} ranking")):
        gold = np.asarray(
            [corpus.id_to_index[item] for item in task["source_conversation_ids"]], dtype=np.int32
        )
        for view, embedding in (("request", q[idx]), ("schema", schema[idx])):
            scores = np.asarray(doc @ embedding, dtype=np.float32)
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite score for {task['qa_id']}/{view}")
            order = base.stable_order(scores)
            metrics = {str(k): task_metrics(order, gold, k) for k in (1, 3, 5, 10)}
            aggregate_rows[view].append(metrics["5"])
            predictions.append(
                {"qa_id": task["qa_id"], "view": view,
                 "top10_document_ids": [corpus.documents[int(x)]["id"] for x in order[:10]],
                 "metrics_by_k": metrics}
            )
    aggregates = []
    for view, rows in aggregate_rows.items():
        aggregates.append(
            {"view": view, "n": len(rows),
             **{metric: float(np.mean([row[metric] for row in rows]))
                for metric in ("hit", "recall", "complete", "mrr", "ndcg")}}
        )
    label = f"{args.model}_mem2act_{args.schema_style}"
    with (base.OUTPUT / f"{label}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "model": MODEL_SPECS[args.model][1], "revision": MODEL_SPECS[args.model][2],
        "max_length": 512, "document_serialization": "content-only SCAR reconstruction",
        "schema_serialization": (
            "labelled request, tool, description, and parameter pairs"
            if args.schema_style == "full" else "labelled request, tool, and description only"
        ),
        "timing_seconds": {"documents": doc_time, "requests": q_time, "schemas": schema_time},
        "aggregates_at_5": aggregates,
    }
    base.write_json(base.OUTPUT / f"{label}_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
