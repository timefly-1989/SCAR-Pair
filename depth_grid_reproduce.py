#!/usr/bin/env python3
"""Reconstruct the complete 4x4 SCAR-Pair candidate-depth grid."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import scar_pair_reproduce as base
import stage2_evaluate as stage2


LIGHT_DEPTHS = (50, 100, 200, 400)
CE_DEPTHS = (25, 50, 100, 200)


def prepare() -> None:
    """Compute depth-400 first-stage features; no new cross-encoder calls are needed."""
    corpus = base.load_corpus()
    old = joblib.load(base.CACHE / "stage2_records.joblib")
    texts = [row["text"] for row in corpus.documents]
    bm25 = BM25Okapi([base.tokenize(text) for text in tqdm(texts, desc="Tokenize memory")], k1=1.5, b=0.75)
    model = base.load_encoder()
    documents = np.load(base.CACHE / "bge_small_documents.npy")
    queries = [base.query_text(task) for task in corpus.tasks]
    tools = [base.tool_view(task) for task in corpus.tasks]
    fields: list[str] = []
    field_slices = []
    for task in corpus.tasks:
        start = len(fields); fields.extend(base.field_views(task)); field_slices.append((start, len(fields)))
    q_emb = base.encode(model, queries, "Depth-grid request encoding")
    t_emb = base.encode(model, tools, "Depth-grid tool encoding")
    f_emb = base.encode(model, fields, "Depth-grid field encoding")
    records = []
    for i, (task, original) in enumerate(tqdm(zip(corpus.tasks, old), total=len(old), desc="Depth-400 features")):
        bq = np.asarray(bm25.get_scores(base.tokenize(queries[i])), dtype=np.float32)
        bt = np.asarray(bm25.get_scores(base.tokenize(tools[i])), dtype=np.float32)
        cq = np.asarray(documents @ q_emb[i], dtype=np.float32)
        ct = np.asarray(documents @ t_emb[i], dtype=np.float32)
        obq, ocq, obt, oct = map(base.stable_order, (bq, cq, bt, ct))
        rbq, rcq, rbt, rct = map(base.ranks_from_order, (obq, ocq, obt, oct))
        rrfq = base.top_from_rrf(rbq, rcq, 400)
        rrfs = base.top_from_rrf(rbt, rct, 400)
        broad = base.union_in_rank_order(obq[:400], ocq[:400], rrfq, rrfs)
        train_ids = base.union_in_rank_order(broad, original["gold"])
        f_bm25 = []
        f_dense = []
        start, end = field_slices[i]
        for local in range(start, end):
            f_bm25.append(base.minmax(np.asarray(bm25.get_scores(base.tokenize(fields[local])), dtype=np.float32))[train_ids])
            f_dense.append(((np.asarray(documents @ f_emb[local]) + 1.0) / 2.0)[train_ids])
        fb = np.stack(f_bm25); fd = np.stack(f_dense)
        light = np.column_stack([
            base.minmax(bq)[train_ids], (cq[train_ids] + 1.0) / 2.0,
            base.minmax(bt)[train_ids], (ct[train_ids] + 1.0) / 2.0,
            fb.max(axis=0), fb.mean(axis=0), fd.max(axis=0), fd.mean(axis=0),
            1.0 / rbq[train_ids], 1.0 / rcq[train_ids],
            1.0 / rbt[train_ids], 1.0 / rct[train_ids],
        ]).astype(np.float32)
        records.append({
            "qa_id": task["qa_id"], "task": task, "gold": original["gold"],
            "orders": {"bm25": obq, "dense": ocq, "rrf_q": rrfq, "rrf_s": rrfs},
            "broad_400": broad, "train_ids_400": train_ids, "light_400": light,
        })
    joblib.dump(records, base.CACHE / "depth_grid_base.joblib", compress=3)
    print(json.dumps({"records": len(records), "mean_depth_400_pool": float(np.mean([len(r['broad_400']) for r in records]))}, indent=2))


def select_rows(ids: np.ndarray, source_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    pos = {int(doc): idx for idx, doc in enumerate(source_ids)}
    return values[np.asarray([pos[int(doc)] for doc in ids], dtype=np.int32)]


def cell_records(
    prepared: list[dict[str, Any]], originals: list[dict[str, Any]], d_light: int, d_ce: int,
) -> list[dict[str, Any]]:
    result = []
    for prep, original in zip(prepared, originals):
        order = prep["orders"]
        broad = base.union_in_rank_order(
            order["bm25"][:d_light], order["dense"][:d_light],
            order["rrf_q"][:d_light], order["rrf_s"][:d_light],
        )
        train_ids = base.union_in_rank_order(broad, prep["gold"])
        light = select_rows(train_ids, prep["train_ids_400"], prep["light_400"])
        ce_pool = base.union_in_rank_order(
            order["bm25"][:d_ce], order["dense"][:d_ce],
            order["rrf_q"][:d_ce], order["rrf_s"][:d_ce],
        )
        old_pos = {int(doc): idx for idx, doc in enumerate(original["train_ids"])}
        if any(int(doc) not in old_pos for doc in ce_pool):
            raise ValueError("CE pool escaped the cached depth-200 broad union")
        take = np.asarray([old_pos[int(doc)] for doc in ce_pool], dtype=np.int32)
        q_raw = original["ce_broad"][take, 0]
        s_raw = original["ce_broad"][take, 3]
        ce = np.column_stack((
            stage2.extend_view(train_ids, ce_pool, q_raw),
            stage2.extend_view(train_ids, ce_pool, s_raw),
        )).astype(np.float32)
        result.append({
            "qa_id": prep["qa_id"], "task": prep["task"], "gold": prep["gold"],
            "broad": broad, "train_ids": train_ids, "light": light, "ce_narrow": ce,
        })
    return result


def evaluate() -> None:
    prepared = joblib.load(base.CACHE / "depth_grid_base.joblib")
    originals = joblib.load(base.CACHE / "stage2_records.joblib")
    folds = stage2.protocols(originals)["stratified"]
    results = []
    predictions = []
    diagnostics = []
    for d_light in LIGHT_DEPTHS:
        for d_ce in CE_DEPTHS:
            print(f"depth cell dL={d_light}, dC={d_ce}")
            records = cell_records(prepared, originals, d_light, d_ce)
            cell_metrics = []
            for fold, (train, test) in enumerate(folds):
                scale, model, diag = stage2.fit_pairwise(records, train)
                diagnostics.append({"d_light": d_light, "d_ce": d_ce, "fold": fold, **diag})
                for idx in test:
                    record = records[int(idx)]
                    ranking = stage2.rank_pairwise(record, scale, model)
                    metrics = stage2.task_metrics(ranking, record["gold"])
                    cell_metrics.append(metrics)
                    predictions.append({
                        "d_light": d_light, "d_ce": d_ce, "fold": fold,
                        "qa_id": record["qa_id"],
                        "top10_document_indices": [int(x) for x in ranking[:10]], **metrics,
                    })
            results.append({
                "d_light": d_light, "d_ce": d_ce,
                "mean_pool": float(np.mean([len(r["broad"]) for r in records])),
                **{metric: float(np.mean([row[metric] for row in cell_metrics]))
                   for metric in ("hit", "recall", "complete", "mrr", "ndcg")},
            })
    with (base.OUTPUT / "depth_grid_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "protocol": "fixed stratified five-fold out-of-fold",
        "light_depths": LIGHT_DEPTHS, "cross_encoder_depths": CE_DEPTHS,
        "cross_encoder_reuse": "raw scores from the cached depth-200 broad union; no new CE inference",
        "results": results, "fit_diagnostics": diagnostics,
    }
    base.write_json(base.OUTPUT / "depth_grid_report.json", report)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate", "all"))
    args = parser.parse_args()
    if args.command in {"prepare", "all"}:
        prepare()
    if args.command in {"evaluate", "all"}:
        evaluate()


if __name__ == "__main__":
    main()
