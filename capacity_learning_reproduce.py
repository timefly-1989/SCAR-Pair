#!/usr/bin/env python3
"""Independent capacity controls and SCAR-Pair learning curves."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker
from tqdm import tqdm

import scar_pair_reproduce as base
import stage2_evaluate as evaluation


def record_rows(record: dict[str, Any], ids: np.ndarray) -> np.ndarray:
    return evaluation.rows_for(record, ids, "all")


def reliability(record: dict[str, Any]) -> float:
    bm25 = set(int(x) for x in record["request_orders"]["bm25"][:50])
    dense = set(int(x) for x in record["request_orders"]["dense"][:50])
    agreement = len(bm25 & dense) / len(bm25 | dense)
    values = evaluation.rows_for(record, record["broad"], "light")[:, :4]
    separation = float(np.mean(np.maximum(values.max(axis=0) - np.median(values, axis=0), 0.0)))
    return float(np.clip(0.5 * agreement + 0.5 * separation, 0.0, 1.0))


def ensure_graph_features(records: list[dict[str, Any]]) -> None:
    cache = base.CACHE / "capacity_graph_features_candidates_only.joblib"
    if cache.exists():
        saved = joblib.load(cache)
        if len(saved) == len(records):
            for record, values in zip(records, saved):
                record["graph"] = values
            return
    document_embeddings = np.load(base.CACHE / "bge_small_documents.npy")
    saved = []
    for record in tqdm(records, desc="Five-neighbour graph features"):
        ids = record["train_ids"]
        embeddings = document_embeddings[ids]
        neighbour_ids = record["broad"]
        similarity = np.asarray(embeddings @ document_embeddings[neighbour_ids].T, dtype=np.float32)
        similarity[ids[:, None] == neighbour_ids[None, :]] = -np.inf
        k = min(5, len(neighbour_ids) - 1)
        neighbours = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        neighbour_similarity = np.take_along_axis(similarity, neighbours, axis=1)
        ce = evaluation.rows_for(record, neighbour_ids, "ce")
        neighbour_ce = ce[neighbours].mean(axis=1)
        graph = np.column_stack(
            (neighbour_ce, neighbour_similarity.mean(axis=1), neighbour_similarity.max(axis=1))
        ).astype(np.float32)
        if graph.shape != (len(ids), 8) or not np.isfinite(graph).all():
            raise ValueError(f"Invalid graph features for {record['qa_id']}")
        record["graph"] = graph
        saved.append(graph)
    joblib.dump(saved, cache, compress=3)


def rows_kind(record: dict[str, Any], ids: np.ndarray, kind: str) -> np.ndarray:
    x = record_rows(record, ids)
    if kind == "base":
        return x
    if kind == "reliability":
        return np.column_stack((x, x * reliability(record))).astype(np.float32)
    if kind == "graph":
        pos = {int(doc): idx for idx, doc in enumerate(record["train_ids"])}
        take = np.asarray([pos[int(doc)] for doc in ids])
        return np.column_stack((x, record["graph"][take])).astype(np.float32)
    raise KeyError(kind)


def fit_reliability(
    records: list[dict[str, Any]], train: Iterable[int]
) -> tuple[StandardScaler, LogisticRegression]:
    xs, ys = [], []
    for idx in train:
        record = records[int(idx)]
        ids = record["train_ids"]
        gold = set(int(x) for x in record["gold"])
        xs.append(rows_kind(record, ids, "reliability"))
        ys.append(np.asarray([int(int(doc) in gold) for doc in ids]))
    x, y = np.vstack(xs), np.concatenate(ys)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1, class_weight="balanced", fit_intercept=True, solver="lbfgs",
        max_iter=2000, tol=1e-7, random_state=base.SEED
    ).fit(scaler.transform(x), y)
    if not np.isfinite(model.coef_).all():
        raise ValueError("Non-finite reliability coefficients")
    return scaler, model


def fit_ranker(records: list[dict[str, Any]], train: Iterable[int], kind: str) -> XGBRanker:
    xs, ys, groups = [], [], []
    for idx in train:
        record = records[int(idx)]
        ids = record["train_ids"]
        gold = set(int(x) for x in record["gold"])
        xs.append(rows_kind(record, ids, kind))
        ys.append(np.asarray([int(int(doc) in gold) for doc in ids]))
        groups.append(len(ids))
    model = XGBRanker(
        objective="rank:pairwise", n_estimators=300, max_depth=3,
        learning_rate=0.03, subsample=0.85, colsample_bytree=0.85,
        random_state=base.SEED, tree_method="hist", n_jobs=8,
    )
    model.fit(np.vstack(xs), np.concatenate(ys), group=np.asarray(groups))
    return model


def rank(record: dict[str, Any], model: Any, kind: str, scaler: StandardScaler | None = None) -> np.ndarray:
    ids = record["broad"]
    x = rows_kind(record, ids, kind)
    if scaler is not None:
        x = scaler.transform(x)
        scores = model.decision_function(x)
    else:
        scores = model.predict(x)
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite {kind} score for {record['qa_id']}")
    return ids[np.lexsort((np.arange(len(ids)), -scores))]


def cmd_capacity() -> None:
    records = joblib.load(base.CACHE / "stage2_records.joblib")
    ensure_graph_features(records)
    all_protocols = evaluation.protocols(records)
    selected = {
        "stratified": all_protocols["stratified"],
        "unseen_tool": all_protocols["unseen_tool"],
        "leave_source_out": all_protocols["leave_source_out"],
    }
    predictions = []
    for protocol, folds in selected.items():
        for fold, (train, test) in enumerate(folds):
            print(f"{protocol} fold {fold + 1}/{len(folds)}")
            scaler, gate = fit_reliability(records, train)
            lambdamart = fit_ranker(records, train, "base")
            graph = fit_ranker(records, train, "graph")
            for idx in test:
                record = records[int(idx)]
                rankings = {
                    "Reliability-gate": rank(record, gate, "reliability", scaler),
                    "LambdaMART": rank(record, lambdamart, "base"),
                    "LambdaMART-graph": rank(record, graph, "graph"),
                }
                for method, ranking in rankings.items():
                    metrics = evaluation.task_metrics(ranking, record["gold"], 5)
                    predictions.append(
                        {"protocol": protocol, "fold": fold, "qa_id": record["qa_id"],
                         "method": method, "top10_document_indices": [int(x) for x in ranking[:10]],
                         **metrics}
                    )
    with (base.OUTPUT / "capacity_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["protocol"], row["method"])].append(row)
    aggregates = [
        {"protocol": protocol, "method": method, "n": len(rows),
         **{metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("hit", "recall", "complete", "mrr", "ndcg")}}
        for (protocol, method), rows in sorted(grouped.items())
    ]
    report = {
        "conventions": {
            "reliability": "mean of top-50 BM25/dense Jaccard and four-feature top-minus-median separation",
            "lambdamart": "XGBRanker rank:pairwise; 300 depth-3 trees; eta=.03; row/column subsample=.85",
            "graph": "five BGE-nearest neighbours from unaugmented broad candidates only; six mean CE features plus mean/max neighbour cosine",
            "gold_isolation": "query reliability and neighbour sets depend only on retrieved candidates; inserted training gold may be a scored center but cannot alter test features",
            "historical_exact_settings_available": False,
        },
        "aggregates": aggregates,
    }
    base.write_json(base.OUTPUT / "capacity_report.json", report)
    print(json.dumps(report, indent=2))


def stratified_subset(records: list[dict[str, Any]], train: np.ndarray, fraction: float, repeat: int) -> np.ndarray:
    if fraction == 1.0:
        return train
    rng = np.random.default_rng(base.SEED + int(fraction * 1000) + repeat)
    by_level: dict[str, list[int]] = defaultdict(list)
    for idx in train:
        by_level[records[int(idx)]["task"]["complexity_metadata"]["level"]].append(int(idx))
    result = []
    for values in by_level.values():
        count = max(1, int(round(len(values) * fraction)))
        result.extend(rng.choice(values, size=count, replace=False).tolist())
    return np.asarray(sorted(result), dtype=np.int32)


def cmd_learning() -> None:
    records = joblib.load(base.CACHE / "stage2_records.joblib")
    folds = evaluation.protocols(records)["stratified"]
    output = []
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.0):
        repeats = range(1) if fraction == 1.0 else range(3)
        for repeat in repeats:
            metrics = []
            train_counts = []
            for fold, (train, test) in enumerate(folds):
                subset = stratified_subset(records, train, fraction, repeat)
                train_counts.append(len(subset))
                scale, model, _ = evaluation.fit_pairwise(records, subset)
                for idx in test:
                    record = records[int(idx)]
                    ranking = evaluation.rank_pairwise(record, scale, model)
                    metrics.append(evaluation.task_metrics(ranking, record["gold"], 5))
            output.append(
                {"fraction": fraction, "repeat": repeat, "mean_train_tasks": float(np.mean(train_counts)),
                 **{metric: float(np.mean([row[metric] for row in metrics]))
                    for metric in ("hit", "recall", "complete", "mrr", "ndcg")}}
            )
    summary = []
    for fraction in (0.10, 0.25, 0.50, 0.75, 1.0):
        rows = [row for row in output if row["fraction"] == fraction]
        summary.append(
            {"fraction": fraction, "repeats": len(rows),
             "mean_train_tasks": float(np.mean([row["mean_train_tasks"] for row in rows])),
             **{f"{metric}_{stat}": float(getattr(np, stat)([row[metric] for row in rows]))
                for metric in ("recall", "ndcg") for stat in ("mean", "std")}}
        )
    report = {
        "subsample_rule": "level-stratified without replacement; seed + 1000*fraction + repeat",
        "runs": output, "summary": summary,
    }
    base.write_json(base.OUTPUT / "learning_curve_report.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("capacity", "learning"))
    args = parser.parse_args()
    if args.command == "capacity":
        cmd_capacity()
    else:
        cmd_learning()


if __name__ == "__main__":
    main()
