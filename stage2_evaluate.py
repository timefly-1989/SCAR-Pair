#!/usr/bin/env python3
"""Cross-encoder scoring and fold-local evaluation for the SCAR reconstruction."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import scar_pair_reproduce as base


CE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CE_CACHE = (
    Path.home()
    / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots"
    / CE_REVISION
)
FEATURE_NAMES = (
    "bm25_q", "dense_q", "bm25_tool", "dense_tool",
    "bm25_field_max", "bm25_field_mean", "dense_field_max", "dense_field_mean",
    "rr_bm25_q", "rr_dense_q", "rr_bm25_tool", "rr_dense_tool",
    "ce_q", "rr_ce_q", "available_ce_q", "ce_schema", "rr_ce_schema",
    "available_ce_schema",
)
EPSILON = 1e-6

# NumPy 2.0 on the test Apple Silicon runtime sometimes reports stale floating
# point status flags after an otherwise finite Accelerate-backed matmul. Every
# feature, coefficient, and ranking score is asserted finite below, so suppress
# only that repeated warning rather than weakening the numerical checks.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def load_records() -> list[dict[str, Any]]:
    path = base.CACHE / "stage1_records.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scar_pair_reproduce.py stage1 first")
    return joblib.load(path)


def load_ce() -> CrossEncoder:
    source = str(CE_CACHE) if CE_CACHE.exists() else "cross-encoder/ms-marco-MiniLM-L6-v2"
    return CrossEncoder(source, max_length=512, device="mps",
                        revision=None if CE_CACHE.exists() else CE_REVISION)


def score_flat(
    model: CrossEncoder,
    records: list[dict[str, Any]],
    documents: list[str],
    pool_key: str,
    view: str,
) -> tuple[list[np.ndarray], float]:
    pairs: list[tuple[str, str]] = []
    slices: list[tuple[int, int]] = []
    for record in records:
        start = len(pairs)
        left = base.query_text(record["task"]) if view == "q" else base.schema_view(record["task"])
        pairs.extend((left, documents[int(idx)]) for idx in record[pool_key])
        slices.append((start, len(pairs)))
    start_time = time.perf_counter()
    scores = np.asarray(
        model.predict(
            pairs,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    ).reshape(-1)
    elapsed = time.perf_counter() - start_time
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError(f"Invalid {view}/{pool_key} CE scores: {scores.shape}")
    return [scores[start:end] for start, end in slices], elapsed


def extend_view(
    train_ids: np.ndarray,
    pool: np.ndarray,
    pool_scores: np.ndarray,
) -> np.ndarray:
    floor = float(np.min(pool_scores)) - EPSILON
    features = np.zeros((len(train_ids), 3), dtype=np.float32)
    features[:, 0] = floor
    positions = {int(doc): pos for pos, doc in enumerate(train_ids)}
    order = np.lexsort((np.arange(len(pool_scores)), -pool_scores))
    ranks = np.empty(len(pool_scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(pool_scores) + 1)
    for local, raw_doc in enumerate(pool):
        pos = positions.get(int(raw_doc))
        if pos is not None:
            features[pos] = (pool_scores[local], 1.0 / ranks[local], 1.0)
    return features


def cmd_score(include_broad: bool) -> None:
    records = load_records()
    corpus = base.load_corpus()
    documents = [row["text"] for row in corpus.documents]
    model = load_ce()
    pools = ["narrow"] + (["broad"] if include_broad else [])
    timing: dict[str, float] = {}
    pair_counts: dict[str, int] = {}
    for pool_key in pools:
        q_scores, q_time = score_flat(model, records, documents, pool_key, "q")
        s_scores, s_time = score_flat(model, records, documents, pool_key, "schema")
        timing[f"{pool_key}_q_seconds"] = q_time
        timing[f"{pool_key}_schema_seconds"] = s_time
        pair_counts[pool_key] = int(sum(len(record[pool_key]) for record in records))
        for record, q_score, s_score in zip(records, q_scores, s_scores):
            record[f"ce_{pool_key}"] = np.column_stack(
                (
                    extend_view(record["train_ids"], record[pool_key], q_score),
                    extend_view(record["train_ids"], record[pool_key], s_score),
                )
            ).astype(np.float32)
            if pool_key == "narrow":
                strict_pool = record["strict_narrow"]
                pool_position = {int(doc): i for i, doc in enumerate(record["narrow"])}
                strict_scores = np.asarray([q_score[pool_position[int(doc)]] for doc in strict_pool])
                record["ce_query_narrow"] = extend_view(
                    record["train_ids"], strict_pool, strict_scores
                )
    joblib.dump(records, base.CACHE / "stage2_records.joblib", compress=3)
    report = {
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "revision": CE_REVISION,
        "max_length": 512,
        "batch_size": 128,
        "device": "mps",
        "score_floor_epsilon": EPSILON,
        "candidate_pairs_per_view": pair_counts,
        "timing": timing,
    }
    base.write_json(base.OUTPUT / "stage2_report.json", report)
    print(json.dumps(report, indent=2))


def combined(record: dict[str, Any], ce_key: str = "ce_narrow") -> np.ndarray:
    return np.column_stack((record["light"], record[ce_key])).astype(np.float32)


def rows_for(
    record: dict[str, Any], ids: np.ndarray, kind: str
) -> np.ndarray:
    pos = {int(doc): i for i, doc in enumerate(record["train_ids"])}
    take = np.asarray([pos[int(doc)] for doc in ids], dtype=np.int32)
    if kind == "light":
        values = record["light"]
    elif kind == "all":
        values = combined(record)
    elif kind == "ce":
        values = record["ce_narrow"]
    elif kind == "ce_broad":
        values = record["ce_broad"]
    elif kind == "q_light":
        values = record["light"][:, [0, 1, 8, 9]]
    elif kind == "q_fuse":
        values = np.column_stack(
            (record["light"][:, [0, 1, 8, 9]], record["ce_query_narrow"])
        )
    elif kind == "q_ce":
        values = record["ce_query_narrow"]
    elif kind == "q_ce_broad":
        values = record["ce_broad"][:, :3]
    else:
        raise KeyError(kind)
    return np.asarray(values[take], dtype=np.float32)


def augmented_ids(record: dict[str, Any], pool_key: str) -> np.ndarray:
    return base.union_in_rank_order(record[pool_key], record["gold"])


def fit_pointwise(
    records: list[dict[str, Any]], train_indices: Iterable[int], pool_key: str, kind: str
) -> tuple[StandardScaler, LogisticRegression]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for idx in train_indices:
        record = records[int(idx)]
        ids = augmented_ids(record, pool_key)
        gold = set(int(x) for x in record["gold"])
        xs.append(rows_for(record, ids, kind))
        ys.append(np.asarray([int(int(doc) in gold) for doc in ids], dtype=np.int8))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        fit_intercept=True,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-7,
        random_state=base.SEED,
    ).fit(scaler.transform(x), y)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("Non-finite pointwise logistic coefficients")
    return scaler, model


def fit_pairwise(
    records: list[dict[str, Any]], train_indices: Iterable[int]
) -> tuple[np.ndarray, LogisticRegression, dict[str, float]]:
    diffs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    query_count = 0
    for idx in train_indices:
        record = records[int(idx)]
        ids = record["train_ids"]
        gold = set(int(x) for x in record["gold"])
        x = rows_for(record, ids, "all")
        positive = x[[int(doc) in gold for doc in ids]]
        negative = x[[int(doc) not in gold for doc in ids]]
        if not len(positive) or not len(negative):
            raise ValueError(f"Bad pair set for {record['qa_id']}")
        raw = (positive[:, None, :] - negative[None, :, :]).reshape(-1, x.shape[1])
        mass = 1.0 / (2.0 * len(raw))
        diffs.extend((raw, -raw))
        labels.extend((np.ones(len(raw), dtype=np.int8), np.zeros(len(raw), dtype=np.int8)))
        weights.extend((np.full(len(raw), mass), np.full(len(raw), mass)))
        query_count += 1
    delta = np.vstack(diffs).astype(np.float32)
    y = np.concatenate(labels)
    sample_weight = np.concatenate(weights)
    scale = delta.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
        tol=1e-7,
        random_state=base.SEED,
    ).fit(delta / scale, y, sample_weight=sample_weight)
    if not np.isfinite(scale).all() or not np.isfinite(model.coef_).all():
        raise ValueError("Non-finite pairwise scale or coefficients")
    diagnostics = {
        "mirrored_rows": int(len(delta)),
        "sample_weight_sum": float(sample_weight.sum()),
        "training_queries": query_count,
        "iterations": int(model.n_iter_[0]),
    }
    return scale, model, diagnostics


def rank_pointwise(
    record: dict[str, Any], pool_key: str, kind: str,
    scaler: StandardScaler, model: LogisticRegression,
) -> np.ndarray:
    ids = record[pool_key]
    scores = model.decision_function(scaler.transform(rows_for(record, ids, kind)))
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite pointwise ranking scores for {record['qa_id']}")
    return ids[np.lexsort((np.arange(len(ids)), -scores))]


def rank_pairwise(
    record: dict[str, Any], scale: np.ndarray, model: LogisticRegression
) -> np.ndarray:
    ids = record["broad"]
    scores = model.decision_function(rows_for(record, ids, "all") / scale)
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite pairwise ranking scores for {record['qa_id']}")
    return ids[np.lexsort((np.arange(len(ids)), -scores))]


def task_metrics(ranking: np.ndarray, gold_values: np.ndarray, k: int = 5) -> dict[str, float]:
    gold = set(int(x) for x in gold_values)
    top = [int(x) for x in ranking[:k]]
    rel = np.asarray([int(doc in gold) for doc in top], dtype=np.float64)
    found = int(rel.sum())
    rr = float(1.0 / (np.flatnonzero(rel)[0] + 1)) if found else 0.0
    discount = 1.0 / np.log2(np.arange(2, 2 + len(rel)))
    ideal = float((1.0 / np.log2(np.arange(2, 2 + min(k, len(gold))))).sum())
    return {
        "hit": float(found > 0),
        "recall": found / len(gold),
        "complete": float(found == len(gold)),
        "mrr": rr,
        "ndcg": float((rel * discount).sum() / ideal),
    }


def shared_source_groups(records: list[dict[str, Any]]) -> np.ndarray:
    parent = list(range(len(records)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    owners: dict[int, int] = {}
    for i, record in enumerate(records):
        for doc in record["gold"]:
            doc = int(doc)
            if doc in owners:
                union(i, owners[doc])
            else:
                owners[doc] = i
    roots = [find(i) for i in range(len(records))]
    remap = {root: j for j, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([remap[root] for root in roots], dtype=np.int32)


def protocols(records: list[dict[str, Any]]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    indices = np.arange(len(records))
    levels = np.asarray([r["task"]["complexity_metadata"]["level"] for r in records])
    strat = list(StratifiedKFold(5, shuffle=True, random_state=base.SEED).split(indices, levels))
    tools = np.asarray([r["task"]["target_tool_schema"]["name"] for r in records])
    unseen = list(GroupKFold(5).split(indices, groups=tools))
    components = shared_source_groups(records)
    shared = list(GroupKFold(5).split(indices, groups=components))
    corpus = base.load_corpus()
    source_by_doc = {i: row["source"] for i, row in enumerate(corpus.documents)}
    lso: list[tuple[np.ndarray, np.ndarray]] = []
    for heldout in ("ToolACE", "BFCL-live", "OASST1"):
        source_sets = [set(source_by_doc[int(doc)] for doc in r["gold"]) for r in records]
        train = np.asarray([i for i, sources in enumerate(source_sets) if heldout not in sources])
        test = np.asarray([i for i, sources in enumerate(source_sets) if sources == {heldout}])
        lso.append((train, test))
    return {"stratified": strat, "unseen_tool": unseen, "shared_source": shared, "leave_source_out": lso}


POINTWISE = {
    "Calibrated-Q": ("broad", "q_light"),
    "Query-Fuse-strict": ("strict", "q_fuse"),
    "Query-Fuse-schema-pool": ("broad", "q_fuse"),
    "SCAR-Light": ("broad", "light"),
    "CE-Calibrated": ("narrow", "ce"),
    "SCAR-Fuse": ("broad", "all"),
}
BROAD_POINTWISE = {
    "CE-Query-Calibrated-Broad": ("broad", "q_ce_broad"),
    "CE-Calibrated-Broad": ("broad", "ce_broad"),
}


def fixed_rankings(record: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {
        "BM25-Q": record["request_orders"]["bm25"],
        "Dense-Q": record["request_orders"]["dense"],
        "Hybrid-Q": record["request_orders"]["rrf"],
        "Schema-Hybrid": record["schema_rrf_200"],
    }
    for name, pool, column in (("CE-Q", "narrow", 0), ("CE-Schema", "narrow", 3)):
        ids = record[pool]
        values = rows_for(record, ids, "ce")[:, column]
        result[name] = ids[np.lexsort((np.arange(len(ids)), -values))]
    if "ce_broad" in record:
        for name, column in (("CE-Query-Broad", 0), ("CE-Schema-Broad", 3)):
            ids = record["broad"]
            values = rows_for(record, ids, "ce_broad")[:, column]
            result[name] = ids[np.lexsort((np.arange(len(ids)), -values))]
    return result


def cmd_evaluate() -> None:
    path = base.CACHE / "stage2_records.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run this script score first")
    records: list[dict[str, Any]] = joblib.load(path)
    corpus = base.load_corpus()
    document_ids = [row["id"] for row in corpus.documents]
    has_broad = all("ce_broad" in record for record in records)
    methods = dict(POINTWISE)
    if has_broad:
        methods.update(BROAD_POINTWISE)
    fold_sets = protocols(records)
    predictions: list[dict[str, Any]] = []
    fold_manifest: dict[str, Any] = {}
    fit_diagnostics: list[dict[str, Any]] = []
    for protocol_name, folds in fold_sets.items():
        fold_manifest[protocol_name] = []
        for fold, (train, test) in enumerate(folds):
            print(f"{protocol_name} fold {fold + 1}/{len(folds)}: train={len(train)}, test={len(test)}")
            fold_manifest[protocol_name].append(
                {"fold": fold, "train_qa_ids": [records[i]["qa_id"] for i in train],
                 "test_qa_ids": [records[i]["qa_id"] for i in test]}
            )
            fitted = {
                name: (*fit_pointwise(records, train, pool, kind), pool, kind)
                for name, (pool, kind) in methods.items()
            }
            scale, pair_model, diag = fit_pairwise(records, train)
            fit_diagnostics.append({"protocol": protocol_name, "fold": fold, "method": "SCAR-Pair", **diag})
            for idx in tqdm(test, desc=f"{protocol_name}/{fold}"):
                record = records[int(idx)]
                rankings = fixed_rankings(record)
                for name, (scaler, model, pool, kind) in fitted.items():
                    rankings[name] = rank_pointwise(record, pool, kind, scaler, model)
                rankings["SCAR-Pair"] = rank_pairwise(record, scale, pair_model)
                for method, ranking in rankings.items():
                    metrics = task_metrics(ranking, record["gold"])
                    metrics_by_k = {
                        str(k): task_metrics(ranking, record["gold"], k)
                        for k in (1, 3, 5, 10)
                    }
                    predictions.append(
                        {"protocol": protocol_name, "fold": fold, "qa_id": record["qa_id"],
                         "method": method, "top10_document_indices": [int(x) for x in ranking[:10]],
                         "top10_document_ids": [document_ids[int(x)] for x in ranking[:10]],
                         "metrics_by_k": metrics_by_k, **metrics}
                    )
    base.write_json(base.OUTPUT / "fold_assignments.json", fold_manifest)
    with (base.OUTPUT / "retrieval_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["protocol"], row["method"])].append(row)
    aggregates = []
    for (protocol, method), rows in sorted(grouped.items()):
        aggregates.append(
            {"protocol": protocol, "method": method, "n": len(rows),
             **{metric: float(np.mean([r[metric] for r in rows]))
                for metric in ("hit", "recall", "complete", "mrr", "ndcg")}}
        )
    aggregates_by_k = []
    for (protocol, method), rows in sorted(grouped.items()):
        for k in (1, 3, 5, 10):
            values = [row["metrics_by_k"][str(k)] for row in rows]
            aggregates_by_k.append(
                {"protocol": protocol, "method": method, "k": k, "n": len(rows),
                 **{metric: float(np.mean([r[metric] for r in values]))
                    for metric in ("hit", "recall", "complete", "mrr", "ndcg")}}
            )
    lso_source_macro = []
    for method in sorted({row["method"] for row in predictions if row["protocol"] == "leave_source_out"}):
        source_means = []
        for fold in range(3):
            subset = [row for row in predictions if row["protocol"] == "leave_source_out"
                      and row["method"] == method and row["fold"] == fold]
            source_means.append(
                {metric: float(np.mean([row[metric] for row in subset]))
                 for metric in ("hit", "recall", "complete", "mrr", "ndcg")}
            )
        lso_source_macro.append(
            {"method": method, "sources": 3,
             **{metric: float(np.mean([row[metric] for row in source_means]))
                for metric in ("hit", "recall", "complete", "mrr", "ndcg")}}
        )
    manuscript_stratified = {
        "BM25-Q": (0.292, 0.277, 0.266, 0.212, 0.222),
        "Dense-Q": (0.312, 0.290, 0.271, 0.219, 0.230),
        "Hybrid-Q": (0.350, 0.334, 0.320, 0.231, 0.251),
        "Schema-Hybrid": (0.338, 0.314, 0.297, 0.227, 0.243),
        "CE-Q": (0.419, 0.396, 0.376, 0.297, 0.313),
        "CE-Schema": (0.450, 0.429, 0.409, 0.305, 0.330),
        "Calibrated-Q": (0.396, 0.371, 0.348, 0.258, 0.279),
        "SCAR-Light": (0.412, 0.389, 0.366, 0.297, 0.312),
        "CE-Calibrated": (0.468, 0.443, 0.422, 0.318, 0.341),
        "SCAR-Fuse": (0.473, 0.446, 0.419, 0.335, 0.354),
        "SCAR-Pair": (None, 0.465, 0.440, None, 0.371),
    }
    lookup = {(r["protocol"], r["method"]): r for r in aggregates}
    comparison = []
    for method, expected in manuscript_stratified.items():
        reproduced = lookup.get(("stratified", method))
        if reproduced:
            comparison.append(
                {"method": method,
                 "reproduced": {m: reproduced[m] for m in ("hit", "recall", "complete", "mrr", "ndcg")},
                 "manuscript": dict(zip(("hit", "recall", "complete", "mrr", "ndcg"), expected))}
            )
    report = {
        "conventions": {
            "pointwise_scaler": "unweighted StandardScaler over fold-training candidates",
            "pointwise_estimator": "class-balanced lbfgs logistic, C=1, intercept=True",
            "pairwise_scale": "unweighted population SD over mirrored differences",
            "pairwise_estimator": "query-weighted mirrored lbfgs logistic, C=1, no intercept",
            "score_ties": "candidate-pool order",
            "unresolved_original_settings": True,
        },
        "feature_names": FEATURE_NAMES,
        "fit_diagnostics": fit_diagnostics,
        "aggregates": aggregates,
        "aggregates_by_k": aggregates_by_k,
        "leave_source_out_source_macro": lso_source_macro,
        "stratified_manuscript_comparison": comparison,
    }
    base.write_json(base.OUTPUT / "retrieval_report.json", report)
    print(json.dumps([r for r in aggregates if r["protocol"] == "stratified"], indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--include-broad", action="store_true")
    sub.add_parser("evaluate")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "score":
        cmd_score(args.include_broad)
    else:
        cmd_evaluate()


if __name__ == "__main__":
    main()
