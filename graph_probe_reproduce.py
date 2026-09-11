#!/usr/bin/env python3
"""Controlled relation-graph probe over fixed SCAR-Fuse top-100 nodes.

The manuscript omits its rank-prior temperature, edge normalization, isolated
node rule, and retained field profiles.  This executable probe records explicit
independent choices for each of those items instead of claiming an exact match.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import joblib
import numpy as np

import scar_pair_reproduce as base
import stage2_evaluate as stage2


K_VALUES = (3, 5, 10, 20)
ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30)
PRIOR_TAU = 20.0
PROTOCOLS = ("stratified", "unseen_tool", "shared_source", "leave_source_out")


def cosine_nonnegative(x: np.ndarray, center: bool = False) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if center:
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        values = (values - values.mean(axis=0)) / scale
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norm, out=np.zeros_like(values), where=norm > 0)
    similarity = np.maximum(normalized @ normalized.T, 0.0)
    np.fill_diagonal(similarity, 0.0)
    return similarity


def transition(similarity: np.ndarray, k: int) -> np.ndarray:
    n = len(similarity)
    directed = np.zeros_like(similarity)
    for row in range(n):
        order = np.lexsort((np.arange(n), -similarity[row]))
        keep = [idx for idx in order if idx != row and similarity[row, idx] > 0][:k]
        directed[row, keep] = similarity[row, keep]
    weights = np.maximum(directed, directed.T)
    row_sum = weights.sum(axis=1)
    isolated = np.flatnonzero(row_sum == 0)
    weights[isolated, isolated] = 1.0
    row_sum = weights.sum(axis=1)
    return weights / row_sum[:, None]


def diffuse(prior: np.ndarray, matrix: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    h = prior.copy()
    for iteration in range(1, 51):
        updated = (1.0 - alpha) * prior + alpha * (matrix.T @ h)
        if float(np.max(np.abs(updated - h))) < 1e-10:
            return updated, iteration
        h = updated
    return h, 50


def main() -> None:
    records = joblib.load(base.CACHE / "stage2_records.joblib")
    documents = np.load(base.CACHE / "bge_small_documents.npy")
    fold_sets = stage2.protocols(records)
    predictions = []
    iteration_counts = defaultdict(list)
    for protocol in PROTOCOLS:
        for fold, (train, test) in enumerate(fold_sets[protocol]):
            print(f"{protocol} fold {fold + 1}/{len(fold_sets[protocol])}")
            scaler, model = stage2.fit_pointwise(records, train, "broad", "all")
            for raw_idx in test:
                record = records[int(raw_idx)]
                ranking = stage2.rank_pointwise(record, "broad", "all", scaler, model)
                nodes = ranking[:100]
                base_metrics = stage2.task_metrics(nodes, record["gold"])
                predictions.append({
                    "protocol": protocol, "fold": fold, "qa_id": record["qa_id"],
                    "graph": "none", "k_graph": None, "alpha": 0.0,
                    "top10_document_indices": [int(x) for x in nodes[:10]], **base_metrics,
                })
                rank = np.arange(len(nodes), dtype=np.float64)
                prior = np.exp(-rank / PRIOR_TAU); prior /= prior.sum()
                semantic = cosine_nonnegative(documents[nodes])
                field = cosine_nonnegative(stage2.rows_for(record, nodes, "light")[:, 4:8], center=True)
                similarities = {
                    "semantic": semantic,
                    "field": field,
                    "combined": 0.5 * (semantic + field),
                }
                for graph, similarity in similarities.items():
                    for k in K_VALUES:
                        matrix = transition(similarity, k)
                        for alpha in ALPHAS:
                            scores, iterations = diffuse(prior, matrix, alpha)
                            order = np.lexsort((np.arange(len(nodes)), -scores))
                            result = nodes[order]
                            metrics = stage2.task_metrics(result, record["gold"])
                            iteration_counts[(graph, k, alpha)].append(iterations)
                            predictions.append({
                                "protocol": protocol, "fold": fold, "qa_id": record["qa_id"],
                                "graph": graph, "k_graph": k, "alpha": alpha,
                                "top10_document_indices": [int(x) for x in result[:10]], **metrics,
                            })
    with (base.OUTPUT / "graph_probe_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped = defaultdict(list)
    for row in predictions:
        grouped[(row["protocol"], row["graph"], row["k_graph"], row["alpha"])].append(row)
    results = []
    for (protocol, graph, k, alpha), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        results.append({
            "protocol": protocol, "graph": graph, "k_graph": k, "alpha": alpha,
            "tasks": len(rows),
            **{metric: float(np.mean([row[metric] for row in rows]))
               for metric in ("hit", "recall", "complete", "mrr", "ndcg")},
        })
    report = {
        "status": "independent reconstruction; original graph conventions were not released",
        "node_set": "fold-local SCAR-Fuse top 100",
        "rank_prior": f"normalized exp(-(rank-1)/tau), tau={PRIOR_TAU:g}",
        "semantic_similarity": "positive part of cosine between normalized BGE-small document vectors",
        "field_profile": "within-node standardized four aggregated field-support features (lexical max/mean, dense max/mean)",
        "symmetrization": "elementwise maximum of directed kNN weights",
        "transition": "row normalization; isolated nodes receive a self-loop",
        "diffusion": "at most 50 iterations; max absolute change below 1e-10",
        "mean_iterations": [
            {"graph": key[0], "k_graph": key[1], "alpha": key[2], "mean": float(np.mean(value))}
            for key, value in sorted(iteration_counts.items())
        ],
        "results": results,
    }
    base.write_json(base.OUTPUT / "graph_probe_report.json", report)
    fixed = [row for row in results if row["graph"] in {"none", "semantic", "combined"}
             and (row["graph"] == "none" or (row["k_graph"] == 5 and row["alpha"] == 0.1))]
    print(json.dumps(fixed, indent=2))


if __name__ == "__main__":
    main()
