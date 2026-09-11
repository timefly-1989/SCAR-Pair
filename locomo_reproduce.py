#!/usr/bin/env python3
"""Reproduce request-side retrieval on the official LoCoMo release."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import scar_pair_reproduce as base
from stage2_evaluate import task_metrics


LOCOMO = base.REPRO / "vendor" / "locomo"
LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
EVIDENCE_RE = re.compile(r"D(\d+):(\d+)")
MODEL_SPECS = {
    "bge-small": (base.BGE_CACHE, "BAAI/bge-small-en-v1.5", base.BGE_REVISION),
    "mgte": (
        Path.home() / ".cache/huggingface/hub/models--Alibaba-NLP--gte-multilingual-base/snapshots/9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
        "Alibaba-NLP/gte-multilingual-base",
        "9bbca17d9273fd0d03d5725c7a4b0f6b45142062",
    ),
    "bge-m3": (
        Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        "BAAI/bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
    ),
}


def parse_evidence(values: list[str]) -> set[str]:
    return {f"D{int(a)}:{int(b)}" for value in values for a, b in EVIDENCE_RE.findall(value)}


def load_data(style: str) -> tuple[list[str], list[dict[str, Any]], list[slice]]:
    raw = json.loads((LOCOMO / "data/locomo10.json").read_text(encoding="utf-8"))
    documents: list[str] = []
    questions: list[dict[str, Any]] = []
    slices: list[slice] = []
    for conversation_idx, sample in enumerate(raw):
        start = len(documents)
        id_to_local: dict[str, int] = {}
        sessions = sorted(
            (key for key in sample["conversation"]
             if key.startswith("session_") and not key.endswith("_date_time")),
            key=lambda key: int(key.split("_")[1]),
        )
        for session_key in sessions:
            session_number = int(session_key.split("_")[1])
            date = sample["conversation"].get(f"session_{session_number}_date_time", "")
            for turn in sample["conversation"][session_key]:
                normalized_id = f"D{int(turn['dia_id'].split(':')[0][1:])}:{int(turn['dia_id'].split(':')[1])}"
                id_to_local[normalized_id] = len(documents) - start
                text = str(turn.get("text", "")).strip()
                speaker = str(turn.get("speaker", "")).strip()
                if style == "text":
                    serialized = text
                elif style == "speaker":
                    serialized = f"{speaker}: {text}"
                elif style == "timestamped":
                    serialized = f"Date: {date}\nSpeaker: {speaker}\nText: {text}"
                elif style == "timestamp_plain":
                    serialized = f"{date}\n{speaker}: {text}"
                elif style == "date_text":
                    serialized = f"{date}\n{text}"
                else:
                    raise ValueError(style)
                documents.append(serialized)
        end = len(documents)
        slices.append(slice(start, end))
        for question_idx, qa in enumerate(sample["qa"]):
            evidence = parse_evidence(qa.get("evidence", []))
            gold = sorted(id_to_local[item] for item in evidence if item in id_to_local)
            if not gold:
                continue
            questions.append(
                {"sample_id": sample["sample_id"], "conversation_idx": conversation_idx,
                 "question_idx": question_idx, "question": qa["question"],
                 "category": int(qa["category"]), "gold": np.asarray(gold, dtype=np.int32)}
            )
    if len(documents) != 5882 or len(questions) != 1982:
        raise ValueError(f"LoCoMo audit mismatch: documents={len(documents)}, questions={len(questions)}")
    return documents, questions, slices


def load_model(name: str) -> SentenceTransformer:
    local, remote, revision = MODEL_SPECS[name]
    source = str(local) if local.exists() else remote
    model = SentenceTransformer(source, device="mps", trust_remote_code=(name == "mgte"),
                                revision=None if local.exists() else revision)
    model.max_seq_length = 512
    return model


def evaluate(
    documents: list[str], questions: list[dict[str, Any]], slices: list[slice],
    tokenizer: Callable[[str], list[str]], embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bm25_by_conversation = [
        BM25Okapi([tokenizer(text) for text in documents[sl]], k1=1.5, b=0.75)
        for sl in slices
    ]
    predictions: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    for q_idx, qa in enumerate(tqdm(questions, desc="LoCoMo ranking")):
        conv = qa["conversation_idx"]
        sl = slices[conv]
        bm25_score = np.asarray(
            bm25_by_conversation[conv].get_scores(tokenizer(qa["question"])), dtype=np.float32
        )
        dense_score = np.asarray(embeddings[sl] @ query_embeddings[q_idx], dtype=np.float32)
        bm25_order = base.stable_order(bm25_score)
        dense_order = base.stable_order(dense_score)
        hybrid_order = base.top_from_rrf(
            base.ranks_from_order(bm25_order), base.ranks_from_order(dense_order), len(bm25_order)
        )
        for method, ranking in (("BM25", bm25_order), ("Dense", dense_order), ("RRF", hybrid_order)):
            values = {str(k): task_metrics(ranking, qa["gold"], k) for k in (1, 5, 10, 20)}
            by_method[method].append(values["5"])
            predictions.append(
                {"sample_id": qa["sample_id"], "question_idx": qa["question_idx"],
                 "category": qa["category"], "method": method,
                 "top20_local_indices": [int(x) for x in ranking[:20]], "metrics_by_k": values}
            )
    aggregate = []
    for method, rows in by_method.items():
        aggregate.append(
            {"method": method, "n": len(rows),
             **{key: float(np.mean([row[key] for row in rows]))
                for key in ("hit", "recall", "complete", "mrr", "ndcg")}}
        )
    return predictions, aggregate


def run_variant(model_name: str, style: str, tokenizer_name: str) -> dict[str, Any]:
    documents, questions, slices = load_data(style)
    tokenizer = (
        (lambda text: text.lower().split()) if tokenizer_name == "whitespace"
        else (lambda text: base.TOKEN_RE.findall(text.lower()))
    )
    model = load_model(model_name)
    cache_prefix = f"locomo_{model_name}_{style}"
    doc_path = base.CACHE / f"{cache_prefix}_documents.npy"
    query_path = base.CACHE / f"{cache_prefix}_queries.npy"
    start = time.perf_counter()
    if doc_path.exists():
        doc_embeddings = np.load(doc_path)
    else:
        doc_embeddings = base.encode(model, documents, f"{model_name} LoCoMo documents", batch_size=64)
        np.save(doc_path, doc_embeddings)
    if query_path.exists():
        query_embeddings = np.load(query_path)
    else:
        query_embeddings = base.encode(
            model, [qa["question"] for qa in questions], f"{model_name} LoCoMo questions", batch_size=64
        )
        np.save(query_path, query_embeddings)
    predictions, aggregate = evaluate(
        documents, questions, slices, tokenizer, doc_embeddings, query_embeddings
    )
    label = f"{model_name}_{style}_{tokenizer_name}"
    with (base.OUTPUT / f"locomo_predictions_{label}.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source_commit": LOCOMO_COMMIT, "documents": len(documents),
        "questions": len(questions), "model": model_name,
        "model_revision": MODEL_SPECS[model_name][2], "max_length": 512,
        "document_style": style, "bm25_tokenizer": tokenizer_name,
        "elapsed_seconds_including_cached_load_and_ranking": time.perf_counter() - start,
        "aggregate_at_5": aggregate,
    }
    base.write_json(base.OUTPUT / f"locomo_report_{label}.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--style", choices=("text", "speaker", "timestamped", "timestamp_plain", "date_text"),
        default="timestamped"
    )
    parser.add_argument("--tokenizer", choices=("whitespace", "unicode"), default="whitespace")
    args = parser.parse_args()
    run_variant(args.model, args.style, args.tokenizer)


if __name__ == "__main__":
    main()
