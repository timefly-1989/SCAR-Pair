#!/usr/bin/env python3
"""Independent SCAR-Pair experiment reconstruction.

The original experiment implementation was not present in the manuscript
repository. This script implements the equations and protocols stated in the
paper, while recording every convention that the paper left unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduction"
VENDOR = REPRO / "vendor" / "Mem2ActBench"
CACHE = REPRO / "cache"
OUTPUT = REPRO / "output"

SOURCE_FILES = (
    "toolace_formatted_conversations.jsonl",
    "bfcl_formatted_conversations.jsonl",
    "oasst1_formatted_conversations.jsonl",
)
QA_FILE = "Mem2ActBench/qa_dataset.jsonl"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_CACHE = (
    Path.home()
    / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots"
    / BGE_REVISION
)
SEED = 20260730
TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_name(doc_id: str) -> str:
    if doc_id.startswith("toolace_"):
        return "ToolACE"
    if doc_id.startswith("oss_"):
        return "OASST1"
    return "BFCL-live"


def conversation_text(row: dict[str, Any]) -> str:
    """Serialize the text visible in a conversation.

    This explicit replication convention omits role labels and tool-call
    metadata, because the public benchmark's source facts point into textual
    turn content. Empty textual conversations are excluded.
    """

    parts = []
    for turn in row.get("conversation_history", []):
        content = turn.get("content")
        if content is not None and str(content).strip():
            parts.append(str(content).strip())
    return "\n".join(parts)


def tokenize(text: str) -> list[str]:
    # The public Mem2ActBench construction code uses lower().split() for its
    # BM25 index. Adopting the same rule minimizes an otherwise undocumented
    # preprocessing difference.
    return text.lower().split()


def properties(task: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    params = task.get("target_tool_schema", {}).get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    return list(props.items())


def query_text(task: dict[str, Any]) -> str:
    return str(task.get("query", "")).strip()


def tool_view(task: dict[str, Any]) -> str:
    schema = task.get("target_tool_schema", {}) or {}
    return (
        f"Request: {query_text(task)}\n"
        f"Tool name: {schema.get('name', '')}\n"
        f"Tool description: {schema.get('description', '')}"
    )


def tool_view_plain(task: dict[str, Any]) -> str:
    schema = task.get("target_tool_schema", {}) or {}
    return "\n".join(
        (query_text(task), str(schema.get("name", "")), str(schema.get("description", "")))
    )


def field_views(task: dict[str, Any]) -> list[str]:
    items = properties(task)
    effective = [(name, spec) for name, spec in items if "default" not in spec]
    if not effective:
        effective = items
    base = tool_view(task)
    return [
        f"{base}\nParameter name: {name}\n"
        f"Parameter description: {spec.get('description', '')}"
        for name, spec in effective
    ]


def schema_view(task: dict[str, Any]) -> str:
    base = tool_view(task)
    suffix = "".join(
        f"\nParameter name: {name}\nParameter description: {spec.get('description', '')}"
        for name, spec in properties(task)
    )
    return base + suffix


@dataclass
class Corpus:
    documents: list[dict[str, Any]]
    tasks_all: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    id_to_index: dict[str, int]
    empty_document_ids: list[str]


def load_corpus() -> Corpus:
    if not VENDOR.exists():
        raise FileNotFoundError(
            f"Missing {VENDOR}. Clone https://github.com/Cantaloupe-M/Mem2ActBench.git there."
        )
    documents: list[dict[str, Any]] = []
    empty: list[str] = []
    for relpath in SOURCE_FILES:
        for row in jsonl(VENDOR / relpath):
            text = conversation_text(row)
            if not text:
                empty.append(row["id"])
                continue
            documents.append(
                {
                    "id": row["id"],
                    "source": source_name(row["id"]),
                    "text": text,
                }
            )
    ids = [row["id"] for row in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate memory identifiers after filtering")
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(ids)}
    tasks_all = jsonl(VENDOR / QA_FILE)
    tasks = [row for row in tasks_all if row.get("source_conversation_ids")]
    missing = sorted(
        {
            doc_id
            for task in tasks
            for doc_id in task["source_conversation_ids"]
            if doc_id not in id_to_index
        }
    )
    if missing:
        raise ValueError(f"Gold source IDs missing from bank: {missing[:10]}")
    return Corpus(documents, tasks_all, tasks, id_to_index, empty)


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def audit_payload(corpus: Corpus) -> dict[str, Any]:
    levels_all = Counter(
        task.get("complexity_metadata", {}).get("level") for task in corpus.tasks_all
    )
    levels_evidence = Counter(
        task.get("complexity_metadata", {}).get("level") for task in corpus.tasks
    )
    tool_counts = Counter(
        task.get("target_tool_schema", {}).get("name") for task in corpus.tasks
    )
    gold_counts = Counter(len(task["source_conversation_ids"]) for task in corpus.tasks)
    file_hashes = {rel: sha256(VENDOR / rel) for rel in (*SOURCE_FILES, QA_FILE)}
    return {
        "source_commit": git_commit(VENDOR),
        "source_files_sha256": file_hashes,
        "raw_source_rows": sum(len(jsonl(VENDOR / rel)) for rel in SOURCE_FILES),
        "empty_text_rows_excluded": len(corpus.empty_document_ids),
        "empty_text_ids": corpus.empty_document_ids,
        "memory_documents": len(corpus.documents),
        "released_tasks": len(corpus.tasks_all),
        "evidence_tasks": len(corpus.tasks),
        "empty_source_task_ids": [
            task["qa_id"]
            for task in corpus.tasks_all
            if not task.get("source_conversation_ids")
        ],
        "levels_all": dict(sorted(levels_all.items())),
        "levels_evidence": dict(sorted(levels_evidence.items())),
        "gold_source_counts": {str(k): v for k, v in sorted(gold_counts.items())},
        "unique_tool_names_evidence": len(tool_counts),
        "singleton_tool_names_evidence": sum(value == 1 for value in tool_counts.values()),
        "python": sys.version,
        "platform": platform.platform(),
        "replication_conventions": {
            "conversation_serialization": "newline-joined nonempty content fields",
            "bm25_tokenizer": "lowercase whitespace split (matches public benchmark code)",
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "tie_break": "memory-bank position",
            "seed": SEED,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_audit() -> None:
    corpus = load_corpus()
    payload = audit_payload(corpus)
    write_json(OUTPUT / "data_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    expected = {
        "raw_source_rows": 19679,
        "empty_text_rows_excluded": 18,
        "memory_documents": 19661,
        "released_tasks": 400,
        "evidence_tasks": 391,
        "unique_tool_names_evidence": 244,
        "singleton_tool_names_evidence": 186,
    }
    failures = {
        key: (payload[key], value)
        for key, value in expected.items()
        if payload[key] != value
    }
    if failures:
        raise SystemExit(f"Audit mismatch: {failures}")


def stable_order(scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(scores.size), -scores))


def ranks_from_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(order.size, dtype=np.int32)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int32)
    return ranks


def minmax(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def union_in_rank_order(*ranked_lists: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    result: list[int] = []
    for ranked in ranked_lists:
        for raw_idx in ranked:
            idx = int(raw_idx)
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
    return np.asarray(result, dtype=np.int32)


def top_from_rrf(rank_a: np.ndarray, rank_b: np.ndarray, depth: int) -> np.ndarray:
    scores = 1.0 / (60.0 + rank_a) + 1.0 / (60.0 + rank_b)
    return stable_order(scores)[:depth]


def load_encoder() -> SentenceTransformer:
    model_source: str
    if BGE_CACHE.exists():
        model_source = str(BGE_CACHE)
    else:
        model_source = "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(model_source, device="mps",
                                revision=None if BGE_CACHE.exists() else BGE_REVISION)
    model.max_seq_length = 512
    return model


def encode(
    model: SentenceTransformer, texts: list[str], description: str, batch_size: int = 128
) -> np.ndarray:
    start = time.perf_counter()
    result = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"{description}: {len(texts)} texts in {time.perf_counter() - start:.1f}s")
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite embedding in {description}")
    return result


def candidate_metrics(candidates: Iterable[int], gold: set[int]) -> tuple[float, float, float]:
    hits = len(set(int(x) for x in candidates) & gold)
    return float(hits > 0), hits / len(gold), float(hits == len(gold))


def cmd_stage1() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    audit = audit_payload(corpus)
    write_json(OUTPUT / "data_audit.json", audit)

    texts = [row["text"] for row in corpus.documents]
    tokenized = [tokenize(text) for text in tqdm(texts, desc="Tokenize memory")]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    model = load_encoder()
    doc_embedding_path = CACHE / "bge_small_documents.npy"
    if doc_embedding_path.exists():
        # A normal in-memory array avoids spurious Accelerate/NumPy matmul
        # warnings observed with a memory-mapped array on Apple Silicon.
        doc_embeddings = np.load(doc_embedding_path)
        if doc_embeddings.shape != (len(texts), 384):
            raise ValueError(f"Unexpected cached embedding shape: {doc_embeddings.shape}")
    else:
        doc_embeddings = encode(model, texts, "BGE-small document encoding")
        np.save(doc_embedding_path, doc_embeddings)

    query_texts = [query_text(task) for task in corpus.tasks]
    tool_texts = [tool_view(task) for task in corpus.tasks]
    all_field_views: list[str] = []
    field_slices: list[tuple[int, int]] = []
    for task in corpus.tasks:
        start = len(all_field_views)
        all_field_views.extend(field_views(task))
        field_slices.append((start, len(all_field_views)))
    q_embeddings = encode(model, query_texts, "BGE-small request encoding")
    t_embeddings = encode(model, tool_texts, "BGE-small tool-view encoding")
    f_embeddings = encode(model, all_field_views, "BGE-small field-view encoding")

    records: list[dict[str, Any]] = []
    audits: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    pool_sizes: dict[str, list[int]] = defaultdict(list)
    for task_idx, task in enumerate(tqdm(corpus.tasks, desc="Stage-I ranking/features")):
        q = query_texts[task_idx]
        tv = tool_texts[task_idx]
        bq_raw = np.asarray(bm25.get_scores(tokenize(q)), dtype=np.float32)
        bt_raw = np.asarray(bm25.get_scores(tokenize(tv)), dtype=np.float32)
        cq_raw = np.asarray(doc_embeddings @ q_embeddings[task_idx], dtype=np.float32)
        ct_raw = np.asarray(doc_embeddings @ t_embeddings[task_idx], dtype=np.float32)
        if not all(np.isfinite(x).all() for x in (bq_raw, bt_raw, cq_raw, ct_raw)):
            raise ValueError(f"Non-finite first-stage score for {task['qa_id']}")

        obq, ocq, obt, oct = map(stable_order, (bq_raw, cq_raw, bt_raw, ct_raw))
        rbq, rcq, rbt, rct = map(ranks_from_order, (obq, ocq, obt, oct))
        rrf_q_200 = top_from_rrf(rbq, rcq, 200)
        rrf_s_200 = top_from_rrf(rbt, rct, 200)
        rrf_q_50 = top_from_rrf(rbq, rcq, 50)
        rrf_s_50 = top_from_rrf(rbt, rct, 50)
        strict = union_in_rank_order(obq[:200], ocq[:200], rrf_q_200)
        strict_narrow = union_in_rank_order(obq[:50], ocq[:50], rrf_q_50)
        broad = union_in_rank_order(obq[:200], ocq[:200], rrf_q_200, rrf_s_200)
        narrow = union_in_rank_order(obq[:50], ocq[:50], rrf_q_50, rrf_s_50)

        gold = {corpus.id_to_index[x] for x in task["source_conversation_ids"]}
        train_ids = union_in_rank_order(broad, np.asarray(sorted(gold), dtype=np.int32))
        wanted = train_ids

        first = field_slices[task_idx]
        f_bm25 = []
        f_dense = []
        for local, text in enumerate(all_field_views[first[0] : first[1]], start=first[0]):
            f_bm25.append(minmax(np.asarray(bm25.get_scores(tokenize(text)), dtype=np.float32))[wanted])
            f_dense.append(((np.asarray(doc_embeddings @ f_embeddings[local]) + 1.0) / 2.0)[wanted])
        if not f_bm25:
            raise ValueError(f"Zero-parameter tool outside stated setting: {task['qa_id']}")
        f_bm25_arr = np.stack(f_bm25)
        f_dense_arr = np.stack(f_dense)
        light = np.column_stack(
            [
                minmax(bq_raw)[wanted],
                (cq_raw[wanted] + 1.0) / 2.0,
                minmax(bt_raw)[wanted],
                (ct_raw[wanted] + 1.0) / 2.0,
                f_bm25_arr.max(axis=0),
                f_bm25_arr.mean(axis=0),
                f_dense_arr.max(axis=0),
                f_dense_arr.mean(axis=0),
                1.0 / rbq[wanted],
                1.0 / rcq[wanted],
                1.0 / rbt[wanted],
                1.0 / rct[wanted],
            ]
        ).astype(np.float32)

        audits["request-only depth-200 union"].append(candidate_metrics(strict, gold))
        audits["schema-aware depth-200 union"].append(candidate_metrics(broad, gold))
        audits["CE depth-50 union"].append(candidate_metrics(narrow, gold))
        pool_sizes["request-only depth-200 union"].append(len(strict))
        pool_sizes["schema-aware depth-200 union"].append(len(broad))
        pool_sizes["CE depth-50 union"].append(len(narrow))
        records.append(
            {
                "qa_id": task["qa_id"],
                "task": task,
                "gold": np.asarray(sorted(gold), dtype=np.int32),
                "strict": strict,
                "strict_narrow": strict_narrow,
                "broad": broad,
                "narrow": narrow,
                "train_ids": train_ids,
                "light": light,
                "request_orders": {"bm25": obq, "dense": ocq, "rrf": rrf_q_200},
                "schema_rrf_200": rrf_s_200,
            }
        )

    audit_rows = []
    manuscript = {
        "request-only depth-200 union": (377.4, 0.803, 0.773, 0.747),
        "schema-aware depth-200 union": (465.0, 0.859, 0.842, 0.829),
        "CE depth-50 union": (121.9, 0.762, 0.739, 0.714),
    }
    for name in manuscript:
        values = np.asarray(audits[name])
        reproduced = (
            float(np.mean(pool_sizes[name])),
            float(values[:, 0].mean()),
            float(values[:, 1].mean()),
            float(values[:, 2].mean()),
        )
        audit_rows.append(
            {
                "candidate_set": name,
                "reproduced": dict(zip(("mean_size", "hit", "recall", "complete"), reproduced)),
                "manuscript": dict(zip(("mean_size", "hit", "recall", "complete"), manuscript[name])),
                "difference": dict(
                    zip(
                        ("mean_size", "hit", "recall", "complete"),
                        np.asarray(reproduced) - np.asarray(manuscript[name]),
                    )
                ),
            }
        )
    stage1_manifest = {
        "data_audit": audit,
        "model": {"name": "BAAI/bge-small-en-v1.5", "revision": BGE_REVISION},
        "candidate_audit": audit_rows,
        "field_views": len(all_field_views),
        "record_count": len(records),
    }
    joblib.dump(records, CACHE / "stage1_records.joblib", compress=3)
    write_json(OUTPUT / "stage1_report.json", stage1_manifest)
    print(json.dumps(stage1_manifest["candidate_audit"], indent=2))


def cmd_calibrate() -> None:
    """Compare unresolved Stage-I text conventions against manuscript audits."""

    corpus = load_corpus()
    texts = [row["text"] for row in corpus.documents]
    doc_embeddings = np.load(CACHE / "bge_small_documents.npy")
    model = load_encoder()
    q_texts = [query_text(task) for task in corpus.tasks]
    q_embeddings = encode(model, q_texts, "Calibration request encoding")
    q_dense = [np.asarray(doc_embeddings @ embedding, dtype=np.float32) for embedding in q_embeddings]
    gold_sets = [
        {corpus.id_to_index[x] for x in task["source_conversation_ids"]}
        for task in corpus.tasks
    ]
    tokenizer_variants = {
        "whitespace": lambda text: text.lower().split(),
        "unicode_words": lambda text: TOKEN_RE.findall(text.lower()),
    }
    tool_variants = {
        "labelled": [tool_view(task) for task in corpus.tasks],
        "plain": [tool_view_plain(task) for task in corpus.tasks],
    }
    tool_embeddings = {
        name: encode(model, values, f"Calibration {name} tool-view encoding")
        for name, values in tool_variants.items()
    }
    output: list[dict[str, Any]] = []
    for tokenizer_name, tokenizer in tokenizer_variants.items():
        bm25 = BM25Okapi([tokenizer(text) for text in texts], k1=1.5, b=0.75)
        q_cache = []
        request_audits = []
        request_sizes = []
        for idx, text in enumerate(tqdm(q_texts, desc=f"{tokenizer_name}: request")):
            bq = np.asarray(bm25.get_scores(tokenizer(text)), dtype=np.float32)
            cq = q_dense[idx]
            obq, ocq = stable_order(bq), stable_order(cq)
            rbq, rcq = ranks_from_order(obq), ranks_from_order(ocq)
            rrf = top_from_rrf(rbq, rcq, 200)
            strict = union_in_rank_order(obq[:200], ocq[:200], rrf)
            request_audits.append(candidate_metrics(strict, gold_sets[idx]))
            request_sizes.append(len(strict))
            q_cache.append((obq, ocq, rbq, rcq, strict))
        req = np.asarray(request_audits)
        for tool_name, tool_texts in tool_variants.items():
            broad_audits = []
            broad_sizes = []
            narrow_audits = []
            narrow_sizes = []
            for idx, text in enumerate(tqdm(tool_texts, desc=f"{tokenizer_name}/{tool_name}")):
                bt = np.asarray(bm25.get_scores(tokenizer(text)), dtype=np.float32)
                ct = np.asarray(doc_embeddings @ tool_embeddings[tool_name][idx], dtype=np.float32)
                obt, oct = stable_order(bt), stable_order(ct)
                rbt, rct = ranks_from_order(obt), ranks_from_order(oct)
                obq, ocq, rbq, rcq, strict = q_cache[idx]
                rrf_s = top_from_rrf(rbt, rct, 200)
                broad = union_in_rank_order(strict, rrf_s)
                narrow = union_in_rank_order(
                    obq[:50], ocq[:50], top_from_rrf(rbq, rcq, 50),
                    top_from_rrf(rbt, rct, 50),
                )
                broad_audits.append(candidate_metrics(broad, gold_sets[idx]))
                broad_sizes.append(len(broad))
                narrow_audits.append(candidate_metrics(narrow, gold_sets[idx]))
                narrow_sizes.append(len(narrow))
            broad_values = np.asarray(broad_audits)
            narrow_values = np.asarray(narrow_audits)
            output.append(
                {
                    "tokenizer": tokenizer_name,
                    "tool_view": tool_name,
                    "request": [float(np.mean(request_sizes)), *req.mean(axis=0).tolist()],
                    "broad": [float(np.mean(broad_sizes)), *broad_values.mean(axis=0).tolist()],
                    "narrow": [float(np.mean(narrow_sizes)), *narrow_values.mean(axis=0).tolist()],
                }
            )
    write_json(OUTPUT / "stage1_convention_calibration.json", output)
    print(json.dumps(output, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "calibrate", "stage1"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit":
        cmd_audit()
    elif args.command == "calibrate":
        cmd_calibrate()
    elif args.command == "stage1":
        cmd_stage1()


if __name__ == "__main__":
    main()
