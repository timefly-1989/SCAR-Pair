#!/usr/bin/env python3
"""Evaluate role-labelled conversation serialization as a replication choice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import scar_pair_reproduce as scar


def role_text(row: dict) -> str:
    parts = []
    for turn in row.get("conversation_history", []):
        content = turn.get("content")
        if content is not None and str(content).strip():
            parts.append(f"{turn.get('role', 'unknown')}: {str(content).strip()}")
    return "\n".join(parts)


def main() -> None:
    corpus = scar.load_corpus()
    texts = []
    for relpath in scar.SOURCE_FILES:
        for row in scar.jsonl(scar.VENDOR / relpath):
            text = role_text(row)
            if text:
                texts.append(text)
    cache = scar.CACHE / "bge_small_documents_role.npy"
    model = scar.load_encoder()
    if cache.exists():
        document_embeddings = np.load(cache)
    else:
        document_embeddings = scar.encode(model, texts, "Role-labelled document encoding")
        np.save(cache, document_embeddings)
    q_texts = [scar.query_text(task) for task in corpus.tasks]
    t_texts = [scar.tool_view_plain(task) for task in corpus.tasks]
    q_embeddings = scar.encode(model, q_texts, "Request encoding")
    t_embeddings = scar.encode(model, t_texts, "Plain tool-view encoding")
    gold = [
        {corpus.id_to_index[x] for x in task["source_conversation_ids"]}
        for task in corpus.tasks
    ]
    results = []
    for tokenizer_name, tokenizer in {
        "whitespace": lambda value: value.lower().split(),
        "unicode_words": lambda value: scar.TOKEN_RE.findall(value.lower()),
    }.items():
        bm25 = BM25Okapi([tokenizer(text) for text in texts], k1=1.5, b=0.75)
        request_metrics, broad_metrics, narrow_metrics = [], [], []
        request_sizes, broad_sizes, narrow_sizes = [], [], []
        for idx in tqdm(range(len(corpus.tasks)), desc=tokenizer_name):
            bq = np.asarray(bm25.get_scores(tokenizer(q_texts[idx])), dtype=np.float32)
            bt = np.asarray(bm25.get_scores(tokenizer(t_texts[idx])), dtype=np.float32)
            cq = np.asarray(document_embeddings @ q_embeddings[idx], dtype=np.float32)
            ct = np.asarray(document_embeddings @ t_embeddings[idx], dtype=np.float32)
            obq, ocq, obt, oct = map(scar.stable_order, (bq, cq, bt, ct))
            rbq, rcq, rbt, rct = map(scar.ranks_from_order, (obq, ocq, obt, oct))
            rq200 = scar.top_from_rrf(rbq, rcq, 200)
            rs200 = scar.top_from_rrf(rbt, rct, 200)
            strict = scar.union_in_rank_order(obq[:200], ocq[:200], rq200)
            broad = scar.union_in_rank_order(strict, rs200)
            narrow = scar.union_in_rank_order(
                obq[:50], ocq[:50], scar.top_from_rrf(rbq, rcq, 50),
                scar.top_from_rrf(rbt, rct, 50),
            )
            for values, sizes, candidates in (
                (request_metrics, request_sizes, strict),
                (broad_metrics, broad_sizes, broad),
                (narrow_metrics, narrow_sizes, narrow),
            ):
                values.append(scar.candidate_metrics(candidates, gold[idx]))
                sizes.append(len(candidates))
        results.append(
            {
                "conversation_text": "role-labelled content",
                "tokenizer": tokenizer_name,
                "tool_view": "plain",
                "request": [float(np.mean(request_sizes)), *np.mean(request_metrics, axis=0).tolist()],
                "broad": [float(np.mean(broad_sizes)), *np.mean(broad_metrics, axis=0).tolist()],
                "narrow": [float(np.mean(narrow_sizes)), *np.mean(narrow_metrics, axis=0).tolist()],
            }
        )
    scar.write_json(scar.OUTPUT / "conversation_text_calibration.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
