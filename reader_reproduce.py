#!/usr/bin/env python3
"""Frozen Qwen reader reconstruction with resumable, auditable traces.

The released manuscript does not contain its original prompt builder.  This
script therefore records an independent fixed prompt convention and never
repairs model output before computing strict validity and exact-match metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import scar_pair_reproduce as base


QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
QWEN_CACHE = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
    / QWEN_REVISION
)
CONDITIONS = {
    "No-memory": None,
    "Query-Fuse-strict": "Query-Fuse-strict",
    "CE-Calibrated-Broad": "CE-Calibrated-Broad",
    "SCAR-Fuse": "SCAR-Fuse",
    "SCAR-Pair": "SCAR-Pair",
    "Oracle-Gold": "Oracle-Gold",
    "Hybrid-Q": "Hybrid-Q",
    "CE-Calibrated": "CE-Calibrated",
}
MANUSCRIPT_RESULTS = {
    "No-memory": (0.995, 0.979, 0.092, 0.055, 0.036),
    "Query-Fuse-strict": (0.987, 0.777, 0.168, 0.143, 0.101),
    "CE-Calibrated-Broad": (0.977, 0.753, 0.149, 0.130, 0.094),
    "SCAR-Fuse": (0.995, 0.797, 0.136, 0.106, 0.081),
    "SCAR-Pair": (0.974, 0.756, 0.122, 0.096, 0.070),
    "Oracle-Gold": (0.985, 0.784, 0.154, 0.130, 0.086),
}
SYSTEM = (
    "You generate one tool call. Use the current request, selected tool schema, "
    "and any retrieved memory conversations. Return exactly one JSON object with "
    'keys "name" and "arguments". Use the exact selected tool name. Include only '
    "parameters defined by the schema. Do not add Markdown or prose. Infer missing "
    "values only when supported by the request or memory; use a schema default when "
    "one is explicitly specified."
)
OMISSION = "\n[... omitted ...]\n"
MAX_MEMORY_TOKENS = 1024
MAX_PROMPT_TOKENS = 6144
MAX_NEW_TOKENS = 512


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prediction_rankings() -> dict[tuple[str, str], list[int]]:
    path = base.OUTPUT / "retrieval_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        (row["qa_id"], row["method"]): [int(x) for x in row["top10_document_indices"][:5]]
        for row in rows
        if row["protocol"] == "stratified" and row["method"] in set(CONDITIONS.values())
    }


def condition_ids(
    task: dict[str, Any], condition: str, rankings: dict[tuple[str, str], list[int]],
    corpus: base.Corpus,
) -> list[int]:
    if condition == "No-memory":
        return []
    if condition == "Oracle-Gold":
        gold = [corpus.id_to_index[item] for item in task["source_conversation_ids"]]
        distractors = rankings[(task["qa_id"], "SCAR-Fuse")]
        out: list[int] = []
        for idx in gold + distractors:
            if idx not in out:
                out.append(idx)
            if len(out) == 5:
                break
        return out
    method = CONDITIONS[condition]
    assert method is not None
    return rankings[(task["qa_id"], method)]


def truncate_tokens(text: str, tokenizer: Any, budget: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    marker = tokenizer.encode(OMISSION, add_special_tokens=False)
    keep = max(0, budget - len(marker))
    left = (keep + 1) // 2
    right = keep // 2
    clipped = ids[:left] + marker + (ids[-right:] if right else [])
    return tokenizer.decode(clipped, skip_special_tokens=True)


def messages_for(
    task: dict[str, Any], doc_ids: list[int], corpus: base.Corpus,
    tokenizer: Any, memory_budget: int = MAX_MEMORY_TOKENS,
) -> list[dict[str, str]]:
    memories = []
    for rank, idx in enumerate(doc_ids, 1):
        doc = corpus.documents[idx]
        body = truncate_tokens(doc["text"], tokenizer, memory_budget)
        memories.append(f"[Memory {rank} | id={doc['id']}]\n{body}")
    memory_block = "\n\n".join(memories) if memories else "(none)"
    schema = json.dumps(task["target_tool_schema"], ensure_ascii=False, indent=2)
    user = (
        f"Current request:\n{task['query']}\n\n"
        f"Selected tool schema:\n{schema}\n\n"
        f"Retrieved memory conversations:\n{memory_block}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_prompt(task: dict[str, Any], doc_ids: list[int], corpus: base.Corpus, tokenizer: Any):
    budget = MAX_MEMORY_TOKENS
    while True:
        messages = messages_for(task, doc_ids, corpus, tokenizer, budget)
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        if len(ids) <= MAX_PROMPT_TOKENS:
            break
        if not doc_ids or budget <= 32:
            raise ValueError(f"Base prompt exceeds {MAX_PROMPT_TOKENS} tokens for {task['qa_id']}")
        budget -= max(1, min(64, (len(ids) - MAX_PROMPT_TOKENS + len(doc_ids) - 1) // len(doc_ids)))
    return messages, ids, budget


def parse_exact(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text.strip()), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def validate_value(value: Any, schema: dict[str, Any], path: str) -> tuple[bool, str]:
    kind = {"int": "integer", "float": "number", "dict": "object"}.get(
        schema.get("type"), schema.get("type")
    )
    if value is None:
        return False, f"{path}: null is not allowed"
    if kind == "string" and not isinstance(value, str):
        return False, f"{path}: expected string"
    if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return False, f"{path}: expected integer"
    if kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return False, f"{path}: expected number"
    if kind == "boolean" and not isinstance(value, bool):
        return False, f"{path}: expected boolean"
    if kind == "array":
        if not isinstance(value, list):
            return False, f"{path}: expected array"
        item_schema = schema.get("items", {}) or {}
        for idx, item in enumerate(value):
            valid, error = validate_value(item, item_schema, f"{path}[{idx}]")
            if not valid:
                return valid, error
    if kind == "object":
        if not isinstance(value, dict):
            return False, f"{path}: expected object"
        props = schema.get("properties", {}) or {}
        for key in schema.get("required", []) or []:
            if key not in value:
                return False, f"{path}: missing required field {key}"
        # An empty properties map denotes an unconstrained payload in this corpus.
        if props:
            for key, item in value.items():
                if key not in props:
                    return False, f"{path}: unknown field {key}"
                valid, error = validate_value(item, props[key], f"{path}.{key}")
                if not valid:
                    return valid, error
    if "enum" in schema and value not in schema["enum"]:
        return False, f"{path}: outside enum"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False, f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return False, f"{path}: above maximum"
    if isinstance(value, str) and "pattern" in schema:
        try:
            if re.search(schema["pattern"], value) is None:
                return False, f"{path}: pattern mismatch"
        except re.error as exc:
            return False, f"{path}: invalid released pattern ({exc})"
    return True, ""


def validate_arguments(arguments: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(arguments, dict):
        return False, "arguments: expected object"
    parameters = schema.get("parameters", {}) or {}
    props = parameters.get("properties", {}) or {}
    for key in parameters.get("required", []) or []:
        if key not in arguments:
            return False, f"arguments: missing required parameter {key}"
    for key, value in arguments.items():
        if key not in props:
            return False, f"arguments: unknown parameter {key}"
        valid, error = validate_value(value, props[key], key)
        if not valid:
            return valid, error
    return True, ""


def validate_call(parsed: Any, task: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(parsed, dict):
        return False, "top level: expected object"
    if set(parsed) != {"name", "arguments"}:
        return False, "top level: keys must be exactly name and arguments"
    if parsed["name"] != task["target_tool_schema"]["name"]:
        return False, "name: selected tool mismatch"
    return validate_arguments(parsed["arguments"], task["target_tool_schema"])


def equal_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isfinite(left) and np.isfinite(right) and float(left) == float(right))
    if isinstance(left, str) and isinstance(right, str):
        return left.strip() == right.strip()
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equal_value(left[k], right[k]) for k in left)
    return type(left) is type(right) and left == right


def score_output(raw: str, task: dict[str, Any]) -> dict[str, Any]:
    parsed, parse_error = parse_exact(raw)
    json_valid = parsed is not None
    schema_valid, schema_error = validate_call(parsed, task) if json_valid else (False, parse_error or "parse")
    args = parsed.get("arguments", {}) if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), dict) else {}
    name_ok = isinstance(parsed, dict) and parsed.get("name") == task["target_tool_schema"]["name"]
    gold = task["tool_call"].get("arguments", {}) or {}
    grounding = task["tool_call"].get("grounding_info", {}) or {}
    memory_names = [k for k, info in grounding.items() if info.get("type") in {"explicit", "inferred"}]
    per_arg = {key: bool(key in args and key in gold and equal_value(args[key], gold[key])) for key in memory_names}
    arg_em = float(np.mean(list(per_arg.values()))) if per_arg else 1.0
    memory_call = bool(name_ok and all(per_arg.values()))
    full_call = bool(
        name_ok and set(args) == set(gold)
        and all(equal_value(args[key], gold[key]) for key in gold)
    )
    return {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "parse_error": parse_error,
        "schema_error": schema_error,
        "memory_argument_em": arg_em,
        "memory_call_em": memory_call,
        "full_call_em": full_call,
        "memory_argument_matches": per_arg,
    }


def gold_consistency(corpus: base.Corpus) -> tuple[set[str], list[dict[str, Any]]]:
    valid_ids: set[str] = set()
    failures = []
    for task in corpus.tasks:
        valid, error = validate_arguments(
            task["tool_call"].get("arguments", {}), task["target_tool_schema"]
        )
        if valid:
            valid_ids.add(task["qa_id"])
        else:
            failures.append({"qa_id": task["qa_id"], "error": error})
    return valid_ids, failures


def load_existing(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_hash: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            by_key[(row["condition"], row["qa_id"])] = row
            by_hash[row["prompt_sha256"]] = row
    return by_key, by_hash


def load_tokenizer():
    source = str(QWEN_CACHE) if QWEN_CACHE.exists() else "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=QWEN_CACHE.exists(),
        revision=None if QWEN_CACHE.exists() else QWEN_REVISION)
    return source, tokenizer


def load_model():
    source, tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=None if QWEN_CACHE.exists() else QWEN_REVISION,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=QWEN_CACHE.exists(),
    ).to("mps")
    model.eval()
    return tokenizer, model


def aggregate(rows: list[dict[str, Any]], valid_gold_ids: set[str] | None) -> list[dict[str, Any]]:
    out = []
    for condition in CONDITIONS:
        subset = [
            r for r in rows if r["condition"] == condition
            and (valid_gold_ids is None or r["qa_id"] in valid_gold_ids)
        ]
        if not subset:
            continue
        out.append({
            "condition": condition,
            "tasks": len(subset),
            "json_validity": float(np.mean([r["json_valid"] for r in subset])),
            "schema_validity": float(np.mean([r["schema_valid"] for r in subset])),
            "memory_argument_em": float(np.mean([r["memory_argument_em"] for r in subset])),
            "memory_call_em": float(np.mean([r["memory_call_em"] for r in subset])),
            "full_call_em": float(np.mean([r["full_call_em"] for r in subset])),
            "max_output_hits": int(sum(r["output_tokens"] >= MAX_NEW_TOKENS for r in subset)),
        })
    return out


def audit_prompts(conditions: list[str]) -> None:
    corpus = base.load_corpus()
    rankings = prediction_rankings()
    _, tokenizer = load_tokenizer()
    lengths = []
    budgets = []
    hashes = []
    for condition in conditions:
        for task in corpus.tasks:
            doc_ids = condition_ids(task, condition, rankings, corpus)
            _, ids, budget = build_prompt(task, doc_ids, corpus, tokenizer)
            lengths.append(len(ids)); budgets.append(budget)
            hashes.append(hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest())
    print(json.dumps({
        "prompts": len(lengths), "distinct_prompts": len(set(hashes)),
        "input_tokens": {"min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths)},
        "memory_budget": {"min": min(budgets), "max": max(budgets)},
    }, indent=2))


def run(conditions: list[str], limit: int | None) -> None:
    corpus = base.load_corpus()
    rankings = prediction_rankings()
    valid_gold_ids, gold_failures = gold_consistency(corpus)
    if len(valid_gold_ids) != 385:
        raise ValueError(f"Expected 385 schema-consistent gold calls, got {len(valid_gold_ids)}")
    tokenizer, model = load_model()
    path = base.OUTPUT / "reader_predictions.jsonl"
    existing, by_hash = load_existing(path)
    jobs = [(condition, task) for condition in conditions for task in corpus.tasks]
    if limit is not None:
        jobs = jobs[:limit]
    reused = Counter()
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as handle:
        for condition, task in tqdm(jobs, desc="reader"):
            key = (condition, task["qa_id"])
            doc_ids = condition_ids(task, condition, rankings, corpus)
            messages, prompt_ids, memory_budget = build_prompt(task, doc_ids, corpus, tokenizer)
            prompt_hash = hashlib.sha256(np.asarray(prompt_ids, dtype=np.int32).tobytes()).hexdigest()
            if key in existing:
                if existing[key]["prompt_sha256"] != prompt_hash:
                    raise ValueError(f"Changed prompt for existing output {key}")
                reused["existing_key"] += 1
                continue
            generation_seconds = 0.0
            if prompt_hash in by_hash:
                raw = by_hash[prompt_hash]["raw_output"]
                output_tokens = by_hash[prompt_hash]["output_tokens"]
                reused["identical_prompt"] += 1
            else:
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="mps")
                attention_mask = torch.ones_like(input_ids)
                begin = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        do_sample=False, repetition_penalty=1.0,
                        max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generation_seconds = time.perf_counter() - begin
                new_ids = generated[0, input_ids.shape[1]:].detach().cpu().tolist()
                raw = tokenizer.decode(new_ids, skip_special_tokens=True)
                output_tokens = len(new_ids)
            row = {
                "condition": condition, "qa_id": task["qa_id"],
                "document_indices": doc_ids,
                "document_ids": [corpus.documents[idx]["id"] for idx in doc_ids],
                "prompt_sha256": prompt_hash, "input_tokens": len(prompt_ids),
                "memory_token_budget": memory_budget, "output_tokens": output_tokens,
                "generation_seconds": generation_seconds, "raw_output": raw,
                **score_output(raw, task),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            existing[key] = row; by_hash[prompt_hash] = row
    disk_rows = base.jsonl(path)
    if len(disk_rows) == len(CONDITIONS) * len(corpus.tasks):
        summarize()
    else:
        print(json.dumps({"status": "partial", "completed_by_condition": dict(Counter(
            row['condition'] for row in disk_rows
        ))}, indent=2))


def summarize() -> None:
    """Read the complete disk trace after workers finish and verify every prompt."""
    corpus = base.load_corpus()
    valid_gold_ids, gold_failures = gold_consistency(corpus)
    rankings = prediction_rankings()
    _, tokenizer = load_tokenizer()
    path = base.OUTPUT / "reader_predictions.jsonl"
    rows = base.jsonl(path)
    expected = {(condition, task["qa_id"]) for condition in CONDITIONS for task in corpus.tasks}
    keys = [(row["condition"], row["qa_id"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Reader trace must contain exactly the complete condition/task product")
    tasks = {task["qa_id"]: task for task in corpus.tasks}
    hash_outputs = defaultdict(set)
    for row in tqdm(rows, desc="Verify saved reader prompts and metrics"):
        task = tasks[row["qa_id"]]
        ids = condition_ids(task, row["condition"], rankings, corpus)
        _, tokens, budget = build_prompt(task, ids, corpus, tokenizer)
        digest = hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()
        if (digest != row["prompt_sha256"] or ids != row["document_indices"]
                or len(tokens) != row["input_tokens"] or budget != row["memory_token_budget"]
                or row["document_ids"] != [corpus.documents[i]["id"] for i in ids]):
            raise ValueError(f"Prompt/context mismatch: {row['condition']}/{row['qa_id']}")
        for metric, value in score_output(row["raw_output"], task).items():
            if row[metric] != value:
                raise ValueError(f"Stored metric mismatch: {row['qa_id']}/{metric}")
        hash_outputs[digest].add(row["raw_output"])
    completed = Counter(row["condition"] for row in rows)
    completed_rows = [row for row in rows if row["condition"] in CONDITIONS]
    consistent_results = aggregate(rows, valid_gold_ids)
    failures = {}
    for condition in CONDITIONS:
        subset = [row for row in completed_rows if row["condition"] == condition
                  and row["qa_id"] in valid_gold_ids and not row["schema_valid"]]
        failures[condition] = dict(Counter(
            (row.get("schema_error") or "unknown").split(":", 1)[0] for row in subset
        ).most_common())
    report = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct", "revision": QWEN_REVISION,
        "dtype": "float16", "device": "mps", "batch_size": 1,
        "decoding": {"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": MAX_NEW_TOKENS},
        "prompt_convention": {
            "instruction": SYSTEM, "per_memory_tokens": MAX_MEMORY_TOKENS,
            "total_prompt_tokens": MAX_PROMPT_TOKENS,
            "long_memory": "equal token prefix/suffix around an omission marker",
            "total_overflow": "reduce every memory allowance equally until within budget",
            "document_header": "rank and released conversation ID",
        },
        "gold_schema_consistent": len(valid_gold_ids),
        "gold_schema_inconsistent": gold_failures,
        "completed_by_condition": dict(completed),
        "verified_prompt_and_metric_rows": len(rows),
        "prediction_file_sha256": base.sha256(path),
        "identical_prompt_groups_with_different_outputs": sum(len(values) > 1 for values in hash_outputs.values()),
        "compute_note": "Two concurrent MPS processes, batch one each; summed elapsed generation calls are not isolated kernel time and can overlap.",
        "prompt_audit_completed": {
            "condition_predictions": len(completed_rows),
            "distinct_prompt_hashes": len({row["prompt_sha256"] for row in completed_rows}),
            "input_tokens_min": min((row["input_tokens"] for row in completed_rows), default=None),
            "input_tokens_median": float(np.median([row["input_tokens"] for row in completed_rows])) if completed_rows else None,
            "input_tokens_max": max((row["input_tokens"] for row in completed_rows), default=None),
            "generation_kernel_seconds_recorded": float(sum(row["generation_seconds"] for row in completed_rows)),
        },
        "results_all_tasks": aggregate(rows, None),
        "results_consistent_tasks": consistent_results,
        "matched_manuscript_comparison": [
            {
                "condition": row["condition"],
                "reproduced": {key: row[key] for key in (
                    "json_validity", "schema_validity", "memory_argument_em",
                    "memory_call_em", "full_call_em",
                )},
                "manuscript": dict(zip((
                    "json_validity", "schema_validity", "memory_argument_em",
                    "memory_call_em", "full_call_em",
                ), MANUSCRIPT_RESULTS[row["condition"]])),
            }
            for row in consistent_results if row["condition"] in MANUSCRIPT_RESULTS
        ],
        "schema_failure_prefix_counts_consistent_tasks": failures,
    }
    base.write_json(base.OUTPUT / "reader_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "run", "summarize"))
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "audit":
        audit_prompts(args.conditions)
    elif args.command == "summarize":
        summarize()
    else:
        run(args.conditions, args.limit)


if __name__ == "__main__":
    main()
