#!/usr/bin/env python3
"""Complete source-macro and strict CE summaries from frozen predictions."""
import json
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent / 'output'
METRICS = ('hit', 'recall', 'complete', 'mrr', 'ndcg')


def rows(name):
    return [json.loads(s) for s in (OUT / name).read_text().splitlines()]


def main():
    predictions = rows('retrieval_predictions.jsonl') + rows('publication_pair_controls_predictions.jsonl')
    macro = []
    for method in sorted({r['method'] for r in predictions if r['protocol'] == 'leave_source_out'}):
        source_means = []
        for fold in range(3):
            subset = [r for r in predictions if r['protocol'] == 'leave_source_out' and r['method'] == method and r['fold'] == fold]
            assert subset and len({r['qa_id'] for r in subset}) == len(subset)
            source_means.append(dict(fold=fold, tasks=len(subset), **{k: mean(r[k] for r in subset) for k in METRICS}))
        macro.append(dict(method=method, sources=3, source_means=source_means,
                          **{k: mean(r[k] for r in source_means) for k in METRICS}))
    ce = rows('query_ce_predictions.jsonl')
    strict_ce = []
    for protocol in sorted({r['protocol'] for r in ce}):
        subset = [r for r in ce if r['protocol'] == protocol]
        assert len({r['qa_id'] for r in subset}) == len(subset)
        strict_ce.append(dict(protocol=protocol, method='Query-CE-strict', n=len(subset),
                              **{k: mean(r[k] for r in subset) for k in METRICS}))
    result = dict(source_macro=macro, strict_ce=strict_ce)
    (OUT / 'publication_extra_summaries.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(next(r for r in macro if r['method'] == 'Query-Pair-strict'), indent=2))


if __name__ == '__main__':
    main()
