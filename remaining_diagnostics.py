#!/usr/bin/env python3
"""Complete request-only CE and LoCoMo category/cutoff summaries from caches."""
import json
from collections import defaultdict

import joblib
import numpy as np

import scar_pair_reproduce as base
import stage2_evaluate as ev


def main():
    records = joblib.load(base.CACHE / 'stage2_records.joblib')
    predictions = []
    for protocol, folds in ev.protocols(records).items():
        for fold, (train, test) in enumerate(folds):
            scaler, model = ev.fit_pointwise(records, train, 'strict_narrow', 'q_ce')
            for i in test:
                record = records[int(i)]
                ranking = ev.rank_pointwise(record, 'strict_narrow', 'q_ce', scaler, model)
                predictions.append(dict(protocol=protocol, fold=fold, qa_id=record['qa_id'],
                    method='Query-CE-Calibrated', top10_document_indices=ranking[:10].tolist(),
                    **ev.task_metrics(ranking, record['gold'])))
    path = base.OUTPUT / 'query_ce_predictions.jsonl'
    with path.open('w') as stream:
        for row in predictions:
            stream.write(json.dumps(row) + '\n')
    groups = defaultdict(list)
    for row in predictions:
        groups[row['protocol']].append(row)
    base.write_json(base.OUTPUT / 'query_ce_report.json', {
        'pool': 'strict request-only depth-50 union', 'features': 'three request CE coordinates',
        'results': [dict(protocol=p, tasks=len(rows), **{
            m: float(np.mean([r[m] for r in rows])) for m in ('recall','complete','mrr','ndcg')
        }) for p, rows in groups.items()]})

    summaries = []
    for model in ('bge-small', 'mgte', 'bge-m3'):
        path = base.OUTPUT / f'locomo_predictions_{model}_timestamped_unicode.jsonl'
        rows = base.jsonl(path)
        groups = defaultdict(list)
        for row in rows:
            groups[(row['method'], 'all')].append(row)
            groups[(row['method'], str(row['category']))].append(row)
        for (method, category), group in groups.items():
            for k in (1, 5, 10, 20):
                summaries.append(dict(model=model, method=method, category=category,
                    k=k, tasks=len(group), **{m: float(np.mean([
                        r['metrics_by_k'][str(k)][m] for r in group
                    ])) for m in ('hit','recall','complete','mrr','ndcg')}))
    base.write_json(base.OUTPUT / 'locomo_categories_report.json', {
        'serialization': 'timestamped, Unicode BM25', 'results': summaries})
    print(f'Query-CE: {len(predictions)} predictions; LoCoMo: {len(summaries)} category/cutoff cells')


if __name__ == '__main__':
    main()
