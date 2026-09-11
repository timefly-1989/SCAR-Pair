#!/usr/bin/env python3
"""Publication analyses derived from frozen predictions, with explicit corrections."""
import hashlib
import json
from collections import defaultdict

import numpy as np
import scar_pair_reproduce as base
import reader_reproduce as reader
from reader_parser_audit import unique_object, reject_constant
from coverage_reproduce import normalize, contains
from statistics_reproduce import holm
from stage2_evaluate import shared_source_groups


def paired(left, right, metrics, components, seed):
    ids = sorted(set(left) & set(right))
    delta = np.array([[float(left[q][m]) - float(right[q][m]) for m in metrics] for q in ids])
    members = defaultdict(list)
    for i, q in enumerate(ids):
        members[components[q]].append(i)
    groups = list(members.values())
    sums = np.array([delta[g].sum(axis=0) for g in groups])
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    draw = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=10000)
    boots = (draw @ sums) / (draw @ sizes)[:, None]
    observed = delta.mean(axis=0)
    exceed = np.zeros(len(metrics), dtype=int)
    for _ in range(50):
        signs = rng.choice((-1., 1.), size=(1000, len(groups)))
        randomized = signs @ sums / len(ids)
        exceed += (np.abs(randomized) >= np.maximum(0, np.abs(observed) - 1e-12)).sum(axis=0)
    p = dict(zip(metrics, (exceed + 1) / 50001))
    adjusted = holm(p)
    return [dict(metric=m, tasks=len(ids), components=len(groups), difference=float(observed[i]),
                 ci95=np.quantile(boots[:, i], [.025, .975]).tolist(),
                 component_p=float(p[m]), component_p_holm=float(adjusted[m]))
            for i, m in enumerate(metrics)]


def main():
    corpus = base.load_corpus()
    tasks = {t['qa_id']: t for t in corpus.tasks}
    records = [{'qa_id': t['qa_id'], 'gold': np.array([corpus.id_to_index[x]
                for x in t['source_conversation_ids']])} for t in corpus.tasks]
    groups = shared_source_groups(records)
    components = {t['qa_id']: int(g) for t, g in zip(corpus.tasks, groups)}
    raw = base.jsonl(base.OUTPUT / 'reader_predictions.jsonl')
    valid, failures = reader.gold_consistency(corpus)
    corrected, rejected = [], []
    for row in raw:
        updated = dict(row)
        try:
            json.loads(row['raw_output'].strip(), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
            updated.update(reader.score_output(row['raw_output'], tasks[row['qa_id']]))
        except ValueError as exc:
            updated.update(reader.score_output('', tasks[row['qa_id']]))
            updated['parse_error'] = str(exc)
            updated['schema_error'] = str(exc)
            if row['json_valid']:
                rejected.append({'condition': row['condition'], 'qa_id': row['qa_id'], 'error': str(exc)})
        corrected.append(updated)
    trace = base.OUTPUT / 'publication_reader_predictions.jsonl'
    trace.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in corrected))
    base.write_json(base.OUTPUT / 'publication_reader_report.json', {
        'raw_prediction_sha256': base.sha256(base.OUTPUT / 'reader_predictions.jsonl'),
        'parser': 'Complete JSON document; reject duplicate object keys and non-JSON numeric constants; no repair',
        'scoring_change': 'Rescore frozen strings only; generation was not repeated',
        'additional_rejections': rejected, 'gold_failures': failures,
        'consistent': reader.aggregate(corrected, valid), 'all': reader.aggregate(corrected, None),
    })
    retrieval = base.jsonl(base.OUTPUT / 'retrieval_predictions.jsonl')
    controls = base.jsonl(base.OUTPUT / 'publication_pair_controls_predictions.jsonl')
    statistics = []
    family_idx = 0
    for protocol in ('stratified', 'unseen_tool', 'shared_source', 'leave_source_out'):
        lookup = defaultdict(dict)
        for r in retrieval + controls:
            if r['protocol'] == protocol:
                lookup[r['method']][r['qa_id']] = r
        pairs = [('SCAR-Pair', b) for b in ('Query-Fuse-strict', 'SCAR-Fuse', 'CE-Calibrated-Broad')]
        pairs.append(('SCAR-Pair', 'Query-Pair-strict'))
        if protocol == 'stratified':
            pairs += [('SCAR-Pair', b) for b in ('SCAR-Pair-no-field', 'SCAR-Pair-uniform-pairs', 'SCAR-Pair-one-mask')]
            pairs += [('Query-Fuse-schema-pool', 'Query-Fuse-strict'),
                      ('CE-Calibrated-Broad', 'CE-Query-Calibrated-Broad')]
        for left, right in pairs:
            assert lookup[left] and lookup[right], (left, right)
            result = paired(lookup[left], lookup[right], ('recall', 'complete', 'ndcg'),
                            components, base.SEED + 700 + family_idx)
            statistics.extend(dict(domain='retrieval', protocol=protocol, left=left, right=right, **r) for r in result)
            family_idx += 1
    lookup = defaultdict(dict)
    for r in corrected:
        if r['qa_id'] in valid:
            lookup[r['condition']][r['qa_id']] = r
    for comparator in reader.CONDITIONS:
        if comparator == 'SCAR-Pair':
            continue
        result = paired(lookup['SCAR-Pair'], lookup[comparator],
                        ('memory_argument_em', 'memory_call_em', 'full_call_em'),
                        components, base.SEED + 700 + family_idx)
        statistics.extend(dict(domain='reader', protocol='stratified', left='SCAR-Pair', right=comparator, **r) for r in result)
        family_idx += 1
    primary = {str(i): r['component_p'] for i, r in enumerate(statistics)
               if r['domain'] == 'retrieval' and r['protocol'] == 'stratified' and r['left'] == 'SCAR-Pair'
               and r['right'] in ('Query-Fuse-strict','SCAR-Fuse','CE-Calibrated-Broad','Query-Pair-strict')}
    for i, p in holm(primary).items():
        statistics[int(i)]['primary_twelve_comparison_holm'] = p
    base.write_json(base.OUTPUT / 'publication_statistics_report.json', {
        'bootstrap': '10000 component resamples, task-weighted statistic, percentile interval',
        'randomization': '50000 joint component sign flips, two-sided, plus-one correction',
        'holm': 'Three metrics per comparator; additional twelve-test primary retrieval sensitivity',
        'seed_rule': '20260730 + 700 + family index in script order',
        'scope': 'Post hoc robustness analysis; conditions on fixed out-of-fold predictions; assumes component sign symmetry',
        'results': statistics,
    })
    # Match within individual documents: a surface cannot span two memories.
    methods = ('Hybrid-Q', 'Query-Fuse-strict', 'SCAR-Light', 'CE-Schema',
               'CE-Calibrated', 'CE-Calibrated-Broad', 'SCAR-Fuse', 'SCAR-Pair')
    rankings = {(r['qa_id'], r['method']): r['top10_document_indices'][:5]
                for r in retrieval if r['protocol'] == 'stratified'}
    old = base.jsonl(base.OUTPUT / 'coverage_predictions.jsonl')
    values = {r['qa_id']: r['values'] for r in old}
    coverage = []
    for q, vals in values.items():
        gold = list(dict.fromkeys(corpus.id_to_index[x] for x in tasks[q]['source_conversation_ids']))
        for method in (*methods, 'Oracle-Gold-top5', 'Annotated-sources-all'):
            ids = gold if method == 'Annotated-sources-all' else gold[:5] if method == 'Oracle-Gold-top5' else rankings[q, method]
            documents = [normalize(corpus.documents[int(i)]['text']) for i in ids]
            hits = [any(contains(text, v['normalized']) for text in documents) for v in vals]
            coverage.append(dict(qa_id=q, method=method, value_count=len(vals), found_count=sum(hits),
                                 coverage=float(np.mean(hits)), document_indices=[int(i) for i in ids]))
    path = base.OUTPUT / 'publication_coverage_predictions.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in coverage))
    aggregates = []
    for method in (*methods, 'Oracle-Gold-top5', 'Annotated-sources-all'):
        subset = [r for r in coverage if r['method'] == method]
        n = sum(r['value_count'] for r in subset)
        found = sum(r['found_count'] for r in subset)
        aggregates.append(dict(method=method, tasks=len(subset), values=n, found=found,
                               micro=found/n, macro=float(np.mean([r['coverage'] for r in subset]))))
    base.write_json(base.OUTPUT / 'publication_coverage_report.json', {
        'normalization': 'NFKC, casefold, non-word to spaces, whitespace collapse, token boundaries',
        'correction': 'Match separately inside each document; report both first-five and all annotated sources',
        'results': aggregates,
    })
    print(json.dumps({'reader_rows': len(corrected), 'new_parser_rejections': len(rejected),
                      'statistics': len(statistics), 'coverage_rows': len(coverage),
                      'coverage': aggregates}, indent=2))


if __name__ == '__main__':
    main()
