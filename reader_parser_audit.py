#!/usr/bin/env python3
"""Audit duplicate JSON keys without changing the frozen primary parser."""
import json
from collections import Counter

import scar_pair_reproduce as base
import reader_reproduce as reader


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'Non-JSON number: {value}')


def main():
    corpus = base.load_corpus()
    tasks = {t['qa_id']: t for t in corpus.tasks}
    valid, _ = reader.gold_consistency(corpus)
    rows = base.jsonl(base.OUTPUT / 'reader_predictions.jsonl')
    alternative = []
    issues = []
    for row in rows:
        updated = dict(row)
        try:
            json.loads(row['raw_output'].strip(), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
        except ValueError as exc:
            if row['json_valid']:
                issues.append(dict(condition=row['condition'], qa_id=row['qa_id'], error=str(exc)))
                updated.update(reader.score_output('', tasks[row['qa_id']]))
        alternative.append(updated)
    primary = reader.aggregate(rows, valid)
    strict = reader.aggregate(alternative, valid)
    deltas = []
    for a, b in zip(primary, strict):
        deltas.append(dict(condition=a['condition'], **{
            key: b[key] - a[key] for key in ('json_validity', 'schema_validity',
            'memory_argument_em', 'memory_call_em', 'full_call_em')
        }))
    report = dict(
        prediction_file_sha256=base.sha256(base.OUTPUT / 'reader_predictions.jsonl'),
        audited_rows=len(rows),
        primary_parser='Python json.loads: complete document, duplicate object keys take the final value',
        sensitivity_parser='Reject duplicate object keys and non-JSON numeric constants',
        additional_rejections=len(issues), issues=issues,
        rejection_counts=dict(Counter(r['condition'] for r in issues)),
        strict_results_consistent_tasks=strict,
        strict_minus_primary=deltas,
    )
    base.write_json(base.OUTPUT / 'reader_parser_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
