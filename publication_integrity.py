#!/usr/bin/env python3
"""Verify publication predictions, grouping, metric recomputation, and gold isolation."""
import copy
import json
from collections import Counter
import joblib
import numpy as np
import scar_pair_reproduce as base
import stage2_evaluate as ev
import reader_reproduce as reader
from reader_parser_audit import unique_object,reject_constant
from capacity_learning_reproduce import reliability
from validate_outputs import unique,finite


def main():
    records=joblib.load(base.CACHE/'stage2_records.joblib')
    by_id={r['qa_id']:r for r in records}
    corpus=base.load_corpus()
    checks=[]
    for protocol,folds in ev.protocols(records).items():
        tests=[]
        for train,test in folds:
            assert not set(train)&set(test)
            tests.extend(test)
            if protocol=='unseen_tool':
                assert not ({records[i]['task']['target_tool_schema']['name'] for i in train}
                            &{records[i]['task']['target_tool_schema']['name'] for i in test})
            if protocol=='shared_source':
                assert not (set(np.concatenate([records[i]['gold'] for i in train]))
                            &set(np.concatenate([records[i]['gold'] for i in test])))
        assert len(tests)==len(set(tests))==(383 if protocol=='leave_source_out' else 391)
    checks.append({'check':'Fold uniqueness, task separation, tool/source-group isolation','status':'passed'})
    paths={'retrieval_predictions.jsonl':26452,'publication_pair_controls_predictions.jsonl':2729,
           'publication_reader_predictions.jsonl':3128,'publication_coverage_predictions.jsonl':3010}
    for name,n in paths.items():
        rows=base.jsonl(base.OUTPUT/name)
        assert len(rows)==n,(name,len(rows),n)
        keys=('condition','qa_id') if 'reader' in name else ('method','qa_id') if 'coverage' in name else ('protocol','method','qa_id')
        unique(rows,keys,name);finite(rows,name)
        if 'reader' not in name and 'coverage' not in name:
            for row in rows:
                actual=ev.task_metrics(np.asarray(row['top10_document_indices']),by_id[row['qa_id']]['gold'])
                for metric,value in actual.items(): assert np.isclose(value,row[metric],rtol=0,atol=1e-12),(name,row['qa_id'],metric)
        if 'reader' in name:
            for row in rows:
                try:
                    json.loads(row['raw_output'].strip(),object_pairs_hook=unique_object,parse_constant=reject_constant)
                    score=reader.score_output(row['raw_output'],by_id[row['qa_id']]['task'])
                except ValueError:
                    score=reader.score_output('',by_id[row['qa_id']]['task'])
                for metric in ('json_valid','schema_valid','memory_argument_em','memory_call_em','full_call_em'):
                    assert score[metric]==row[metric],(row['qa_id'],metric)
        checks.append({'artifact':name,'rows':len(rows),'sha256':base.sha256(base.OUTPUT/name),'status':'passed'})
    # Poison training-only inserted rows: candidate features and reliability must not change.
    affected=0
    for record in records:
        inserted=~np.isin(record['train_ids'],record['broad'])
        if not inserted.any():continue
        affected+=1
        poisoned=copy.deepcopy(record)
        for key in ('light','ce_narrow','ce_broad','ce_query_narrow'):
            poisoned[key][inserted]=1e6
        for kind in ('light','all','ce','ce_broad','q_light','q_fuse'):
            assert np.array_equal(ev.rows_for(record,record['broad'],kind),ev.rows_for(poisoned,record['broad'],kind))
        assert reliability(record)==reliability(poisoned)
    checks.append({'check':'Gold-only feature poisoning leaves retrieved rows and reliability unchanged',
                   'tasks_with_insertions':affected,'status':'passed'})
    for path in sorted(base.OUTPUT.glob('publication_*report.json')):
        finite(json.loads(path.read_text()),path.name)
    stats=json.loads((base.OUTPUT/'publication_statistics_report.json').read_text())['results']
    assert len(stats)==84
    for row in stats:
        assert 0<=row['component_p']<=row['component_p_holm']<=1
        assert row['ci95'][0]<=row['ci95'][1]
    checks.append({'check':'Publication statistics have 84 contrasts and valid probability/interval bounds','status':'passed'})
    report={'status':'passed','checks':checks}
    base.write_json(base.OUTPUT/'publication_validation_report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
