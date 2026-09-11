#!/usr/bin/env python3
"""Match the pairwise objective for request-only and feature/weighting controls."""
import json
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import scar_pair_reproduce as base
import stage2_evaluate as ev

VARIANTS = {
    'Query-Pair-strict': ('strict', 'q_fuse', None, False),
    'SCAR-Pair-no-field': ('broad', 'all', [0,1,2,3,8,9,10,11,12,13,14,15,16,17], False),
    'SCAR-Pair-uniform-pairs': ('broad', 'all', None, True),
    'SCAR-Pair-one-mask': ('broad', 'all', list(range(17)), False),
}


def fit(records, train, spec):
    pool, kind, columns, uniform = spec
    deltas, weights, labels = [], [], []
    for idx in train:
        r = records[int(idx)]
        ids = ev.augmented_ids(r, pool)
        x = ev.rows_for(r, ids, kind)
        if columns is not None: x = x[:, columns]
        positive = np.isin(ids, r['gold'])
        assert positive.any() and (~positive).any()
        d = (x[positive,None,:]-x[None,~positive,:]).reshape(-1,x.shape[1])
        deltas.extend((d,-d))
        labels.extend((np.ones(len(d)),np.zeros(len(d))))
        weights.extend((np.full(len(d),1/(2*len(d))),np.full(len(d),1/(2*len(d)))))
    x, y, w = np.vstack(deltas), np.concatenate(labels), np.concatenate(weights)
    if uniform: w[:] = len(train)/len(w)
    assert np.isclose(w.sum(),len(train))
    scale = x.std(axis=0,ddof=0); scale[scale==0]=1
    model = LogisticRegression(C=1,fit_intercept=False,class_weight=None,solver='lbfgs',
                               tol=1e-7,max_iter=2000,random_state=base.SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x/scale,y,sample_weight=w)
    return scale, model, dict(training_queries=len(train),mirrored_rows=len(x),weight_sum=float(w.sum()),
                              iterations=int(model.n_iter_[0]),scale=scale.tolist(),weights=model.coef_[0].tolist())


def main():
    records=joblib.load(base.CACHE/'stage2_records.joblib')
    predictions=[]; fits=[]
    for protocol,folds in ev.protocols(records).items():
        for fold,(train,test) in enumerate(folds):
            for name,spec in VARIANTS.items():
                if protocol!='stratified' and name!='Query-Pair-strict': continue
                print(protocol,fold,name,flush=True)
                scale,model,diag=fit(records,train,spec)
                fits.append(dict(protocol=protocol,fold=fold,method=name,**diag))
                pool,kind,columns,_=spec
                for idx in test:
                    r=records[int(idx)]; ids=r[pool]; x=ev.rows_for(r,ids,kind)
                    if columns is not None: x=x[:,columns]
                    scores=model.decision_function(x/scale)
                    assert np.isfinite(scores).all()
                    ranked=ids[np.lexsort((np.arange(len(ids)),-scores))]
                    predictions.append(dict(protocol=protocol,fold=fold,method=name,qa_id=r['qa_id'],
                                            top10_document_indices=[int(v) for v in ranked[:10]],
                                            **ev.task_metrics(ranked,r['gold'])))
    path=base.OUTPUT/'publication_pair_controls_predictions.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    aggregate=[]
    for protocol,name in sorted({(r['protocol'],r['method']) for r in predictions}):
        rows=[r for r in predictions if r['protocol']==protocol and r['method']==name]
        aggregate.append(dict(protocol=protocol,method=name,tasks=len(rows),
                              **{m:float(np.mean([r[m] for r in rows])) for m in ('hit','recall','complete','mrr','ndcg')}))
    base.write_json(base.OUTPUT/'publication_pair_controls_report.json',{
        'status':'Post hoc controlled ablations; main configuration is unchanged',
        'uniform_weight_rule':'All mirrored pairs share weight Q / total_rows; total mass and C match SCAR-Pair',
        'one_mask_rule':'Drop duplicate schema availability coordinate and refit',
        'no_field_rule':'Remove four pooled field coordinates; retain tool/schema CE views and candidates',
        'request_pair_rule':'Strict request candidate pools and seven request features with the same pairwise objective',
        'fits':fits,'aggregates':aggregate})
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__': main()
