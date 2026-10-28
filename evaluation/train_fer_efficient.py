"""Train a frozen-feature FER head on train only; select bounded candidates on public."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from evaluation.fer_benchmark import metrics,sweep
from evaluation.calibrate_fer_fusion import inputs,records
from fer_model import LABELS

OUT=ROOT/'outputs/fer-efficient-v2'


def softmax(x):
    p=np.exp(x-x.max(1,keepdims=True));return p/p.sum(1,keepdims=True)


def current_predictions(split, oof=False):
    ids,x,y,legacy=inputs(split)
    f=json.loads((ROOT/'fer_vit_fusion.json').read_text())
    if not oof:
        p=softmax(((x-np.asarray(f['mean']))/np.asarray(f['scale']))@np.asarray(f['coefficients']).T+np.asarray(f['intercept']))
        return ids,y,p
    manifest=json.loads((ROOT/'.runtime/fer2013-benchmark-data/manifest.json').read_text())
    hashes=np.array([manifest['splits']['publicTest']['pixel_hashes'][int(Path(path).stem)] for path in ids])
    private=set(manifest['splits']['privateTest']['pixel_hashes'])
    keep=np.array([h not in private for h in hashes])
    x,y,hashes=x[keep],y[keep],hashes[keep]
    p=np.zeros((len(y),7))
    for train,valid in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(x,y,hashes):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=f['C'],max_iter=1500,solver='lbfgs'))
        model.fit(x[train],y[train]);p[valid]=model.predict_proba(x[valid])
    return np.asarray(ids)[keep].tolist(),y,p


def main():
    deadline=time.time()+2400
    while True:
        path=OUT/'extraction.json'
        extraction=json.loads(path.read_text()) if path.exists() else {}
        if 'privateTest' in extraction.get('splits',{}):
            break
        if time.time()>deadline:
            raise RuntimeError('Feature extraction did not finish')
        time.sleep(5)
    train=np.load(OUT/'train_features.npz',allow_pickle=False)
    public=np.load(OUT/'publicTest_features.npz',allow_pickle=False)
    ids,y,base=current_predictions('publicTest',oof=True)
    lookup={int(index):i for i,index in enumerate(public['ids'])}
    order=[lookup[int(Path(path).stem)] for path in ids]
    assert np.array_equal(y,public['labels'][order])
    assert not set(train['hashes']) & set(public['hashes'])
    baseline=metrics(records(y,base),0,0)
    models={}
    efficient={'pretrained':public['probabilities'][order]}
    exports={}
    # A fixed, small search. Preprocessing is fit exclusively on train inside Pipeline.
    for strength in (.001,.01,.1):
        name=f'transfer_C{strength}'
        started=time.perf_counter()
        model=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=800,solver='lbfgs',tol=1e-4))
        model.fit(train['features'],train['labels'])
        assert model[-1].n_iter_.max()<800,'Head did not converge'
        efficient[name]=model.predict_proba(public['features'][order])
        scaler,head=model[0],model[-1]
        exports[name]={'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficients':head.coef_.tolist(),'intercept':head.intercept_.tolist(),'labels':list(LABELS),'C':strength,'training_split':'train only; public/private pixel overlaps and train duplicates removed','training_rows':len(train['labels'])}
        models[name]=model
        print(name,'trained',round(time.perf_counter()-started,1),'seconds',flush=True)
    candidates=[]
    for name,p in efficient.items():
        for weight in (.2,.4,.6,1.):
            combined=(1-weight)*base+weight*p
            raw=metrics(records(y,combined),0,0)
            candidates.append({'head':name,'weight':weight,'public':raw})
            print('PUBLIC',name,weight,raw['accuracy_overall'],raw['macro_f1'],flush=True)
    eligible=[c for c in candidates if c['public']['accuracy_overall']>=baseline['accuracy_overall']+.003 and c['public']['macro_f1']>=baseline['macro_f1']+.003]
    selection={'baseline_public_oof':baseline,'candidates':candidates,'gate':'public accuracy and macro-F1 +0.003 over current OOF; private accuracy and macro-F1 improve; selective accuracy and coverage do not regress','train_data':extraction,'vit_fusion_sha256':hashlib.sha256((ROOT/'fer_vit_fusion.json').read_bytes()).hexdigest(),'validation':'train-only head; public selection using OOF current-model probabilities; no private fitting or selection'}
    if not eligible:
        (OUT/'rejected.json').write_text(json.dumps(selection,indent=2))
        print('NO PUBLIC IMPROVEMENT; keep current deployment',flush=True)
        return
    chosen=max(eligible,key=lambda c:(c['public']['macro_f1'],c['public']['accuracy_overall']))
    p=(1-chosen['weight'])*base+chosen['weight']*efficient[chosen['head']]
    _,threshold=sweep(records(y,p),.6877)
    selection['selected']=chosen
    selection['threshold']=threshold
    (OUT/'public-selection.json').write_text(json.dumps(selection,indent=2))
    private=np.load(OUT/'privateTest_features.npz',allow_pickle=False)
    private_ids,private_y,private_base=current_predictions('privateTest')
    index={int(v):i for i,v in enumerate(private['ids'])}
    order=[index[int(Path(path).stem)] for path in private_ids]
    assert np.array_equal(private_y,private['labels'][order])
    assert not set(train['hashes']) & set(private['hashes'])
    name=chosen['head']
    modern=private['probabilities'][order] if name=='pretrained' else models[name].predict_proba(private['features'][order])
    new=(1-chosen['weight'])*private_base+chosen['weight']*modern
    old_raw=metrics(records(private_y,private_base),0,0)
    old_app=metrics(records(private_y,private_base),.6,.3)
    raw=metrics(records(private_y,new),0,0)
    app=metrics(records(private_y,new),threshold['min_confidence'],threshold['min_margin'])
    promote=raw['accuracy_overall']>old_raw['accuracy_overall'] and raw['macro_f1']>old_raw['macro_f1'] and app['accuracy_answered']>=old_app['accuracy_answered'] and app['coverage']>=old_app['coverage']
    result={'selection':selection,'private_baseline':old_raw,'private_baseline_app':old_app,'private_new':raw,'private_new_app':app,'promote':promote,'limitations':['Private FER2013 has already been observed in earlier development: reference comparison, not a new unseen holdout.','Upstream training membership not independently verified.','No full-camera validation or mental-state inference.']}
    (OUT/'report.json').write_text(json.dumps(result,indent=2))
    if name!='pretrained':
        (OUT/'head.json').write_text(json.dumps(exports[name],indent=2))
    np.savez_compressed(OUT/'selected-predictions.npz',public=p,private=new,private_ids=np.asarray(private_ids),private_labels=private_y)
    print('FINAL',json.dumps({'promote':promote,'accuracy':raw['accuracy_overall'],'macro_f1':raw['macro_f1'],'answered_accuracy':app['accuracy_answered'],'coverage':app['coverage']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
