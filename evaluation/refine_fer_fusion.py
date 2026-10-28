"""Bounded public grouped-OOF comparison of richer fusion features; no extra backbone."""
from __future__ import annotations
import hashlib
import json
import sys
import warnings
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'))
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from evaluation.calibrate_fer_fusion import inputs,records
from evaluation.fer_benchmark import metrics,sweep
from fer_model import LABELS
OUT=ROOT/os.getenv('FER_FUSION_OUTPUT','outputs/fer-fusion-v3')


def features(split,layout):
    ids,x,y,old=inputs(split)
    saved=np.load(ROOT/'outputs/fer-improvement-fp32'/f'{split}_views.npz',allow_pickle=False)
    views=saved['views'].astype(np.float64)
    if layout=='log_probabilities_vit_then_legacy':
        return ids,x,y
    if layout=='log_views_vit_then_legacy':
        values=np.concatenate([views[:,0],views[:,1],old],axis=1)
        return ids,np.log(np.clip(values,1e-6,1)),y
    if layout=='mean_log_views_vit_then_legacy':
        return ids,np.concatenate([np.log(np.clip(views,1e-6,1)).mean(1),np.log(np.clip(old,1e-6,1))],axis=1),y
    raise ValueError(layout)


def model_for(kind,value):
    if kind=='linear':
        head=LogisticRegression(C=value,max_iter=1500,solver='lbfgs')
    elif kind=='svm':
        head=SVC(C=value,kernel='rbf',gamma='scale',probability=True,random_state=42)
    else:
        head=MLPClassifier(hidden_layer_sizes=(24,),activation='tanh',solver='lbfgs',alpha=value,max_iter=700,max_fun=20000,random_state=42,tol=1e-5)
    return make_pipeline(StandardScaler(),head)


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'.runtime/fer2013-benchmark-data/manifest.json').read_text())
    ids,x,y=features('publicTest','log_probabilities_vit_then_legacy')
    hashes=np.array([manifest['splits']['publicTest']['pixel_hashes'][int(Path(path).stem)] for path in ids])
    forbidden=set(manifest['splits']['privateTest']['pixel_hashes'])
    keep=np.array([h not in forbidden for h in hashes])
    y,groups=y[keep],hashes[keep]
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(x[keep],y,groups))
    base_config=json.loads((ROOT/'fer_vit_fusion.json').read_text())
    specs=[('log_probabilities_vit_then_legacy','linear',base_config['C'])]
    for layout in ('log_views_vit_then_legacy','mean_log_views_vit_then_legacy'):
        specs.extend((layout,'linear',c) for c in (.3,1.,3.))
    for layout in ('log_probabilities_vit_then_legacy','mean_log_views_vit_then_legacy'):
        specs.extend((layout,'svm',c) for c in (1.,3.,10.))
    candidates,probabilities=[],[]
    for layout,kind,value in specs:
        _,all_x,_=features('publicTest',layout)
        x=all_x[keep]
        predictions=np.zeros((len(y),7))
        valid=True
        for train,test in folds:
            assert not set(groups[train]) & set(groups[test])
            pipeline=model_for(kind,value)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                pipeline.fit(x[train],y[train])
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                valid=False
            predictions[test]=pipeline.predict_proba(x[test])
        raw=metrics(records(y,predictions),0,0)
        candidates.append({'layout':layout,'kind':kind,'value':value,'converged':valid,'public_oof':raw})
        probabilities.append(predictions)
        print('PUBLIC',layout,kind,value,raw['accuracy_overall'],raw['macro_f1'],'converged',valid,flush=True)
        (OUT/'public-progress.json').write_text(json.dumps(candidates,indent=2))
    baseline=candidates[0]['public_oof']
    eligible=[i for i,c in enumerate(candidates[1:],1) if c['converged'] and c['public_oof']['accuracy_overall']>=baseline['accuracy_overall']+.003 and c['public_oof']['macro_f1']>=baseline['macro_f1']+.003]
    selection={'baseline':baseline,'candidates':candidates,'validation':'5-fold stratified group OOF on public; private exact overlaps excluded; fold-local Pipeline scaling','gate':'OOF accuracy and macro-F1 +0.003; private raw accuracy and macro-F1 improve; no regression in selective accuracy and coverage'}
    if not eligible:
        (OUT/'rejected.json').write_text(json.dumps(selection,indent=2));print('REJECTED ON PUBLIC',flush=True);return
    index=max(eligible,key=lambda i:(candidates[i]['public_oof']['macro_f1'],candidates[i]['public_oof']['accuracy_overall']))
    chosen=candidates[index]
    selection['selected']=chosen
    _,threshold=sweep(records(y,probabilities[index]),.6877)
    selection['threshold']=threshold
    _,all_x,_=features('publicTest',chosen['layout'])
    pipeline=model_for(chosen['kind'],chosen['value'])
    pipeline.fit(all_x[keep],y)
    scaler,head=pipeline[0],pipeline[-1]
    fusion={'feature':chosen['layout'],'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'labels':list(LABELS),'training_split':'publicTest only; private overlaps excluded','public_rows':len(y),'kind':chosen['kind']}
    if chosen['kind']=='linear':
        fusion.update(coefficients=head.coef_.tolist(),intercept=head.intercept_.tolist(),C=chosen['value'],balanced=False)
    elif chosen['kind']=='mlp':
        fusion.update(layers=[{'weights':w.tolist(),'bias':b.tolist()} for w,b in zip(head.coefs_,head.intercepts_)],activation='tanh',alpha=chosen['value'])
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        exported=convert_sklearn(pipeline,initial_types=[('features',FloatTensorType([None,all_x.shape[1]]))],options={id(head):{'zipmap':False}},target_opset=17)
        onnx_path=OUT/'fusion.onnx'
        onnx_path.write_bytes(exported.SerializeToString())
        fusion={'feature':chosen['layout'],'kind':'onnx','labels':list(LABELS),'onnx_filename':'vit-fer-fusion-v2.onnx','onnx_sha256':hashlib.sha256(onnx_path.read_bytes()).hexdigest(),'public_rows':len(y),'training_split':'public only; private overlaps excluded'}
    (OUT/'fusion.json').write_text(json.dumps(fusion,indent=2))
    # Only the selected candidate is evaluated on the already-observed reference private split.
    private_ids,private_x,private_y=features('privateTest',chosen['layout'])
    p=pipeline.predict_proba(private_x)
    if chosen['kind']=='svm':
        import onnxruntime as ort
        session=ort.InferenceSession(str(OUT/'fusion.onnx'),providers=['CPUExecutionProvider'])
        exported_p=session.run(None,{'features':private_x.astype(np.float32)})[1]
        parity={'max_probability_difference':float(abs(exported_p-p).max()),'top1_disagreements':int(np.sum(exported_p.argmax(1)!=p.argmax(1)))}
        (OUT/'export-parity.json').write_text(json.dumps(parity,indent=2))
        assert parity['max_probability_difference']<.001 and parity['top1_disagreements']==0,parity
    _,base_x,_,_=inputs('privateTest')
    z=((base_x-np.array(base_config['mean']))/np.array(base_config['scale']))@np.array(base_config['coefficients']).T+np.array(base_config['intercept'])
    old=np.exp(z-z.max(1,keepdims=True));old/=old.sum(1,keepdims=True)
    old_raw=metrics(records(private_y,old),0,0);raw=metrics(records(private_y,p),0,0)
    old_app=metrics(records(private_y,old),.6,.3);app=metrics(records(private_y,p),threshold['min_confidence'],threshold['min_margin'])
    promote=raw['accuracy_overall']>old_raw['accuracy_overall'] and raw['macro_f1']>old_raw['macro_f1'] and app['accuracy_answered']>=old_app['accuracy_answered'] and app['coverage']>=old_app['coverage']
    report={'selection':selection,'private_baseline':old_raw,'private_baseline_app':old_app,'private_new':raw,'private_new_app':app,'promote':promote,'fusion_sha256':hashlib.sha256((OUT/'fusion.json').read_bytes()).hexdigest(),'limitations':['Previously observed FER2013 reference; not a fresh independent holdout.','Repeated public model selection can overfit public; independent camera data still needed.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(OUT/'selected-predictions.npz',public=probabilities[index],private=p,private_ids=np.asarray(private_ids),private_labels=private_y)
    print('FINAL',json.dumps({'promote':promote,'accuracy':raw['accuracy_overall'],'macro_f1':raw['macro_f1'],'answered_accuracy':app['accuracy_answered'],'coverage':app['coverage']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
