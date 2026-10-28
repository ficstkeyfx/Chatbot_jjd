"""Learn a pixel-source-independent fusion; require both FER and RAF gains."""
from __future__ import annotations
import hashlib,io,json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from evaluation.calibrate_fer_fusion import inputs,records
from evaluation.train_fer_efficient import current_predictions
from evaluation.train_fer_multidomain import fuse,pipeline
from evaluation.fer_benchmark import metrics
from fer_model import LABELS,probability_views
DATA=ROOT/'outputs/fer-multidomain-v2'
EMBEDDING=os.getenv('FER_FUSION_EMBEDDING')=='1'
OUT=ROOT/os.getenv('FER_JOINT_OUTPUT','outputs/fer-joint-fusion-v3')


def head_predict(features,head):
    z=((features-np.asarray(head['mean']))/np.asarray(head['scale']))@np.asarray(head['coefficients']).T+np.asarray(head['intercept'])
    p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)


def data():
    ft=np.load(ROOT/'outputs/fer-efficient-v2/train_features.npz',allow_pickle=False)
    fp=np.load(ROOT/'outputs/fer-efficient-v2/publicTest_features.npz',allow_pickle=False)
    fv=np.load(ROOT/'outputs/fer-efficient-v2/privateTest_features.npz',allow_pickle=False)
    rt=np.load(DATA/'raf-train-features.npz',allow_pickle=False)
    re=np.load(DATA/'raf-test-features.npz',allow_pickle=False)
    fold=np.load(DATA/'raf-fold.npz',allow_pickle=False);train,valid=fold['train'],fold['valid']
    ft_hashes,fp_hashes,fv_hashes,rt_hashes,re_hashes=(saved['hashes'] for saved in (ft,fp,fv,rt,re))
    forbidden=set(re_hashes)|set(fp_hashes)|set(fv_hashes)
    valid_hashes=set(rt_hashes[valid]);seen=set();fi=[];ri=[]
    for i,h in enumerate(ft_hashes):
        if h not in forbidden and h not in valid_hashes:fi.append(i);seen.add(h)
    for i in train:
        h=rt_hashes[i]
        if h not in forbidden and h not in seen:ri.append(i);seen.add(h)
    cache=OUT/'expert-validation.npz'
    config=json.loads((ROOT/'fer_camera_config.json').read_text())
    config_hash=hashlib.sha256((ROOT/'fer_camera_config.json').read_bytes()).hexdigest()
    cache_valid=False
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            assert str(saved['camera_config_sha256'])==config_hash
            assert np.array_equal(saved['valid_indices'],valid)
            if 'public_probabilities' in saved:
                expert_valid=saved['probabilities'];expert_public_all=saved['public_probabilities'];cache_valid=True
    if not cache_valid:
        x=np.concatenate([ft['features'][fi],rt['features'][ri]])
        y=np.concatenate([ft['labels'][fi],rt['labels'][ri]])
        weights=np.concatenate([np.ones(len(fi)),np.full(len(ri),len(fi)/len(ri))])
        model=pipeline(.001);started=time.perf_counter()
        model.fit(x,y,logisticregression__sample_weight=weights)
        assert model[-1].n_iter_.max()<900
        expert_valid=model.predict_proba(rt['features'][valid])
        expert_public_all=model.predict_proba(fp['features'])
        np.savez_compressed(cache,probabilities=expert_valid,public_probabilities=expert_public_all,valid_indices=valid,camera_config_sha256=np.asarray(config_hash),protocol_version=np.asarray(2))
        print('Expert outside RAF validation refit',round(time.perf_counter()-started,1),'s',flush=True)
    fer_ids,fer_y,base_fer=current_predictions('publicTest',oof=True)
    all_ids,fer_raw,_,_=inputs('publicTest');raw_lookup={path:i for i,path in enumerate(all_ids)}
    feature_lookup={int(i):j for j,i in enumerate(fp['ids'])}
    feature_order=[feature_lookup[int(Path(path).stem)] for path in fer_ids]
    # Use the same expert (outside all RAF validation) for both meta-training sources.
    expert_fer=expert_public_all[feature_order]
    fer_raw=fer_raw[[raw_lookup[path] for path in fer_ids]]
    xfer=np.concatenate([fer_raw,np.log(np.clip(expert_fer,1e-6,1))],1)
    with np.load(DATA/'raf-valid-baseline.npz',allow_pickle=False) as b:
        assert np.array_equal(b['indices'],valid)
        base_config=json.loads((ROOT/'fer_vit_fusion.json').read_text())
        base_raf=fuse(b['modern'],b['legacy'],base_config)
        xraf=np.log(np.clip(np.concatenate([b['modern'],b['legacy'],expert_valid],1),1e-6,1))
    if EMBEDDING:
        xfer=np.concatenate([xfer,fp['features'][feature_order]],1)
        xraf=np.concatenate([xraf,rt['features'][valid]],1)
    raf_y=rt['labels'][valid];camera_raf=.4*base_raf+.6*expert_valid
    # Exclude any exact test overlap also from the fusion training data.
    keep_raf=np.array([h not in forbidden for h in rt_hashes[valid]])
    raf_test_hashes=set(re_hashes)
    keep_fer=np.array([h not in raf_test_hashes for h in fp_hashes[feature_order]])
    arrays={'xfer':xfer[keep_fer],'yfer':fer_y[keep_fer],'base_fer':base_fer[keep_fer],
            'xraf':xraf[keep_raf],'yraf':raf_y[keep_raf],'camera_raf':camera_raf[keep_raf],
            'expert_fer':expert_fer[keep_fer],'expert_raf':expert_valid[keep_raf],
            'fer_hashes':fp_hashes[feature_order][keep_fer],'raf_hashes':rt_hashes[valid][keep_raf]}
    arrays['camera_config_sha256']=config_hash
    return arrays


def model_for(kind,value):
    if kind.startswith('poly'):
        components=int(kind[4:])
        transform=ColumnTransformer([('scores',StandardScaler(),list(range(21))),('visual',make_pipeline(StandardScaler(),PCA(n_components=components,whiten=True,svd_solver='randomized',random_state=42)),list(range(21,1429)))])
        return make_pipeline(transform,PolynomialFeatures(degree=2,interaction_only=True,include_bias=False),StandardScaler(),LogisticRegression(C=value,max_iter=2000,solver='lbfgs'))
    if kind=='linear':model=LogisticRegression(C=value,max_iter=1800,solver='lbfgs')
    elif kind=='hist':model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=int(value),min_samples_leaf=35,l2_regularization=10,early_stopping=False,random_state=42)
    else:model=ExtraTreesClassifier(n_estimators=160,max_depth=int(value),min_samples_leaf=4,n_jobs=2,random_state=42)
    return make_pipeline(StandardScaler(),model)


def main():
    OUT.mkdir(exist_ok=True)
    d=data();nf=len(d['yfer'])
    x=np.concatenate([d['xfer'],d['xraf']]);y=np.concatenate([d['yfer'],d['yraf']]);groups=np.concatenate([d['fer_hashes'],d['raf_hashes']])
    source_weights=np.concatenate([np.ones(nf),np.full(len(d['yraf']),nf/len(d['yraf']))])
    splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(x,y,groups))
    bf=metrics(records(d['yfer'],d['base_fer']),0,0);br=metrics(records(d['yraf'],d['camera_raf']),0,0)
    print('BASELINES',bf['accuracy_overall'],bf['macro_f1'],br['accuracy_overall'],br['macro_f1'],flush=True)
    specs=[('poly4',.01),('poly4',.1),('poly8',.01),('poly8',.1)] if EMBEDDING else [('linear',.01),('linear',.1),('linear',1.),('hist',7),('hist',15),('forest',8),('forest',12)]
    candidates=[];probabilities=[]
    for kind,value in specs:
        p=np.zeros((len(y),7));start=time.perf_counter()
        for train,test in splits:
            assert not set(groups[train])&set(groups[test])
            model=model_for(kind,value)
            model.fit(x[train],y[train],**{model.steps[-1][0]+'__sample_weight':source_weights[train]})
            p[test]=model.predict_proba(x[test])
        mf=metrics(records(d['yfer'],p[:nf]),0,0);mr=metrics(records(d['yraf'],p[nf:]),0,0)
        candidates.append({'kind':kind,'value':value,'fer_oof':mf,'raf_oof':mr});probabilities.append(p)
        print('OOF',kind,value,'FER',mf['accuracy_overall'],mf['macro_f1'],'RAF',mr['accuracy_overall'],mr['macro_f1'],'seconds',round(time.perf_counter()-start,1),flush=True)
        (OUT/'progress.json').write_text(json.dumps(candidates,indent=2))
    selection={'baseline_fer_public_oof':bf,'baseline_camera_raf_validation':br,'candidates':candidates,'gate':'Both validation domains: accuracy and macro-F1 +0.003 over strongest applicable baseline; both reference test domains: +0.003','features':'21 log probabilities: raw ViT, raw CNN, train-only EfficientNet expert. No filename, dataset ID, image size or color shortcut.','validation':'5-fold stratified group OOF with fold-local Pipeline; RAF expert trained outside entire RAF validation subset; exact test overlaps excluded','camera_config_sha256':d['camera_config_sha256']}
    if EMBEDDING:
        selection['features']='21 model log probabilities + 1408 frozen visual features; fold-local PCA and polynomial interactions; no filename, dataset ID or image-size flags'
    eligible=[i for i,c in enumerate(candidates) if c['fer_oof']['accuracy_overall']>=bf['accuracy_overall']+.003 and c['fer_oof']['macro_f1']>=bf['macro_f1']+.003 and c['raf_oof']['accuracy_overall']>=br['accuracy_overall']+.003 and c['raf_oof']['macro_f1']>=br['macro_f1']+.003]
    if not eligible:
        (OUT/'rejected.json').write_text(json.dumps(selection,indent=2));print('NO VALIDATION CANDIDATE; active camera model retained',flush=True);return
    index=max(eligible,key=lambda i:(candidates[i]['fer_oof']['macro_f1']+candidates[i]['raf_oof']['macro_f1'])/2)
    chosen=candidates[index];selection['selected']=chosen
    model=model_for(chosen['kind'],chosen['value']);model.fit(x,y,**{model.steps[-1][0]+'__sample_weight':source_weights})
    exported=convert_sklearn(model,initial_types=[('features',FloatTensorType([None,x.shape[1]]))],options={id(model[-1]):{'zipmap':False}},target_opset=17)
    (OUT/'fusion.onnx').write_bytes(exported.SerializeToString())
    session=ort.InferenceSession(str(OUT/'fusion.onnx'),providers=['CPUExecutionProvider'])
    expected=model.predict_proba(x);actual=session.run(None,{'features':x.astype(np.float32)})[1]
    parity={'max_probability_difference':float(abs(actual-expected).max()),'top1_disagreements':int(np.sum(actual.argmax(1)!=expected.argmax(1)))}
    (OUT/'export-parity.json').write_text(json.dumps(parity,indent=2));assert parity['max_probability_difference']<.001 and parity['top1_disagreements']==0,parity
    p=probabilities[index]
    target_f=metrics(records(d['yfer'],d['base_fer']),.6,.3)['coverage'];target_r=metrics(records(d['yraf'],d['camera_raf']),.5,.12)['coverage']
    options=[]
    for confidence in (.3,.4,.5,.6,.7,.8):
        for margin in (0.,.05,.1,.15,.2,.3):
            mf=metrics(records(d['yfer'],p[:nf]),confidence,margin);mr=metrics(records(d['yraf'],p[nf:]),confidence,margin)
            if mf['coverage']>=target_f and mr['coverage']>=target_r:options.append(((mf['accuracy_answered']+mr['accuracy_answered'])/2,confidence,margin))
    _,confidence,margin=max(options);selection['threshold']={'min_confidence':confidence,'min_margin':margin}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    # Evaluate only the chosen model on development reference test sets.
    config=json.loads((ROOT/'fer_camera_config.json').read_text())
    ids,raw_f,fy,_=inputs('privateTest')
    with np.load(ROOT/'outputs/fer-efficient-v2/privateTest_features.npz',allow_pickle=False) as saved:
        lookup={int(v):i for i,v in enumerate(saved['ids'])};order=[lookup[int(Path(path).stem)] for path in ids];fer_embedding=saved['features'][order];ef=head_predict(fer_embedding,config['head'])
    private_x=np.concatenate([raw_f,np.log(np.clip(ef,1e-6,1))],1)
    if EMBEDDING:private_x=np.concatenate([private_x,fer_embedding],1)
    pf=model.predict_proba(private_x)
    cache=OUT/'raf-test-legacy.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:legacy=saved['probabilities']
    else:
        rows=pq.read_table(ROOT/'.runtime/rafdb-reference/test.parquet').to_pylist();legacy=[]
        for row in rows:
            with Image.open(io.BytesIO(row['image']['bytes'])) as image:legacy.append(probability_views(image,ROOT/'.models/fer2013/webapp.onnx').mean(0))
        legacy=np.asarray(legacy);np.savez_compressed(cache,probabilities=legacy)
    with np.load(DATA/'raf-test-features.npz',allow_pickle=False) as saved:raf_embedding=saved['features'];er=head_predict(raf_embedding,config['head']);ry=saved['labels']
    with np.load(ROOT/'outputs/fer-dima-v2/rafdb-previous.npz',allow_pickle=False) as saved:modern=saved['probabilities']
    xr=np.log(np.clip(np.concatenate([modern,legacy,er],1),1e-6,1))
    if EMBEDDING:xr=np.concatenate([xr,raf_embedding],1)
    pr=model.predict_proba(xr)
    baseline=json.loads((DATA/'camera-reference.json').read_text());oldf=baseline['fer_baseline'];oldr=baseline['raf_new']
    mf=metrics(records(fy,pf),0,0);mr=metrics(records(ry,pr),0,0)
    promote=mf['accuracy_overall']>=oldf['accuracy_overall']+.003 and mf['macro_f1']>=oldf['macro_f1']+.003 and mr['accuracy_overall']>=oldr['accuracy_overall']+.003 and mr['macro_f1']>=oldr['macro_f1']+.003
    report={'selection':selection,'fer_baseline':oldf,'fer_new':mf,'raf_baseline':oldr,'raf_new':mr,'fer_app':metrics(records(fy,pf),confidence,margin),'raf_app':metrics(records(ry,pr),confidence,margin),'promote':promote,'limitations':['Reference tests have already been observed in prior development, not fresh holdouts.','No guarantee for unseen camera images; independent full-frame data still required.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'selected-predictions.npz',fer=pf,raf=pr,fer_ids=np.asarray(ids))
    print('FINAL',json.dumps({'promote':promote,'fer_accuracy':mf['accuracy_overall'],'fer_f1':mf['macro_f1'],'raf_accuracy':mr['accuracy_overall'],'raf_f1':mr['macro_f1']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
