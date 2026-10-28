"""Train a joint FER/RAF frozen-feature expert; select on FER public and RAF train validation."""
import hashlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from evaluation.train_fer_efficient import current_predictions
from evaluation.calibrate_fer_fusion import records
from evaluation.fer_benchmark import metrics
from fer_model import LABELS,probability_views
OUT=ROOT/'outputs/fer-multidomain-v2'


def fuse(modern,legacy,config):
    x=np.log(np.clip(np.concatenate([modern,legacy],1).astype(np.float64),1e-6,1))
    z=((x-np.array(config['mean']))/np.array(config['scale']))@np.array(config['coefficients']).T+np.array(config['intercept'])
    p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)


def pipeline(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=900,solver='lbfgs',tol=1e-4))


def main():
    deadline=time.time()+2400
    while not (OUT/'features-ready.json').exists():
        if time.time()>deadline:raise RuntimeError('Features not ready')
        time.sleep(5)
    fer_train=np.load(ROOT/'outputs/fer-efficient-v2/train_features.npz',allow_pickle=False)
    fer_public=np.load(ROOT/'outputs/fer-efficient-v2/publicTest_features.npz',allow_pickle=False)
    raf_train=np.load(OUT/'raf-train-features.npz',allow_pickle=False)
    raf_test=np.load(OUT/'raf-test-features.npz',allow_pickle=False)
    fold=np.load(OUT/'raf-fold.npz',allow_pickle=False)
    t,v=fold['train'],fold['valid']
    ids,fy,base_f=current_predictions('publicTest',oof=True)
    lookup={int(value):i for i,value in enumerate(fer_public['ids'])}
    order=[lookup[int(Path(path).stem)] for path in ids]
    fx=fer_public['features'][order]
    assert np.array_equal(fy,fer_public['labels'][order])
    base_config=json.loads((ROOT/'fer_vit_fusion.json').read_text())
    valid_base=np.load(OUT/'raf-valid-baseline.npz',allow_pickle=False)
    assert np.array_equal(valid_base['indices'],v)
    base_r=fuse(valid_base['modern'],valid_base['legacy'],base_config)
    ry=raf_train['labels'][v]
    forbidden=set(raf_test['hashes'])|set(fer_public['hashes'])
    private_fer=np.load(ROOT/'outputs/fer-efficient-v2/privateTest_features.npz',allow_pickle=False)
    forbidden |= set(private_fer['hashes'])
    seen=set()
    fkeep=[]
    raf_validation_hashes=set(raf_train['hashes'][v])
    for i,h in enumerate(fer_train['hashes']):
        if h not in forbidden and h not in raf_validation_hashes:fkeep.append(i);seen.add(h)
    tkeep=[]
    for i in t:
        h=raf_train['hashes'][i]
        if h not in forbidden and h not in seen:tkeep.append(i);seen.add(h)
    train_x=np.concatenate([fer_train['features'][fkeep],raf_train['features'][tkeep]])
    train_y=np.concatenate([fer_train['labels'][fkeep],raf_train['labels'][tkeep]])
    sample_weights=np.concatenate([np.ones(len(fkeep)),np.full(len(tkeep),len(fkeep)/len(tkeep))])
    baseline_f=metrics(records(fy,base_f),0,0);baseline_r=metrics(records(ry,base_r),0,0)
    candidates=[];predictions=[];models={}
    for c in (.001,.01,.1):
        model=pipeline(c);model.fit(train_x,train_y,logisticregression__sample_weight=sample_weights)
        assert model[-1].n_iter_.max()<900
        pf=model.predict_proba(fx);pr=model.predict_proba(raf_train['features'][v]);models[c]=model
        for weight in (.1,.2,.3,.4,.6):
            mixed_f=(1-weight)*base_f+weight*pf;mixed_r=(1-weight)*base_r+weight*pr
            mf=metrics(records(fy,mixed_f),0,0);mr=metrics(records(ry,mixed_r),0,0)
            candidates.append({'C':c,'weight':weight,'fer_public':mf,'raf_validation':mr});predictions.append((mixed_f,mixed_r))
            print('VALID',c,weight,'FER',mf['accuracy_overall'],mf['macro_f1'],'RAF',mr['accuracy_overall'],mr['macro_f1'],flush=True)
    selection={'baseline_fer_public':baseline_f,'baseline_raf_validation':baseline_r,'candidates':candidates,'training':{'fer_rows':len(fkeep),'raf_rows':len(tkeep),'raf_validation_rows':len(v),'datasets_equal_total_head_weight':True},'gate':'FER public accuracy and F1 do not regress; RAF train-validation accuracy/F1 +0.01; final reference FER metrics retain and RAF metrics +0.01','vit_fusion_sha256':hashlib.sha256((ROOT/'fer_vit_fusion.json').read_bytes()).hexdigest()}
    eligible=[i for i,c in enumerate(candidates) if c['fer_public']['accuracy_overall']>=baseline_f['accuracy_overall'] and c['fer_public']['macro_f1']>=baseline_f['macro_f1'] and c['raf_validation']['accuracy_overall']>=baseline_r['accuracy_overall']+.01 and c['raf_validation']['macro_f1']>=baseline_r['macro_f1']+.01]
    if not eligible:
        (OUT/'rejected.json').write_text(json.dumps(selection,indent=2));print('REJECTED ON VALIDATION',flush=True);return
    index=max(eligible,key=lambda i:(candidates[i]['raf_validation']['macro_f1'],candidates[i]['fer_public']['macro_f1']))
    chosen=candidates[index];selection['selected']=chosen
    pf,pr=predictions[index]
    old_fapp=metrics(records(fy,base_f),.6,.3);old_rapp=metrics(records(ry,base_r),.6,.3)
    thresholds=[]
    for confidence in (.3,.4,.5,.6,.7):
        for margin in (0.,.05,.1,.15,.2,.25,.3):
            mf=metrics(records(fy,pf),confidence,margin);mr=metrics(records(ry,pr),confidence,margin)
            if mf['coverage']>=old_fapp['coverage'] and mr['coverage']>=old_rapp['coverage']:
                thresholds.append((.5*mf['accuracy_answered']+.5*mr['accuracy_answered'],confidence,margin,mf,mr))
    _,confidence,margin,_,_=max(thresholds,key=lambda x:x[0]);selection['threshold']={'min_confidence':confidence,'min_margin':margin}
    (OUT/'public-validation-selection.json').write_text(json.dumps(selection,indent=2))
    # Refit only the chosen expert on all eligible train rows; neither test split is fit.
    seen=set(fer_train['hashes'][fkeep]);all_raf=[]
    for i,h in enumerate(raf_train['hashes']):
        if h not in forbidden and h not in seen:all_raf.append(i);seen.add(h)
    all_x=np.concatenate([fer_train['features'][fkeep],raf_train['features'][all_raf]])
    all_y=np.concatenate([fer_train['labels'][fkeep],raf_train['labels'][all_raf]])
    weights=np.concatenate([np.ones(len(fkeep)),np.full(len(all_raf),len(fkeep)/len(all_raf))])
    model=pipeline(chosen['C']);model.fit(all_x,all_y,logisticregression__sample_weight=weights)
    scaler,head=model[0],model[-1]
    config={'format':1,'labels':list(LABELS),'head':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficients':head.coef_.tolist(),'intercept':head.intercept_.tolist(),'labels':list(LABELS)},'efficient_weight':chosen['weight'],'min_confidence':confidence,'min_margin':margin,'vit_fusion_sha256':selection['vit_fusion_sha256'],'training':{'fer_rows':len(fkeep),'raf_rows':len(all_raf),'source':'frozen EmotiEffLib B2; train only, evaluation pixel overlaps excluded'}}
    (OUT/'deployment-config.json').write_text(json.dumps(config,indent=2))
    private_ids,private_y,old_fer=current_predictions('privateTest')
    lookup={int(value):i for i,value in enumerate(private_fer['ids'])};order=[lookup[int(Path(path).stem)] for path in private_ids]
    expert=model.predict_proba(private_fer['features'][order]);new_fer=(1-chosen['weight'])*old_fer+chosen['weight']*expert
    rows=pq.read_table(ROOT/'.runtime/rafdb-reference/test.parquet').to_pylist()
    legacy=[]
    for row in rows:
        with Image.open(io.BytesIO(row['image']['bytes'])) as image:legacy.append(probability_views(image,ROOT/'.models/fer2013/webapp.onnx').mean(0))
    with np.load(ROOT/'outputs/fer-dima-v2/rafdb-previous.npz',allow_pickle=False) as saved:old_raf=fuse(saved['probabilities'],np.asarray(legacy),base_config)
    new_raf=(1-chosen['weight'])*old_raf+chosen['weight']*model.predict_proba(raf_test['features'])
    old_f=metrics(records(private_y,old_fer),0,0);new_f=metrics(records(private_y,new_fer),0,0)
    old_r=metrics(records(raf_test['labels'],old_raf),0,0);new_r=metrics(records(raf_test['labels'],new_raf),0,0)
    promote=new_f['accuracy_overall']>=old_f['accuracy_overall'] and new_f['macro_f1']>=old_f['macro_f1'] and new_r['accuracy_overall']>=old_r['accuracy_overall']+.01 and new_r['macro_f1']>=old_r['macro_f1']+.01
    report={'selection':selection,'fer_baseline':old_f,'fer_new':new_f,'raf_baseline':old_r,'raf_new':new_r,'fer_app':metrics(records(private_y,new_fer),confidence,margin),'raf_app':metrics(records(raf_test['labels'],new_raf),confidence,margin),'fer_baseline_app':metrics(records(private_y,old_fer),.6,.3),'raf_baseline_app':metrics(records(raf_test['labels'],old_raf),.6,.3),'promote':promote,'limitations':['Both reference test sets have previously been observed for other development candidates; not fresh holdouts.','Upstream membership unknown; no full-frame camera or mental-state validation.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(OUT/'selected-predictions.npz',fer=new_fer,raf=new_raf,fer_ids=np.asarray(private_ids))
    print('FINAL',json.dumps({'promote':promote,'fer_old':old_f['accuracy_overall'],'fer_new':new_f['accuracy_overall'],'raf_old':old_r['accuracy_overall'],'raf_new':new_r['accuracy_overall'],'fer_f1':new_f['macro_f1'],'raf_f1':new_r['macro_f1']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
