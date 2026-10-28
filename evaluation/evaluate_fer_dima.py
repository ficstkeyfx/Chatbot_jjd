"""Evaluate an alternate pinned ViT; select solely on grouped public predictions."""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.runtime/fer-dml'))
os.environ['FER_IMPROVEMENT_OUTPUT']='outputs/fer-dima-v2'
os.environ['FER_CANDIDATE_MODEL']='.models/fer2013/vit-dima-fp32.onnx'
import httpx
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import fer_model
from evaluation.improve_fer import run,baseline
from evaluation.fer_benchmark import metrics,sweep
from evaluation.calibrate_fer_fusion import records

OUT=ROOT/'outputs/fer-dima-v2'
MODEL=ROOT/'.models/fer2013/vit-dima-fp32.onnx'
SHA='3aac38bf9d6119956282fb5dfab1919ffcd4b72422e0be6436e3ca2a084265d4'
REVISION='e039a363ab5d9e1b82eb08b3af90b4648cb8f696'
REPOSITORY='Xenova/facial_emotions_image_detection'


def main():
    OUT.mkdir(exist_ok=True)
    if not MODEL.is_file() or hashlib.sha256(MODEL.read_bytes()).hexdigest()!=SHA:
        temporary=MODEL.with_suffix('.download')
        with httpx.Client(timeout=120,follow_redirects=True) as client,client.stream('GET',f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/onnx/model.onnx') as response:
            response.raise_for_status()
            with temporary.open('wb') as file:
                for chunk in response.iter_bytes(1024*1024):file.write(chunk)
        assert hashlib.sha256(temporary.read_bytes()).hexdigest()==SHA
        temporary.replace(MODEL)
    metadata={'format':1,'architecture':'vit','input_size':224,'mean':[.5]*3,'std':[.5]*3,'labels':['sad','disgust','angry','neutral','fear','surprise','happy'],'tta':'flip','ensemble_legacy_weight':0.,'repository':REPOSITORY,'revision':REVISION,'sha256':SHA}
    Path(str(MODEL)+'.json').write_text(json.dumps(metadata,indent=2))
    options=ort.SessionOptions();options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.intra_op_num_threads=2
    for name,value in [('batch_size',2),('num_channels',3),('height',224),('width',224)]:options.add_free_dimension_override_by_name(name,value)
    session=ort.InferenceSession(str(MODEL),sess_options=options,providers=[('DmlExecutionProvider',{'device_id':0}),'CPUExecutionProvider'])
    fer_model._session=lambda path:session
    old=baseline('public')
    paths,views,_=run('publicTest',old)
    # Raw CPU/GPU check on representative public samples; no labels used to modify predictions.
    cpu_options=ort.SessionOptions();cpu_options.intra_op_num_threads=2
    cpu=ort.InferenceSession(str(MODEL),sess_options=cpu_options,providers=['CPUExecutionProvider'])
    differences,disagreements=[],0
    for i in np.linspace(0,len(paths)-1,21,dtype=int):
        fer_model._session=lambda path:cpu
        with Image.open(paths[i][1]) as image:actual=fer_model.probability_views(image,MODEL)
        differences.append(float(abs(actual-views[i]).max()));disagreements+=int(actual.mean(0).argmax()!=views[i].mean(0).argmax())
    fer_model._session=lambda path:session
    parity={'samples':21,'max_probability_difference':max(differences),'top1_disagreements':disagreements}
    (OUT/'provider-parity.json').write_text(json.dumps(parity,indent=2))
    assert max(differences)<.001 and disagreements==0,parity
    del cpu
    manifest=json.loads((ROOT/'.runtime/fer2013-benchmark-data/manifest.json').read_text())
    hashes=np.array([manifest['splits']['publicTest']['pixel_hashes'][int(p.stem)] for _,p in paths])
    forbidden=set(manifest['splits']['privateTest']['pixel_hashes'])
    keep=np.array([h not in forbidden for h in hashes])
    y=np.array([fer_model.LABELS.index(label) for label,_ in paths])[keep]
    modern=views.mean(1)[keep]
    legacy=np.array([old[p.parent.name+'/'+p.name] for _,p in paths])[keep]
    x=np.log(np.clip(np.concatenate([modern,legacy],axis=1),1e-6,1))
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(x,y,hashes[keep]))
    candidates=[{'kind':'raw','public':metrics(records(y,modern),0,0)}]
    predictions=[modern]
    for strength in (.03,.1,.3,1.):
        p=np.zeros((len(y),7))
        for train,test in folds:
            assert not set(hashes[keep][train])&set(hashes[keep][test])
            model=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=1500,solver='lbfgs'))
            model.fit(x[train],y[train]);p[test]=model.predict_proba(x[test])
        raw=metrics(records(y,p),0,0)
        candidates.append({'kind':'fusion','C':strength,'public':raw});predictions.append(p)
        print('PUBLIC',strength,raw['accuracy_overall'],raw['macro_f1'],flush=True)
    current=json.loads((ROOT/'outputs/fer-improvement-fp32/fusion-report.json').read_text())
    base=current['selection']['selected']['oof']
    selection={'current_public_oof':base,'candidates':candidates,'gate':'public accuracy and macro-F1 +0.003; private raw and selective accuracy/coverage all improve or retain','source':metadata,'validation':'private pixel overlaps excluded; fusion scaling in fold-local Pipeline; no private selection'}
    eligible=[i for i,c in enumerate(candidates) if c['public']['accuracy_overall']>=base['accuracy_overall']+.003 and c['public']['macro_f1']>=base['macro_f1']+.003]
    if not eligible:
        (OUT/'rejected.json').write_text(json.dumps(selection,indent=2));print('REJECTED ON PUBLIC',flush=True);return
    index=max(eligible,key=lambda i:(candidates[i]['public']['macro_f1'],candidates[i]['public']['accuracy_overall']))
    chosen=candidates[index];selection['selected']=chosen
    _,threshold=sweep(records(y,predictions[index]),.6877)
    selection['threshold']=threshold
    fusion=None
    if chosen['kind']=='fusion':
        model=make_pipeline(StandardScaler(),LogisticRegression(C=chosen['C'],max_iter=1500,solver='lbfgs'));model.fit(x,y)
        scaler,head=model[0],model[-1]
        fusion={'feature':'log_probabilities_vit_then_legacy','mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficients':head.coef_.tolist(),'intercept':head.intercept_.tolist(),'labels':list(fer_model.LABELS),'C':chosen['C'],'balanced':False,'public_rows':len(y),'training_split':'public only; private overlaps excluded'}
    (OUT/'public-selection.json').write_text(json.dumps(selection,indent=2))
    private_old=baseline('private')
    private_paths,private_views,_=run('privateTest',private_old)
    py=np.array([fer_model.LABELS.index(label) for label,_ in private_paths])
    private_p=private_views.mean(1)
    if fusion:
        old_p=np.array([private_old[p.parent.name+'/'+p.name] for _,p in private_paths])
        private_p=model.predict_proba(np.log(np.clip(np.concatenate([private_p,old_p],axis=1),1e-6,1)))
    raw=metrics(records(py,private_p),0,0);app=metrics(records(py,private_p),threshold['min_confidence'],threshold['min_margin'])
    previous=current['private_new'];previous_app=current['private_new_app']
    promote=raw['accuracy_overall']>previous['accuracy_overall'] and raw['macro_f1']>previous['macro_f1'] and app['accuracy_answered']>=previous_app['accuracy_answered'] and app['coverage']>=previous_app['coverage']
    result={'selection':selection,'private_baseline':previous,'private_baseline_app':previous_app,'private_new':raw,'private_new_app':app,'promote':promote,'limitations':['Upstream training membership and FER2013 overlaps are not independently established; scores may be optimistic.','Private split already observed in previous development: reference comparison, not a fresh holdout.','Requires independent camera evaluation; expression does not establish mental state.']}
    (OUT/'report.json').write_text(json.dumps(result,indent=2))
    config={'format':1,'source':metadata,'fusion':fusion,'min_confidence':threshold['min_confidence'],'min_margin':threshold['min_margin']}
    (OUT/'deployment-config.json').write_text(json.dumps(config,indent=2))
    np.savez_compressed(OUT/'selected-predictions.npz',private=private_p,private_ids=np.asarray([p.parent.name+'/'+p.name for _,p in private_paths]),private_labels=py)
    print('FINAL',json.dumps({'promote':promote,'accuracy':raw['accuracy_overall'],'macro_f1':raw['macro_f1'],'answered_accuracy':app['accuracy_answered'],'coverage':app['coverage']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
