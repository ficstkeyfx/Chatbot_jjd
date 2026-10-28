"""Nested-CV reliability gate over two independent FER experts and visual features."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'))
os.environ['FER_FUSION_EMBEDDING']='1'
os.environ['FER_JOINT_OUTPUT']='outputs/fer-reliability-v5'
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
from evaluation.train_fer_joint_fusion import data,head_predict
from evaluation.train_fer_multidomain import fuse
from evaluation.train_fer_efficient import current_predictions
from evaluation.calibrate_fer_fusion import records
from evaluation.fer_benchmark import metrics
from fer_model import LABELS
OUT=ROOT/'outputs/fer-reliability-v5'
DATA=ROOT/'outputs/fer-multidomain-v2'


def old_model():return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,solver='lbfgs'))
def gate(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,solver='lbfgs',class_weight='balanced'))
def gate_features(old,expert,visual):return np.concatenate([np.log(np.clip(old,1e-6,1)),np.log(np.clip(expert,1e-6,1)),visual],1)
def combine(old,expert,w,mode):
    if mode=='hard':w=(w>=.5).astype(float)
    return (1-w[:,None])*old+w[:,None]*expert


def main():
    OUT.mkdir(exist_ok=True);d=data();nf=len(d['yfer'])
    x=np.concatenate([d['xfer'],d['xraf']]);y=np.concatenate([d['yfer'],d['yraf']]);groups=np.concatenate([d['fer_hashes'],d['raf_hashes']]);raw=x[:,:14];visual=x[:,21:]
    expert=np.concatenate([d['expert_fer'],d['expert_raf']]);old_full=np.concatenate([d['base_fer'],(d['camera_raf']-.6*d['expert_raf'])/.4])
    weights=np.concatenate([np.ones(nf),np.full(len(d['yraf']),nf/len(d['yraf']))])
    outer=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(x,y,groups));prepared=[]
    for fold,(train,test) in enumerate(outer):
        # The old FER calibration is refit inside each outer split; inner OOF
        # predictions prevent its labels leaking into the reliability targets.
        fer_train=train[train<nf]
        base=old_model();base.fit(raw[fer_train],y[fer_train])
        train_old=base.predict_proba(raw[train]);test_old=base.predict_proba(raw[test])
        positions={int(i):j for j,i in enumerate(train)}
        for inner_train,inner_test in StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(raw[fer_train],y[fer_train],groups[fer_train]):
            model=old_model();model.fit(raw[fer_train[inner_train]],y[fer_train[inner_train]])
            indices=fer_train[inner_test];train_old[[positions[int(i)] for i in indices]]=model.predict_proba(raw[indices])
        useful=(train_old.argmax(1)==y[train])^(expert[train].argmax(1)==y[train]);target=(expert[train].argmax(1)==y[train]).astype(int)
        prepared.append((train,test,train_old,test_old,useful,target))
        print('Nested sources fold',fold+1,'gate training cases',int(useful.sum()),flush=True)
    bf=metrics(records(d['yfer'],d['base_fer']),0,0);br=metrics(records(d['yraf'],d['camera_raf']),0,0)
    candidates=[];predictions=[]
    for c,rare_power in [(c,r) for c in (.0001,.001,.01) for r in (0.,.5)]:
        outputs={mode:np.zeros((len(y),7)) for mode in ('soft','hard')}
        for train,test,old_train,old_test,useful,target in prepared:
            frequency=np.bincount(y[train],minlength=7)
            rarity=(len(train)/np.maximum(frequency,1))**rare_power
            model=gate(c);model.fit(gate_features(old_train[useful],expert[train][useful],visual[train][useful]),target[useful],logisticregression__sample_weight=(weights[train]*rarity[y[train]])[useful])
            w=model.predict_proba(gate_features(old_test,expert[test],visual[test]))[:,1]
            for mode in outputs:outputs[mode][test]=combine(old_test,expert[test],w,mode)
        for mode,p in outputs.items():
            mf=metrics(records(d['yfer'],p[:nf]),0,0);mr=metrics(records(d['yraf'],p[nf:]),0,0)
            candidates.append({'C':c,'rare_power':rare_power,'mode':mode,'fer_oof':mf,'raf_oof':mr});predictions.append(p)
            print('OOF',c,rare_power,mode,'FER',mf['accuracy_overall'],mf['macro_f1'],'RAF',mr['accuracy_overall'],mr['macro_f1'],flush=True)
    selection={'candidates':candidates,'baseline_fer':bf,'baseline_raf_camera':br,'gate':'Both validation/reference domains must gain accuracy and macro-F1 by at least 0.003 over FER-balanced/RAF-camera best baselines','validation':'5 outer grouped folds; old FER classifier has 3-fold inner OOF for gate training; EfficientNet expert excludes all RAF validation; no data-source/name/size flags','camera_config_sha256':d['camera_config_sha256']}
    eligible=[i for i,c in enumerate(candidates) if c['fer_oof']['accuracy_overall']>=bf['accuracy_overall']+.003 and c['fer_oof']['macro_f1']>=bf['macro_f1']+.003 and c['raf_oof']['accuracy_overall']>=br['accuracy_overall']+.003 and c['raf_oof']['macro_f1']>=br['macro_f1']+.003]
    if not eligible:
        (OUT/'rejected.json').write_text(json.dumps(selection,indent=2));print('NO CANDIDATE; current model retained',flush=True);return
    index=max(eligible,key=lambda i:(candidates[i]['fer_oof']['macro_f1']+candidates[i]['raf_oof']['macro_f1'])/2);chosen=candidates[index];selection['selected']=chosen
    useful=(old_full.argmax(1)==y)^(expert.argmax(1)==y);target=(expert.argmax(1)==y).astype(int)
    rarity=(len(y)/np.maximum(np.bincount(y,minlength=7),1))**chosen['rare_power']
    model=gate(chosen['C']);model.fit(gate_features(old_full[useful],expert[useful],visual[useful]),target[useful],logisticregression__sample_weight=(weights*rarity[y])[useful])
    exported=convert_sklearn(model,initial_types=[('features',FloatTensorType([None,1422]))],options={id(model[-1]):{'zipmap':False}},target_opset=17);(OUT/'gate.onnx').write_bytes(exported.SerializeToString())
    all_features=gate_features(old_full,expert,visual);expected=model.predict_proba(all_features);session=ort.InferenceSession(str(OUT/'gate.onnx'),providers=['CPUExecutionProvider']);actual=session.run(None,{'features':all_features.astype(np.float32)})[1]
    parity={'max_probability_difference':float(abs(actual-expected).max()),'top1_disagreements':int(np.sum(actual.argmax(1)!=expected.argmax(1)))};(OUT/'export-parity.json').write_text(json.dumps(parity,indent=2));assert parity['max_probability_difference']<.001 and parity['top1_disagreements']==0,parity
    p=predictions[index];options=[]
    target_f=metrics(records(d['yfer'],d['base_fer']),.6,.3)['coverage'];target_r=metrics(records(d['yraf'],d['camera_raf']),.5,.12)['coverage']
    for confidence in (.3,.4,.5,.6,.7,.8):
        for margin in (0.,.05,.1,.12,.2,.3):
            mf=metrics(records(d['yfer'],p[:nf]),confidence,margin);mr=metrics(records(d['yraf'],p[nf:]),confidence,margin)
            if mf['coverage']>=target_f and mr['coverage']>=target_r:options.append(((mf['accuracy_answered']+mr['accuracy_answered'])/2,confidence,margin))
    _,confidence,margin=max(options);selection['threshold']={'min_confidence':confidence,'min_margin':margin};(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    config=json.loads((ROOT/'fer_camera_config.json').read_text());ids,fy,old_fer=current_predictions('privateTest')
    with np.load(ROOT/'outputs/fer-efficient-v2/privateTest_features.npz',allow_pickle=False) as saved:
        lookup={int(i):j for j,i in enumerate(saved['ids'])};embedding=saved['features'][[lookup[int(Path(path).stem)] for path in ids]]
    ef=head_predict(embedding,config['head']);w=model.predict_proba(gate_features(old_fer,ef,embedding))[:,1];pf=combine(old_fer,ef,w,chosen['mode'])
    with np.load(DATA/'raf-test-features.npz',allow_pickle=False) as saved:embedding=saved['features'];ry=saved['labels']
    er=head_predict(embedding,config['head'])
    with np.load(DATA/'camera-predictions.npz',allow_pickle=False) as saved:old_raf=(saved['raf']-.6*er)/.4
    w=model.predict_proba(gate_features(old_raf,er,embedding))[:,1];pr=combine(old_raf,er,w,chosen['mode'])
    base=json.loads((DATA/'camera-reference.json').read_text());oldf=base['fer_baseline'];oldr=base['raf_new'];mf=metrics(records(fy,pf),0,0);mr=metrics(records(ry,pr),0,0)
    promote=mf['accuracy_overall']>=oldf['accuracy_overall']+.003 and mf['macro_f1']>=oldf['macro_f1']+.003 and mr['accuracy_overall']>=oldr['accuracy_overall']+.003 and mr['macro_f1']>=oldr['macro_f1']+.003
    result={'selection':selection,'fer_baseline':oldf,'fer_new':mf,'raf_baseline':oldr,'raf_new':mr,'fer_app':metrics(records(fy,pf),confidence,margin),'raf_app':metrics(records(ry,pr),confidence,margin),'promote':promote,'limitations':['Reference tests already observed in prior development; not new independent holdouts.','External real-camera quality is not guaranteed; full-frame independent data is still needed.']}
    (OUT/'report.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/'selected-predictions.npz',fer=pf,raf=pr,fer_ids=np.asarray(ids))
    print('FINAL',json.dumps({'promote':promote,'fer_accuracy':mf['accuracy_overall'],'fer_f1':mf['macro_f1'],'raf_accuracy':mr['accuracy_overall'],'raf_f1':mr['macro_f1']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
