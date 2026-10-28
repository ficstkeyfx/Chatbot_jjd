"""Evaluate one frozen validation-selected fine-tune; never select on test scores."""
import hashlib,io,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'))
if os.getenv('FER_EVAL_DML')=='1':sys.path.insert(0,str(ROOT/'.runtime/fer-dml'))
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import cv2
from PIL import Image
from evaluation.calibrate_fer_fusion import records
from evaluation.fer_benchmark import metrics
from evaluation.train_fer_efficient import current_predictions
from evaluation.train_fer_joint_fusion import head_predict
from fer_model import LABELS
OUT=ROOT/'outputs/fer-finetune-v6'
OLD=ROOT/'outputs/fer-multidomain-v2'


def pixels(encoded):
    with Image.open(io.BytesIO(encoded)) as image:rgb=cv2.resize(np.asarray(image.convert('RGB')),(260,260))
    x=rgb.astype(np.float32)/255.
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return x.transpose(2,0,1)


def softmax(z):
    p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)


def main():
    selected=json.loads((OUT/'selected.json').read_text())
    # Preserve the reference tests when the training run already failed its
    # predeclared validation gate. A favorable test score cannot rescue it.
    if not selected['validation']['selected']['eligible']:
        report={'selected':selected,'promote':False,'test_evaluated':False,
                'reason':'Frozen candidate did not improve both validation domains; reference tests were not opened.',
                'production_changed':False}
        (OUT/'report.json').write_text(json.dumps(report,indent=2))
        print('REJECTED ON VALIDATION; reference tests not evaluated; keep production',flush=True)
        return
    path=OUT/'expert.onnx';assert hashlib.sha256(path.read_bytes()).hexdigest()==selected['model_sha256']
    model=onnx.load(str(path));onnx.checker.check_model(model)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    cpu=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    first=cpu.get_inputs()[0].shape[0];batch=16 if not isinstance(first,int) else first
    providers=['CPUExecutionProvider'];accelerated=False
    if os.getenv('FER_EVAL_DML')=='1':
        options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        symbol=model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        if symbol:options.add_free_dimension_override_by_name(symbol,batch)
        providers=[('DmlExecutionProvider',{'device_id':0}),'CPUExecutionProvider'];accelerated=True
    session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
    def run_batch(tensors):
        if batch==1:return np.concatenate([session.run(None,{session.get_inputs()[0].name:t[None]})[0] for t in tensors])
        padded=list(tensors)+[tensors[-1]]*(batch-len(tensors));return session.run(None,{session.get_inputs()[0].name:np.ascontiguousarray(padded)})[0][:len(tensors)]
    public=pq.read_table(ROOT/'.runtime/fer2013-benchmark-data/publicTest.parquet').to_pylist()
    indices=[]
    for label in range(7):indices.extend([i for i,r in enumerate(public) if r['label']==label][:3])
    differences=[];disagreements=0
    for i in indices:
        x=pixels(public[i]['image']['bytes']);tensors=np.array([x,x[:,:,::-1]])
        if batch==1:actual=run_batch(tensors)
        else:actual=run_batch(tensors)
        expected=cpu.run(None,{cpu.get_inputs()[0].name:tensors if not isinstance(first,int) or first==2 else tensors[:1]})[0]
        if first==1:expected=np.concatenate([expected,cpu.run(None,{cpu.get_inputs()[0].name:tensors[1:]})[0]])
        differences.append(float(abs(actual-expected).max()));disagreements+=int(actual.mean(0).argmax()!=expected.mean(0).argmax())
    parity={'samples':len(indices),'max_logit_difference':max(differences),'top1_disagreements':disagreements,'provider':str(providers),'batch':batch};(OUT/'provider-agreement.json').write_text(json.dumps(parity,indent=2));assert max(differences)<.001 and disagreements==0,parity
    def predict(split,rows):
        cache=OUT/(split+'-expert-predictions.npz')
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                assert str(saved['sha256'])==selected['model_sha256'];return saved['probabilities']
        p=[];chunk=max(1,batch//2);start=time.perf_counter()
        for offset in range(0,len(rows),chunk):
            tensors=[]
            for row in rows[offset:offset+chunk]:
                x=pixels(row['image']['bytes']);tensors.extend([x,x[:,:,::-1]])
            z=run_batch(tensors)
            p.extend(softmax(z.reshape(-1,2,7).mean(1)))
            if len(p)%512==0 or len(p)==len(rows):print(split,len(p),'/',len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
        p=np.asarray(p);np.savez_compressed(cache,probabilities=p,sha256=np.asarray(selected['model_sha256']));return p
    fer_rows=pq.read_table(ROOT/'.runtime/fer2013-benchmark-data/privateTest.parquet').to_pylist();raf_rows=pq.read_table(ROOT/'.runtime/rafdb-reference/test.parquet').to_pylist()
    ef=predict('fer-private',fer_rows);er=predict('raf-test',raf_rows)
    ids,y,old_f=current_predictions('privateTest');lookup={int(Path(s).stem):i for i,s in enumerate(ids)};old_f=old_f[[lookup[i] for i in range(len(fer_rows))]];fy=np.array([r['label'] for r in fer_rows])
    config=json.loads((ROOT/'fer_camera_config.json').read_text())
    with np.load(OLD/'raf-test-features.npz',allow_pickle=False) as saved:old_expert=head_predict(saved['features'],config['head']);ry=saved['labels']
    with np.load(OLD/'camera-predictions.npz',allow_pickle=False) as saved:camera_r=saved['raf'];old_r=(camera_r-.6*old_expert)/.4
    weight=selected['weight'];pf=(1-weight)*old_f+weight*ef;pr=(1-weight)*old_r+weight*er
    # Confidence thresholds are derived only from the frozen checkpoint's validation outputs.
    with np.load(OUT/'best-validation.npz',allow_pickle=False) as v:
        vf=(1-weight)*v['fer_old']+weight*v['fer_expert'];vr=(1-weight)*v['raf_old']+weight*v['raf_expert'];vfy=v['fer_labels'];vry=v['raf_labels']
        coverage_f=metrics(records(vfy,v['fer_old']),.6,.3)['coverage']
        with np.load(ROOT/'outputs/fer-reliability-v5/expert-validation.npz',allow_pickle=False) as outside:camera_validation=.4*v['raf_old']+.6*outside['probabilities']
        coverage_r=metrics(records(vry,camera_validation),.5,.12)['coverage']
    thresholds=[]
    for confidence in (.3,.4,.5,.6,.7,.8):
        for margin in (0.,.05,.1,.12,.2,.3):
            mf=metrics(records(vfy,vf),confidence,margin);mr=metrics(records(vry,vr),confidence,margin)
            if mf['coverage']>=min(.95,coverage_f+.02) and mr['coverage']>=min(.95,coverage_r+.02):thresholds.append(((mf['accuracy_answered']+mr['accuracy_answered'])/2,confidence,margin))
    _,confidence,margin=max(thresholds)
    bf=metrics(records(fy,old_f),0,0);br=metrics(records(ry,camera_r),0,0);mf=metrics(records(fy,pf),0,0);mr=metrics(records(ry,pr),0,0)
    promote=mf['accuracy_overall']>=bf['accuracy_overall']+.003 and mf['macro_f1']>=bf['macro_f1']+.003 and mr['accuracy_overall']>=br['accuracy_overall']+.003 and mr['macro_f1']>=br['macro_f1']+.003
    report={'selected':selected,'fer_baseline':bf,'fer_new':mf,'raf_baseline_camera':br,'raf_new':mr,'fer_app':metrics(records(fy,pf),confidence,margin),'raf_app':metrics(records(ry,pr),confidence,margin),'threshold':{'min_confidence':confidence,'min_margin':margin},'promote':promote,'provider_agreement':parity,'accelerated':accelerated,'limitations':['These reference tests have been observed in prior development; not fresh independent holdouts.','Training/checkpoint/mixture selection used only train/validation, never test labels.','No guarantee on unseen full-frame camera images; detector and user-image validation remains necessary.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'final-predictions.npz',fer=pf,raf=pr,fer_labels=fy,raf_labels=ry,fer_old=old_f,raf_camera_old=camera_r)
    print('FINAL',json.dumps({'promote':promote,'fer_accuracy':mf['accuracy_overall'],'fer_f1':mf['macro_f1'],'raf_accuracy':mr['accuracy_overall'],'raf_f1':mr['macro_f1'],'weight':weight}),flush=True)


if __name__=='__main__':main()
