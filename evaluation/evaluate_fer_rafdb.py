"""External, fixed RAF-DB test comparison after public-only FER candidate selection."""
from __future__ import annotations
import gc,hashlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'))
sys.path.insert(0,str(ROOT/'.runtime/fer-dml'))
import httpx
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image
from fer_model import LABELS
from evaluation.fer_benchmark import metrics
OUT=ROOT/'outputs/fer-dima-v2'
DATA=ROOT/'.runtime/rafdb-reference/test.parquet'
DATA_SHA='474d9783da3b8aab2991655dcc3b2c1fa3511cf8170109296c277dd477baf0c7'
MAPPING={1:'surprise',2:'fear',3:'disgust',4:'happy',5:'sad',6:'angry',7:'neutral'}


def softmax(x):
    p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)


def fuse(modern,old,config):
    if not config:return modern
    x=np.log(np.clip(np.concatenate([modern,old],axis=1).astype(np.float64),1e-6,1))
    return softmax(((x-np.array(config['mean']))/np.array(config['scale']))@np.array(config['coefficients']).T+np.array(config['intercept']))


def main():
    deadline=time.time()+1800
    while not (OUT/'report.json').exists():
        if (OUT/'rejected.json').exists():print('Candidate rejected on public; no external promotion',flush=True);return
        if time.time()>deadline:raise RuntimeError('Timed out waiting for selection')
        time.sleep(5)
    assert hashlib.sha256(DATA.read_bytes()).hexdigest()==DATA_SHA
    rows=pq.read_table(DATA).to_pylist()
    assert len(rows)==3068 and set(r['label'] for r in rows)==set(range(1,8))
    labels=[MAPPING[int(r['label'])] for r in rows]
    config=json.loads((OUT/'deployment-config.json').read_text())
    source=json.loads((OUT/'report.json').read_text())
    baseline_fusion=json.loads((ROOT/'fer_vit_fusion.json').read_text())
    models=[('previous',ROOT/'.models/fer2013/vit-fer2013-fp32.onnx',['angry','disgust','fear','happy','neutral','sad','surprise']),('candidate',ROOT/'.models/fer2013/vit-dima-fp32.onnx',['sad','disgust','angry','neutral','fear','surprise','happy'])]
    arrays={}
    cpu_options=ort.SessionOptions();cpu_options.intra_op_num_threads=2;cpu_options.inter_op_num_threads=1
    legacy=ort.InferenceSession(str(ROOT/'.models/fer2013/webapp.onnx'),sess_options=cpu_options,providers=['CPUExecutionProvider'])
    old=[]
    for row in rows:
        with Image.open(io.BytesIO(row['image']['bytes'])) as image:
            pixels=np.asarray(image.convert('L').resize((48,48),Image.Resampling.BILINEAR),np.float32)/255.
        a=legacy.run(None,{legacy.get_inputs()[0].name:pixels[None,:,:,None]})[0][0]
        b=legacy.run(None,{legacy.get_inputs()[0].name:pixels[None,:,::-1,None].copy()})[0][0]
        old.append((a+b)/2)
    old=np.asarray(old)
    for name,path,model_labels in models:
        cache=OUT/f'rafdb-{name}.npz'
        model_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                assert str(saved['model_sha256'])==model_hash and str(saved['data_sha256'])==DATA_SHA
                arrays[name]=saved['probabilities']
            continue
        options=ort.SessionOptions();options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.intra_op_num_threads=2
        for key,value in [('batch_size',8),('num_channels',3),('height',224),('width',224)]:options.add_free_dimension_override_by_name(key,value)
        session=ort.InferenceSession(str(path),sess_options=options,providers=[('DmlExecutionProvider',{'device_id':0}),'CPUExecutionProvider'])
        probabilities=[]
        for offset in range(0,len(rows),4):
            tensors=[]
            for row in rows[offset:offset+4]:
                with Image.open(io.BytesIO(row['image']['bytes'])) as image:
                    pixels=np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),np.float32)/127.5-1.
                tensor=pixels.transpose(2,0,1);tensors.extend([tensor,tensor[:,:,::-1]])
            count=len(tensors);tensors.extend([tensors[-1]]*(8-count))
            logits=session.run(None,{session.get_inputs()[0].name:np.ascontiguousarray(tensors)})[0][:count]
            p=softmax(logits).reshape(-1,2,7).mean(1)
            probabilities.extend(p[:,[model_labels.index(label) for label in LABELS]])
            if len(probabilities)%512==0:print('RAF',name,len(probabilities),flush=True)
        arrays[name]=np.asarray(probabilities)
        np.savez_compressed(cache,probabilities=arrays[name],model_sha256=np.asarray(model_hash),data_sha256=np.asarray(DATA_SHA))
        del session;gc.collect()
    previous=fuse(arrays['previous'],old,baseline_fusion)
    candidate=fuse(arrays['candidate'],old,config['fusion'])
    baseline=metrics(list(zip(labels,previous)),0,0);new=metrics(list(zip(labels,candidate)),0,0)
    previous_app=metrics(list(zip(labels,previous)),.6,.3)
    new_app=metrics(list(zip(labels,candidate)),config['min_confidence'],config['min_margin'])
    promote=new['accuracy_overall']>=baseline['accuracy_overall']+.01 and new['macro_f1']>=baseline['macro_f1']+.01 and new_app['accuracy_answered']>=previous_app['accuracy_answered'] and new_app['coverage']>=previous_app['coverage']
    report={'dataset':{'repo':'elipaluma/RAF-DB_Kaggle','revision':'3a1c56542c8948fe7c4341f992b31194124e9be5','split':'test','rows':len(rows),'parquet_sha256':DATA_SHA,'labels':MAPPING,'mapping_source':'https://www.whdeng.cn/RAF/li_RAFDB_2017_CVPR.pdf'},'baseline':baseline,'candidate':new,'baseline_app':previous_app,'candidate_app':new_app,'promote':promote,'gate':'External accuracy and macro-F1 +0.01; selective accuracy and coverage do not regress; thresholds frozen before RAF evaluation','limitations':['Aligned face crops, not full-frame camera detection.','Public mirror; label mapping follows original RAF-DB paper.','Upstream membership not independently audited; notebook for new model describes another source dataset.','FER2013 candidate score may be contaminated by upstream split/oversampling; external results are the key comparison.']}
    (OUT/'rafdb-report.json').write_text(json.dumps(report,indent=2))
    print('EXTERNAL FINAL',json.dumps({'promote':promote,'old_accuracy':baseline['accuracy_overall'],'new_accuracy':new['accuracy_overall'],'old_f1':baseline['macro_f1'],'new_f1':new['macro_f1'],'old_answered':previous_app['accuracy_answered'],'new_answered':new_app['accuracy_answered'],'old_coverage':previous_app['coverage'],'new_coverage':new_app['coverage']}),flush=True)


if __name__=='__main__':main()
