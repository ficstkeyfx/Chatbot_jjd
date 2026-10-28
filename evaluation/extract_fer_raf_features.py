"""Extract RAF train/test features for multi-domain training; preserve frozen test."""
import hashlib,io,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'));sys.path.insert(0,str(ROOT/'.runtime/fer-dml'))
import httpx
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image
from evaluation.extract_fer_efficient import preprocess
from fer_model import LABELS
OUT=ROOT/'outputs/fer-multidomain-v2'
DATA=ROOT/'.runtime/rafdb-reference'
MAPPING={1:'surprise',2:'fear',3:'disgust',4:'happy',5:'sad',6:'angry',7:'neutral'}
TRAIN_SHA='3febb40970239c3d3909d0efeca714bec7a603ac32ed505d99d9cfe6c5086c7a'


def main():
    OUT.mkdir(exist_ok=True)
    path=DATA/'train.parquet'
    if not path.exists():
        with httpx.Client(timeout=120,follow_redirects=True) as c:
            r=c.get('https://huggingface.co/datasets/elipaluma/RAF-DB_Kaggle/resolve/3a1c56542c8948fe7c4341f992b31194124e9be5/data/train-00000-of-00001.parquet');r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==TRAIN_SHA;path.write_bytes(r.content)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==TRAIN_SHA
    test_path=DATA/'test.parquet'
    test_sha='474d9783da3b8aab2991655dcc3b2c1fa3511cf8170109296c277dd477baf0c7'
    if not test_path.exists():
        with httpx.Client(timeout=120,follow_redirects=True) as c:
            r=c.get('https://huggingface.co/datasets/elipaluma/RAF-DB_Kaggle/resolve/3a1c56542c8948fe7c4341f992b31194124e9be5/data/test-00000-of-00001.parquet');r.raise_for_status();assert hashlib.sha256(r.content).hexdigest()==test_sha;test_path.write_bytes(r.content)
    assert hashlib.sha256(test_path.read_bytes()).hexdigest()==test_sha
    options=ort.SessionOptions();options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.intra_op_num_threads=2;options.add_free_dimension_override_by_name('batch_size',16)
    model=ROOT/'.models/fer2013/enet_b2_7_features.onnx'
    session=ort.InferenceSession(str(model),sess_options=options,providers=[('DmlExecutionProvider',{'device_id':0}),'CPUExecutionProvider'])
    for split in ('train','test'):
        rows=pq.read_table(DATA/(split+'.parquet')).to_pylist()
        path=OUT/f'raf-{split}-features.npz'
        if path.exists():continue
        features,hashes,labels=[],[],[]
        for offset in range(0,len(rows),8):
            batch=[]
            for row in rows[offset:offset+8]:
                with Image.open(io.BytesIO(row['image']['bytes'])) as image:
                    hashes.append(hashlib.sha256(image.convert('L').tobytes()).hexdigest());tensor=preprocess(image)
                batch.extend([tensor,tensor[:,:,::-1]]);labels.append(LABELS.index(MAPPING[row['label']]))
            count=len(batch);batch.extend([batch[-1]]*(16-count))
            logits,embedding=session.run(None,{session.get_inputs()[0].name:np.ascontiguousarray(batch)})
            features.extend(embedding[:count].reshape(-1,2,1408).mean(1))
            if len(features)%2048==0:print(split,len(features),flush=True)
        np.savez(path,features=np.asarray(features),hashes=np.asarray(hashes),labels=np.asarray(labels),model_sha256=np.asarray(hashlib.sha256(model.read_bytes()).hexdigest()),data_sha256=np.asarray(hashlib.sha256((DATA/(split+'.parquet')).read_bytes()).hexdigest()))
        print(split,'complete',len(features),flush=True)
    # The RAF validation fold is fixed before head/mixture selection.
    from sklearn.model_selection import StratifiedGroupKFold
    with np.load(OUT/'raf-train-features.npz',allow_pickle=False) as saved:
        train,valid=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(saved['features'],saved['labels'],saved['hashes']))
    np.savez(OUT/'raf-fold.npz',train=train,valid=valid)
    rows=pq.read_table(DATA/'train.parquet').to_pylist()
    cpu_options=ort.SessionOptions();cpu_options.intra_op_num_threads=2
    legacy_session=ort.InferenceSession(str(ROOT/'.models/fer2013/webapp.onnx'),sess_options=cpu_options,providers=['CPUExecutionProvider'])
    # Drop the feature session before loading the ViT to limit VRAM usage.
    del session
    import gc;gc.collect()
    options=ort.SessionOptions();options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.intra_op_num_threads=2
    for key,value in [('batch_size',8),('num_channels',3),('height',224),('width',224)]:options.add_free_dimension_override_by_name(key,value)
    session=ort.InferenceSession(str(ROOT/'.models/fer2013/vit-fer2013-fp32.onnx'),sess_options=options,providers=[('DmlExecutionProvider',{'device_id':0}),'CPUExecutionProvider'])
    modern,legacy=[],[]
    for offset in range(0,len(valid),4):
        batch=[]
        for i in valid[offset:offset+4]:
            with Image.open(io.BytesIO(rows[i]['image']['bytes'])) as image:
                pixels=np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),np.float32)/127.5-1.;gray=np.asarray(image.convert('L').resize((48,48),Image.Resampling.BILINEAR),np.float32)/255.
            a=legacy_session.run(None,{legacy_session.get_inputs()[0].name:gray[None,:,:,None]})[0][0];b=legacy_session.run(None,{legacy_session.get_inputs()[0].name:gray[None,:,::-1,None].copy()})[0][0];legacy.append((a+b)/2)
            tensor=pixels.transpose(2,0,1);batch.extend([tensor,tensor[:,:,::-1]])
        count=len(batch);batch.extend([batch[-1]]*(8-count));z=session.run(None,{session.get_inputs()[0].name:np.ascontiguousarray(batch)})[0][:count]
        p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);modern.extend(p.reshape(-1,2,7).mean(1)[:,[0,1,2,3,5,6,4]])
        if len(modern)%512==0:print('validation baseline',len(modern),flush=True)
    np.savez(OUT/'raf-valid-baseline.npz',modern=np.asarray(modern),legacy=np.asarray(legacy),indices=valid)
    (OUT/'features-ready.json').write_text(json.dumps({'ready':True,'raf_train':12271,'raf_test':3068,'valid':len(valid)},indent=2))


if __name__=='__main__':main()
