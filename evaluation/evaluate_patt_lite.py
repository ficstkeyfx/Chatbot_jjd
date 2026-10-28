"""Evaluate the archived RAF PAtt-Lite with a fixed, independent-photo contract.

Output labels are recovered on RAF training rows only because the archive does
not include class names. Validation and test predictions never fit the mapping.
"""
import io,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/fer-benchmark-deps'))
import numpy as np
import pyarrow.parquet as pq
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment
from PIL import Image
from evaluation.fer_benchmark import metrics
from evaluation.calibrate_fer_fusion import records
from fer_model import LABELS
OUT=ROOT/'outputs/patt-lite'
PATH=ROOT/'.models/patt-lite/patt-lite-raf.onnx'
RAF_MAP={1:'surprise',2:'fear',3:'disgust',4:'happy',5:'sad',6:'angry',7:'neutral'}


def main():
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(PATH),sess_options=options,providers=['CPUExecutionProvider'])
    def predict(rows,name):
        result=[];start=time.perf_counter()
        for i,row in enumerate(rows):
            with Image.open(io.BytesIO(row['image']['bytes'])) as image:
                x=np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)
            z=session.run(None,{'input':x.transpose(2,0,1)[None]/255.})[0][0]
            p=np.exp(z-z.max());result.append(p/p.sum())
            if (i+1)%500==0:print(name,i+1,'/',len(rows),round(time.perf_counter()-start,1),'seconds',flush=True)
        return np.asarray(result)
    raf_train=pq.read_table(ROOT/'.runtime/rafdb-reference/train.parquet').to_pylist()
    with np.load(ROOT/'outputs/fer-multidomain-v2/raf-fold.npz',allow_pickle=False) as fold:
        train_indices=fold['train'];valid_indices=fold['valid']
    chosen=[]
    for label in range(1,8):chosen.extend([int(i) for i in train_indices if raf_train[int(i)]['label']==label][:100])
    calibration=[raf_train[i] for i in chosen]
    raw=predict(calibration,'label-map train');y=np.array([LABELS.index(RAF_MAP[r['label']]) for r in calibration])
    matrix=np.zeros((7,7))
    for label,p in zip(y,raw):matrix[label]+=p
    canonical,outputs=linear_sum_assignment(-matrix)
    mapping=np.zeros(7,dtype=int);mapping[canonical]=outputs
    labels=[None]*7
    for i,j in enumerate(mapping):labels[j]=LABELS[i]
    label_report={'method':'Hungarian assignment maximizing probability mass on at most 100 training rows per class, outside RAF validation.',
                  'training_indices':chosen,'labels':labels,'probability_mass':matrix.tolist(),
                  'calibration_accuracy':float((raw[:,mapping].argmax(1)==y).mean())}
    (OUT/'labels.json').write_text(json.dumps(label_report,indent=2))
    # Lock the mapping before reading test images/labels.
    result={'source':json.loads((PATH.parent/'source.json').read_text()),
            'onnx_sha256':hashlib.sha256(PATH.read_bytes()).hexdigest(),'labels':labels,
            'input':'RGB [0,1], bilinear 224x224; normalization embedded in checkpoint; batch=1; no flip TTA',
            'limitations':['Archive is a preserved copy of the unavailable author repository, not a currently verified author download.',
                           'raf_validation is a subset of the upstream checkpoint RAF training set; not an unseen validation result for this checkpoint.',
                           'RAF checkpoint training provenance is not independently verified; reference tests have been viewed before in development.',
                           'This is the RAF checkpoint, not the unavailable FER2013 checkpoint used for the paper FER score.']}
    for name,rows,y in [
        ('raf_validation',[raf_train[int(i)] for i in valid_indices],np.array([LABELS.index(RAF_MAP[raf_train[int(i)]['label']]) for i in valid_indices])),
        ('raf_test',pq.read_table(ROOT/'.runtime/rafdb-reference/test.parquet').to_pylist(),None),
        ('fer_private',pq.read_table(ROOT/'.runtime/fer2013-benchmark-data/privateTest.parquet').to_pylist(),None)]:
        if y is None:y=np.array([r['label'] if name=='fer_private' else LABELS.index(RAF_MAP[r['label']]) for r in rows])
        probs=predict(rows,name)[:,mapping]
        result[name]=metrics(records(y,probs),0,0)
        np.savez_compressed(OUT/(name+'-predictions.npz'),probabilities=probs,labels=y)
        (OUT/'report.json').write_text(json.dumps(result,indent=2))
        print('RESULT',name,result[name]['accuracy_overall'],result[name]['macro_f1'],flush=True)


if __name__=='__main__':main()
