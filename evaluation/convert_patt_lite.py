"""Inspect and convert the archived RAF PAtt-Lite checkpoint without custom code.

Run with .runtime/patt-env. TensorFlow is a conversion-only dependency.
"""
import os
os.environ['TF_USE_LEGACY_KERAS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import hashlib,json,sys,zipfile,io,re
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras.src.utils.generic_utils import to_snake_case
import tf2onnx
import onnxruntime as ort
import h5py

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'.models/patt-lite/rafdb.keras'
OUT=ROOT/'outputs/patt-lite'


def load_model():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()=='751a460aeccc7d9243af205db77c82773831a2da59f5ce9c0e0fe1cce158047b'
    with zipfile.ZipFile(MODEL) as archive:
        config=json.loads(archive.read('config.json'))
    allowed={'Functional','InputLayer','TFOpLambda','Conv2D','DepthwiseConv2D',
             'BatchNormalization','ReLU','ZeroPadding2D','Sequential',
             'SeparableConv2D','GlobalAveragePooling2D','Dense','Attention'}
    def check(layer):
        assert layer['class_name'] in allowed,layer['class_name']
        if layer['class_name']=='TFOpLambda':
            assert layer['config']['function'] in {'math.truediv','math.subtract'}
        for child in layer['config'].get('layers',[]):check(child)
    check(config)
    # Keras 2.13 stores legacy dependency paths that tf-keras 2.16 cannot
    # restore directly. Match each built-in layer to its ordered class group,
    # assert every weight shape, and account for every non-optimizer tensor.
    model=keras.models.model_from_json(json.dumps(config))
    consumed=[]
    with zipfile.ZipFile(MODEL) as archive,h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as weights:
        def restore(container,group):
            dependencies=group['_layer_checkpoint_dependencies'];counters={}
            for layer in container.layers:
                if not layer.weights:continue
                base=to_snake_case(layer.__class__.__name__)
                keys=[k for k in dependencies if k==base or re.fullmatch(re.escape(base)+r'_\d+',k)]
                keys.sort(key=lambda k:0 if k==base else int(k.rsplit('_',1)[1]))
                index=counters.get(base,0);counters[base]=index+1
                assert index<len(keys),(layer.name,base,keys)
                saved=dependencies[keys[index]]
                if isinstance(layer,keras.Sequential):restore(layer,saved)
                else:
                    values=[np.asarray(saved['vars'][str(i)]) for i in range(len(layer.weights))]
                    assert [v.shape for v in values]==[tuple(v.shape) for v in layer.weights],layer.name
                    layer.set_weights(values);consumed.extend(saved['vars'][str(i)].name for i in range(len(values)))
        restore(model,weights)
        available=[]
        weights['_layer_checkpoint_dependencies'].visititems(lambda name,value:available.append(value.name) if isinstance(value,h5py.Dataset) else None)
        assert sorted(consumed)==sorted(available),'Not every checkpoint tensor was restored exactly once'
    return model


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    model=load_model()
    print('LOADED',model.input_shape,model.output_shape,model.count_params(),flush=True)
    # The archive embeds RGB /127.5 -1 normalization. The exported graph
    # accepts NCHW RGB [0,1], and a fixed batch of one for web requests.
    head=model.get_layer('classification_head')
    logits=keras.layers.Dense(7,activation=None,name='export_logits')
    logit_model=keras.Model(model.inputs,logits(head.input))
    logits.set_weights(head.get_weights())
    @tf.function(input_signature=[tf.TensorSpec([1,3,224,224],tf.float32,name='input')])
    def predict(x):return logit_model(tf.transpose(x*255.,[0,2,3,1]),training=False)
    destination=ROOT/'.models/patt-lite/patt-lite-raf.onnx'
    tf2onnx.convert.from_function(predict,input_signature=predict.input_signature,opset=17,output_path=str(destination))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(destination),sess_options=options,providers=['CPUExecutionProvider'])
    probe=np.load(ROOT/'.runtime/patt-probe-input.npz',allow_pickle=False)
    x=probe['pixels'].astype(np.float32)
    reference=np.concatenate([np.asarray(model(t[None],training=False)) for t in x])
    actual=np.concatenate([session.run(None,{'input':t.transpose(2,0,1)[None]/255.})[0] for t in x])
    actual=np.exp(actual-actual.max(1,keepdims=True));actual/=actual.sum(1,keepdims=True)
    error=float(abs(reference-actual).max());assert error<.0001,error
    together=np.asarray(model(x,training=False))
    batches=np.concatenate([np.asarray(model(x[i:i+8],training=False)) for i in range(0,len(x),8)])
    report={'source':json.loads((MODEL.parent/'source.json').read_text()),'model_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'parameters':model.count_params(),'probe_samples':len(x),'onnx_probability_max_error':error,
            'batch_all_vs_single_max_difference':float(abs(together-reference).max()),
            'batch8_vs_single_top1_disagreements':int((batches.argmax(1)!=reference.argmax(1)).sum()),
            'single_top1':reference.argmax(1).tolist(),'batch8_top1':batches.argmax(1).tolist(),
            'batch_contract':'Static batch=1; each photo is classified independently; do not mix users or TTA views in attention.'}
    (OUT/'conversion.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
