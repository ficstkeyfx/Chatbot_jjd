"""Extract frozen EmotiEffLib features from pinned FER2013 data, with CPU/GPU parity checks."""
from __future__ import annotations
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.runtime/fer-benchmark-deps'))
if os.getenv('FER_EVAL_DML') == '1':
    sys.path.insert(0, str(ROOT / '.runtime/fer-dml'))
import cv2
import httpx
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image

DATA = ROOT / '.runtime/fer2013-benchmark-data'
OUT = ROOT / 'outputs/fer-efficient-v2'
SOURCE = ROOT / '.models/fer2013/enet_b2_7.onnx'
FEATURE_MODEL = ROOT / '.models/fer2013/enet_b2_7_features.onnx'
MODEL_SHA = '7863032d8c0c0bc259aa0dd7b8ebf6b607af454e7a0ea82be2b37767914ecd0f'
REVISION = 'af833487321c3efdcb1768a91a6c656a1986fdf6'
TRAIN_SHA = 'b1c4bee933d41947d7d67648a54f3877de9365703f40117e0d59495c8eb6647a'
TRAIN_URL = 'https://huggingface.co/datasets/Aaryan333/fer2013_train_publicTest_privateTest/resolve/da8d1643649c86bfc2cffba5b744b51fdf329212/data/train-00000-of-00001-5eab84e1c6a2fc27.parquet'
BATCH = 16


def download(url, path, expected):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    partial = path.with_suffix('.part')
    with httpx.Client(timeout=120, follow_redirects=True) as client, client.stream('GET', url) as response:
        response.raise_for_status()
        with partial.open('wb') as file:
            for chunk in response.iter_bytes(1024*1024):
                file.write(chunk)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
        raise ValueError('Pinned file checksum mismatch')
    partial.replace(path)


def preprocess(image):
    # Match upstream HSEmotion ONNX: RGB, OpenCV INTER_LINEAR, ImageNet normalization.
    pixels = cv2.resize(np.asarray(image.convert('RGB')), (260, 260)).astype(np.float32)/255.
    pixels = (pixels - np.array([.485,.456,.406], np.float32))/np.array([.229,.224,.225], np.float32)
    return pixels.transpose(2,0,1)


def softmax(logits):
    p = np.exp(logits-logits.max(-1,keepdims=True))
    return p/p.sum(-1,keepdims=True)


def main():
    OUT.mkdir(exist_ok=True)
    download(f'https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/{REVISION}/models/affectnet_emotions/onnx/enet_b2_7.onnx', SOURCE, MODEL_SHA)
    download(TRAIN_URL, DATA/'train.parquet', TRAIN_SHA)
    model = onnx.load(str(SOURCE))
    last = model.graph.node[-1]
    assert last.op_type == 'Gemm' and last.input[1] == 'classifier.weight'
    weights = onnx.numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == last.input[1]))
    dim = weights.shape[1]
    model.graph.output.append(onnx.helper.make_tensor_value_info(last.input[0], onnx.TensorProto.FLOAT, ['batch_size',dim]))
    onnx.checker.check_model(model)
    onnx.save(model, str(FEATURE_MODEL))
    model_hash = hashlib.sha256(FEATURE_MODEL.read_bytes()).hexdigest()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    provider = 'CPUExecutionProvider'
    providers = [provider]
    if os.getenv('FER_EVAL_DML') == '1':
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_free_dimension_override_by_name('batch_size', BATCH)
        providers = [('DmlExecutionProvider',{'device_id':0}), 'CPUExecutionProvider']
        provider = 'DirectML device 0'
    session = ort.InferenceSession(str(FEATURE_MODEL), sess_options=options, providers=providers)
    metadata = json.loads((DATA/'manifest.json').read_text())
    evaluation_hashes = set(metadata['splits']['publicTest']['pixel_hashes']) | set(metadata['splits']['privateTest']['pixel_hashes'])
    train_seen = set()
    summary = {'source_model_sha256':MODEL_SHA,'feature_model_sha256':model_hash,'revision':REVISION,'feature_dim':int(dim),'provider':provider,'preprocess':'RGB OpenCV linear 260 ImageNet mean/std; original+flip mean','splits':{}}
    checked = False
    for split in ('publicTest','train','privateTest'):
        rows = pq.read_table(DATA/(split+'.parquet')).to_pylist()
        included, removed, duplicate = [], 0, 0
        hashes = []
        for index, row in enumerate(rows):
            with Image.open(io.BytesIO(row['image']['bytes'])) as image:
                digest = hashlib.sha256(image.convert('L').tobytes()).hexdigest()
            if split == 'train':
                if digest in evaluation_hashes:
                    removed += 1
                    continue
                if digest in train_seen:
                    duplicate += 1
                    continue
                train_seen.add(digest)
            included.append(index)
            hashes.append(digest)
        summary['splits'][split] = {'rows':len(included),'removed_evaluation_overlaps':removed,'removed_train_duplicates':duplicate,'parquet_sha256':hashlib.sha256((DATA/(split+'.parquet')).read_bytes()).hexdigest()}
        cache = OUT/(split+'_features.npz')
        features, probabilities, done = [], [], 0
        if cache.exists():
            saved = np.load(cache, allow_pickle=False)
            assert str(saved['model_sha256']) == model_hash
            assert str(saved['data_sha256']) == summary['splits'][split]['parquet_sha256']
            done = len(saved['features'])
            assert saved['ids'].tolist() == included[:done]
            features, probabilities = list(saved['features']), list(saved['probabilities'])
            saved.close()
        def checkpoint():
            temporary = cache.with_suffix('.tmp.npz')
            n = len(features)
            np.savez(temporary, ids=np.asarray(included[:n]), labels=np.array([rows[i]['label'] for i in included[:n]]), hashes=np.asarray(hashes[:n]), features=np.asarray(features), probabilities=np.asarray(probabilities), model_sha256=np.asarray(model_hash), data_sha256=np.asarray(summary['splits'][split]['parquet_sha256']))
            for attempt in range(30):
                try:
                    temporary.replace(cache)
                    break
                except PermissionError:
                    if attempt == 29:
                        raise
                    time.sleep(.2)
        start = time.perf_counter()
        for offset in range(done, len(included), BATCH//2):
            selected = included[offset:offset+BATCH//2]
            pixels = []
            for index in selected:
                with Image.open(io.BytesIO(rows[index]['image']['bytes'])) as image:
                    tensor = preprocess(image)
                pixels.extend([tensor,tensor[:,:,::-1]])
            count = len(pixels)
            pixels.extend([pixels[-1]]*(BATCH-count))
            batch = np.ascontiguousarray(pixels, dtype=np.float32)
            logits, embedding = session.run(None, {session.get_inputs()[0].name:batch})
            if not checked:
                cpu_options = ort.SessionOptions(); cpu_options.intra_op_num_threads=2
                cpu = ort.InferenceSession(str(FEATURE_MODEL), sess_options=cpu_options, providers=['CPUExecutionProvider'])
                cpu_logits,cpu_features = cpu.run(None,{cpu.get_inputs()[0].name:batch[:2]})
                parity={'max_logit_difference':float(abs(cpu_logits-logits[:2]).max()),'max_feature_difference':float(abs(cpu_features-embedding[:2]).max()),'label_disagreements':int(np.sum(cpu_logits.argmax(1)!=logits[:2].argmax(1)))}
                (OUT/'provider-parity.json').write_text(json.dumps(parity,indent=2))
                assert parity['max_feature_difference']<.005 and parity['label_disagreements']==0, parity
                del cpu
                checked=True
            assert np.isfinite(embedding).all() and np.isfinite(logits).all()
            features.extend(embedding[:count].reshape(-1,2,dim).mean(1))
            # Upstream order angry/disgust/fear/happy/neutral/sad/surprise -> app canonical.
            probabilities.extend(softmax(logits[:count]).reshape(-1,2,7).mean(1)[:,[0,1,2,3,5,6,4]])
            if len(features)%2048==0 or len(features)==len(included):
                checkpoint()
                print(f'{split} {len(features)}/{len(included)} {time.perf_counter()-start:.1f}s',flush=True)
        summary['splits'][split]['seconds']=time.perf_counter()-start
        (OUT/'extraction.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
