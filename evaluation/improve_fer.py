"""Select a FER replacement on deduplicated public data, then evaluate private once."""
import csv
import hashlib
import json
import sys
import time
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.runtime/fer-benchmark-deps'))
if os.getenv('FER_EVAL_DML') == '1':
    sys.path.insert(0, str(ROOT / '.runtime/fer-dml'))
import numpy as np
from PIL import Image
import fer_model
from evaluation.fer_benchmark import load_samples, metrics, sweep

DATA = ROOT / '.runtime/fer2013-benchmark-data'
OUT = ROOT / os.getenv('FER_IMPROVEMENT_OUTPUT', 'outputs/fer-improvement-fp32')
MODEL = ROOT / os.getenv('FER_CANDIDATE_MODEL', '.models/fer2013/vit-fer2013-fp32.onnx')


def baseline(split):
    with (ROOT / 'outputs/fer-benchmark' / (split + '_predictions.csv')).open(encoding='utf-8', newline='') as f:
        return {row['image']: np.array([float(row['p_' + k]) for k in fer_model.LABELS]) for row in csv.DictReader(f)}


def run(split, old):
    paths = load_samples(DATA / split)
    cache = OUT / (split + '_views.npz')
    start_index = 0
    views, latency = [], []
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        assert str(saved['model_sha256']) == hashlib.sha256(MODEL.read_bytes()).hexdigest()
        start_index = len(saved['ids'])
        assert list(saved['ids']) == [p.parent.name + '/' + p.name for _, p in paths[:start_index]]
        views, latency = list(saved['views']), list(saved['latency'])
    def checkpoint():
        temporary = cache.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, ids=np.asarray([p.parent.name + '/' + p.name for _, p in paths[:len(views)]]), views=np.asarray(views), latency=np.asarray(latency), model_sha256=np.asarray(hashlib.sha256(MODEL.read_bytes()).hexdigest()))
        temporary.replace(cache)
    for i in range(start_index, len(paths)):
        _, path = paths[i]
        start = time.perf_counter()
        with Image.open(path) as image:
            result = fer_model.probability_views(image, MODEL)
        views.append(result)
        latency.append(1000 * (time.perf_counter() - start))
        if (i + 1) % 100 == 0 or i + 1 == len(paths):
            checkpoint()
            print(f'{split}: {i+1}/{len(paths)}; mean {np.mean(latency):.1f}ms', flush=True)
    views, latency = np.asarray(views), np.asarray(latency)
    return paths, views, latency


def probabilities(views, old, weight, flip):
    new = views.mean(axis=1) if flip else views[:, 0, :]
    return (1 - weight) * new + weight * old


def main():
    OUT.mkdir(exist_ok=True)
    provider = 'CPUExecutionProvider'
    if os.getenv('FER_EVAL_DML') == '1':
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        for name, value in [('batch_size', 2), ('num_channels', 3), ('height', 224), ('width', 224)]:
            options.add_free_dimension_override_by_name(name, value)
        session = ort.InferenceSession(str(MODEL), sess_options=options, providers=[('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider'])
        fer_model._session = lambda path: session
        provider = 'DmlExecutionProvider (benchmark only)'
    data = json.loads((DATA / 'manifest.json').read_text())
    public_old = baseline('public')
    paths, views, latency = run('publicTest', public_old)
    private_hashes = set(data['splits']['privateTest']['pixel_hashes'])
    keep = [i for i, (_, path) in enumerate(paths) if data['splits']['publicTest']['pixel_hashes'][int(path.stem)] not in private_hashes]
    old = np.array([public_old[p.parent.name + '/' + p.name] for _, p in paths])
    labels = [label for label, _ in paths]
    old_records = [(labels[i], old[i]) for i in keep]
    original = metrics(old_records, 0, 0)
    candidates = []
    for flip, weight in ((False, 0.0), (True, 0.0), (True, 0.2), (True, 0.4)):
        scores = probabilities(views, old, weight, flip)
        records = [(labels[i], scores[i]) for i in keep]
        raw = metrics(records, 0, 0)
        _, threshold = sweep(records, .68)
        candidates.append({'flip': flip, 'legacy_weight': weight, 'raw': raw, 'threshold': threshold})
    selected = max(candidates, key=lambda c: (c['raw']['macro_f1'], c['raw']['accuracy_overall'], -c['legacy_weight']))
    decision = {'selection_split': 'publicTest exact overlap with private removed', 'calibration_count': len(keep), 'candidates': candidates, 'selected': selected, 'baseline_public': original,
                'predeclared_promotion_rule': 'public macro-F1 +0.02 and accuracy +0.01; private top-1 accuracy and macro-F1 must both exceed legacy; never select on private',
                'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(), 'inference_provider': provider}
    (OUT / 'public-selection.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print('PUBLIC_SELECTION ' + json.dumps({'flip': selected['flip'], 'weight': selected['legacy_weight'], 'macro_f1': selected['raw']['macro_f1'], 'accuracy': selected['raw']['accuracy_overall']}), flush=True)
    private_old = baseline('private')
    paths, views, latency = run('privateTest', private_old)
    labels = [label for label, _ in paths]
    old = np.array([private_old[p.parent.name + '/' + p.name] for _, p in paths])
    scores = probabilities(views, old, selected['legacy_weight'], selected['flip'])
    records = list(zip(labels, scores))
    threshold = selected['threshold']
    private_raw = metrics(records, 0, 0)
    private_app = metrics(records, threshold['min_confidence'], threshold['min_margin'])
    old_raw, old_app = metrics(list(zip(labels, old)), 0, 0), metrics(list(zip(labels, old)), .5, .12)
    promote = selected['raw']['macro_f1'] >= original['macro_f1'] + .02 and selected['raw']['accuracy_overall'] >= original['accuracy_overall'] + .01 and private_raw['macro_f1'] > old_raw['macro_f1'] and private_raw['accuracy_overall'] > old_raw['accuracy_overall']
    report = {**decision, 'private_baseline': old_raw, 'private_baseline_app': old_app, 'private_new': private_raw, 'private_new_app': private_app, 'promote': promote,
              'latency_ms': {'mean_two_views': float(latency.mean()), 'p95_two_views': float(np.percentile(latency, 95))},
              'limitations': ['Same FER2013 reference data as previous benchmark; not new user-camera validation.', 'Membership of upstream training sets cannot be fully verified; do not claim strictly independent unseen-data accuracy.', 'Scores describe facial-expression labels, never mental state or diagnosis.']}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT / 'private_predictions.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'truth', 'old_prediction', 'new_prediction', 'new_app_prediction', *['p_' + k for k in fer_model.LABELS]])
        for (label, path), old_p, p in zip(paths, old, scores):
            writer.writerow([path.parent.name + '/' + path.name, label, fer_model.LABELS[int(old_p.argmax())], fer_model.LABELS[int(p.argmax())], fer_model.decide_label(p, threshold['min_confidence'], threshold['min_margin']), *p.tolist()])
    print(json.dumps({'promote': promote, 'old_accuracy': old_raw['accuracy_overall'], 'new_accuracy': private_raw['accuracy_overall'], 'old_f1': old_raw['macro_f1'], 'new_f1': private_raw['macro_f1'], 'new_app_accuracy_answered': private_app['accuracy_answered'], 'new_app_coverage': private_app['coverage']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
