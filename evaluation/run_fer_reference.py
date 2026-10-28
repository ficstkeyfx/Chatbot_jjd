"""Reproducible FER2013 reference evaluation; never changes app thresholds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.runtime/fer-benchmark-deps'))
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import fer_model
from evaluation.fer_benchmark import load_samples, metrics, sweep


def evaluate(paths, model_path, *, detector=False, name=''):
    records, rows, durations, errors = [], [], [], []
    for index, (truth, path) in enumerate(paths):
        started = time.perf_counter()
        probabilities, faces_count, error = None, None, None
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
            if detector:
                faces = fer_model.detect_faces(np.asarray(image), detector='yunet')
                faces_count = len(faces)
                if faces_count == 1:
                    probabilities = fer_model.predict_probabilities(fer_model.crop_face(image, faces[0]), model_path)
            else:
                probabilities = fer_model.predict_probabilities(image, model_path)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)[:200]
            errors.append({'image': str(path), 'error': error})
        elapsed = 1000 * (time.perf_counter() - started)
        durations.append(elapsed)
        records.append((truth, probabilities))
        ordered = np.sort(probabilities) if probabilities is not None else None
        rows.append({'image': path.parent.name + '/' + path.name, 'truth': truth,
                     'raw_prediction': fer_model.LABELS[int(np.argmax(probabilities))] if probabilities is not None else 'unclear',
                     'app_prediction': fer_model.decide_label(probabilities) if probabilities is not None else 'unclear',
                     'max_score': float(ordered[-1]) if ordered is not None else None,
                     'margin': float(ordered[-1] - ordered[-2]) if ordered is not None else None,
                     'face_count': faces_count, 'latency_ms': elapsed, 'error': error,
                     **{f'p_{label}': float(probabilities[i]) if probabilities is not None else None for i, label in enumerate(fer_model.LABELS)}})
        if (index + 1) % 250 == 0 or index + 1 == len(paths):
            print(f'{name}: {index + 1}/{len(paths)}', flush=True)
    latency = {'mean_ms': statistics.mean(durations), 'median_ms': statistics.median(durations), 'p95_ms': float(np.percentile(durations, 95)), 'sum_seconds': sum(durations) / 1000}
    return records, rows, latency, errors


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=fer_model.DEFAULT_MODEL_PATH)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--detector-diagnostics', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = a.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(a.model), sess_options=options, providers=['CPUExecutionProvider'])
    fer_model._session = lambda path: session
    started = time.perf_counter()
    public_paths = load_samples(a.dataset / 'publicTest')
    private_paths = load_samples(a.dataset / 'privateTest')
    manifest = json.loads((a.dataset / 'manifest.json').read_text())
    assert len(public_paths) == len(private_paths) == 3589, 'Incomplete split'
    # Warm up runtime; warmup time is excluded from latency metrics.
    with Image.open(public_paths[0][1]) as warmup:
        for _ in range(10):
            fer_model.predict_probabilities(warmup, a.model)
    validation, validation_rows, validation_latency, validation_errors = evaluate(public_paths, a.model, name='publicTest cropped')
    private_hashes = set(manifest['splits']['privateTest']['pixel_hashes'])
    public_hashes = manifest['splits']['publicTest']['pixel_hashes']
    calibration = [record for (_, path), record in zip(public_paths, validation) if public_hashes[int(path.stem)] not in private_hashes]
    grid, selected = sweep(calibration, min_coverage=0.6)
    private, private_rows, private_latency, private_errors = evaluate(private_paths, a.model, name='privateTest cropped')
    for split in manifest['splits'].values():
        split.pop('pixel_hashes', None)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': manifest,
        'model_sha256': hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'model_source': 'https://github.com/amilkh/cs230-fer/tree/678566372d6d4972691bbd539f5e410bb864a083/webapp/public/jsmodel',
        'preprocessing': '48x48 grayscale, pixels/255, mean of original and horizontal flip; no alignment in cropped mode',
        'runtime': {'platform': platform.platform(), 'processor': platform.processor(), 'onnxruntime': version('onnxruntime'), 'numpy': version('numpy'), 'pillow': version('Pillow'), 'threads': a.threads, 'provider': 'CPUExecutionProvider'},
        'public_app_threshold': metrics(validation, fer_model.MIN_CONFIDENCE, fer_model.MIN_MARGIN),
        'private_raw_top1': metrics(private, 0.0, 0.0),
        'private_app_threshold': metrics(private, fer_model.MIN_CONFIDENCE, fer_model.MIN_MARGIN),
        'public_selected_threshold': selected,
        'calibration_split': {'name': 'publicTest with exact privateTest overlaps removed', 'count': len(calibration), 'excluded_rows': len(validation) - len(calibration)},
        'private_public_selected_threshold': metrics(private, selected['min_confidence'], selected['min_margin']) if selected else None,
        'validation_threshold_grid': grid,
        'latency': {'public_cropped': validation_latency, 'private_cropped': private_latency},
        'errors': {'public_cropped': validation_errors, 'private_cropped': private_errors},
        'limitations': [
            'FER2013 reference benchmark, not validation on current app users or Vietnamese camera photos.',
            'Upstream webapp training image membership is unavailable; overlap with FER2013 cannot be ruled out. Do not describe these scores as certified unseen-data performance.',
            'All test labels describe annotated facial expressions, not mental state or diagnosis.',
            'Threshold candidate is selected only on publicTest; privateTest never selects thresholds. No application configuration is changed.',
            'Confidence scores are softmax scores and are not calibrated probabilities of correctness.',
        ],
    }
    write_rows(a.output / 'public_predictions.csv', validation_rows)
    write_rows(a.output / 'private_predictions.csv', private_rows)
    if a.detector_diagnostics:
        if fer_model.resolve_detector('yunet') != 'yunet':
            raise RuntimeError('YuNet model unavailable; refusing silent Haar fallback')
        diagnostic, diagnostic_rows, latency, errors = evaluate(private_paths, a.model, detector=True, name='privateTest YuNet diagnostic')
        report['private_yunet_diagnostic'] = metrics(diagnostic, fer_model.MIN_CONFIDENCE, fer_model.MIN_MARGIN)
        report['detector_face_counts'] = {str(n): sum(r['face_count'] == n for r in diagnostic_rows) for n in sorted({r['face_count'] for r in diagnostic_rows if r['face_count'] is not None})}
        report['latency']['private_yunet_diagnostic'] = latency
        report['errors']['private_yunet_diagnostic'] = errors
        report['yunet_sha256'] = hashlib.sha256(fer_model.yunet_path().read_bytes()).hexdigest()
        report['limitations'].append('YuNet diagnostic uses original 48x48 face crops. It measures rejection on tiny inputs; it is NOT a full-frame webcam benchmark or face detection accuracy.')
        write_rows(a.output / 'private_yunet_diagnostic.csv', diagnostic_rows)
    report['total_seconds'] = time.perf_counter() - started
    (a.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('private_raw_top1', 'private_app_threshold', 'latency', 'total_seconds')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
