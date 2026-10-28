"""Fit a small fusion head on public-only OOF predictions; evaluate private afterwards."""
import csv
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from evaluation.fer_benchmark import metrics, sweep
from fer_model import LABELS

OUT = ROOT / 'outputs/fer-improvement-fp32'


def old_predictions(split):
    with (ROOT / 'outputs/fer-benchmark' / f'{split}_predictions.csv').open(encoding='utf-8', newline='') as file:
        return {row['image']: np.array([float(row['p_' + label]) for label in LABELS]) for row in csv.DictReader(file)}


def inputs(split):
    saved = np.load(OUT / f'{split}_views.npz', allow_pickle=False)
    old = old_predictions('public' if split == 'publicTest' else 'private')
    ids = saved['ids'].tolist()
    modern = saved['views'].mean(1)
    legacy = np.array([old[path] for path in ids])
    x = np.log(np.clip(np.concatenate([modern, legacy], axis=1), 1e-6, 1.0))
    labels = [str(path).split('/')[0] for path in ids]
    return ids, x, np.array([LABELS.index(label) for label in labels]), legacy


def records(y, probs):
    return [(LABELS[int(label)], p) for label, p in zip(y, probs)]


def main():
    data = json.loads((ROOT / '.runtime/fer2013-benchmark-data/manifest.json').read_text())
    ids, x, y, legacy = inputs('publicTest')
    private_hashes = set(data['splits']['privateTest']['pixel_hashes'])
    hashes = np.array([data['splits']['publicTest']['pixel_hashes'][int(Path(path).stem)] for path in ids])
    keep = np.array([h not in private_hashes for h in hashes])
    x, y, groups, legacy = x[keep], y[keep], hashes[keep], legacy[keep]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, y, groups))
    baseline = metrics(records(y, legacy), 0, 0)
    candidates, oof = [], []
    for strength in (.01, .1, 1., 10.):
        for balanced in (False, True):
            probabilities = np.empty((len(y), 7))
            for train, validation in folds:
                assert not set(groups[train]) & set(groups[validation])
                scaler = StandardScaler().fit(x[train])
                model = LogisticRegression(C=strength, class_weight='balanced' if balanced else None, max_iter=1500, solver='lbfgs')
                model.fit(scaler.transform(x[train]), y[train])
                assert list(model.classes_) == list(range(7))
                probabilities[validation] = model.predict_proba(scaler.transform(x[validation]))
            result = metrics(records(y, probabilities), 0, 0)
            candidates.append({'C': strength, 'balanced': balanced, 'oof': result})
            oof.append(probabilities)
            print('PUBLIC OOF', strength, balanced, result['accuracy_overall'], result['macro_f1'], flush=True)
    eligible = [i for i, row in enumerate(candidates) if row['oof']['macro_f1'] >= baseline['macro_f1'] + .02 and row['oof']['accuracy_overall'] >= baseline['accuracy_overall'] + .01]
    if not eligible:
        (OUT / 'fusion-oof-rejected.json').write_text(json.dumps({'baseline': baseline, 'candidates': candidates, 'reason': 'No candidate passed predeclared public improvement gate'}, indent=2))
        raise RuntimeError('Public OOF gate not passed; no deployment')
    index = max(eligible, key=lambda i: (candidates[i]['oof']['macro_f1'], candidates[i]['oof']['accuracy_overall']))
    chosen = candidates[index]
    _, threshold = sweep(records(y, oof[index]), .68)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=chosen['C'], class_weight='balanced' if chosen['balanced'] else None, max_iter=1500, solver='lbfgs').fit(scaler.transform(x), y)
    fusion = {'feature': 'log_probabilities_vit_then_legacy', 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'coefficients': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
              'labels': list(LABELS), 'training_split': 'publicTest only; private exact overlaps excluded', 'public_rows': len(y), 'C': chosen['C'], 'balanced': chosen['balanced']}
    (OUT / 'fusion.json').write_text(json.dumps(fusion, indent=2))
    selection = {'baseline_public': baseline, 'candidates': candidates, 'selected': chosen, 'threshold_from_oof': threshold, 'validation': '5-fold stratified-group OOF; exact pixel duplicates share a fold; no private rows fit the fusion head'}
    (OUT / 'fusion-public-selection.json').write_text(json.dumps(selection, indent=2))
    private_ids, private_x, private_y, private_legacy = inputs('privateTest')
    probs = model.predict_proba(scaler.transform(private_x))
    raw = metrics(records(private_y, probs), 0, 0)
    app = metrics(records(private_y, probs), threshold['min_confidence'], threshold['min_margin'])
    original = metrics(records(private_y, private_legacy), 0, 0)
    promote = raw['accuracy_overall'] > original['accuracy_overall'] and raw['macro_f1'] > original['macro_f1']
    result = {'selection': selection, 'private_baseline': original, 'private_new': raw, 'private_new_app': app, 'promote': promote,
              'fusion_sha256': hashlib.sha256((OUT / 'fusion.json').read_bytes()).hexdigest(),
              'limitations': ['FER2013 reference only; upstream model pretraining membership is not fully verifiable.', 'Private reference scores were previously observed for a rejected fixed ensemble; this is a subsequent development version, not a fresh unseen holdout.', 'No claim about mental state or diagnosis.']}
    (OUT / 'fusion-report.json').write_text(json.dumps(result, indent=2))
    with (OUT / 'fusion-private-predictions.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'truth', 'old', 'new', 'new_app', *['p_' + label for label in LABELS]])
        for path, label, old_p, probability in zip(private_ids, private_y, private_legacy, probs):
            from fer_model import decide_label
            writer.writerow([path, LABELS[int(label)], LABELS[int(old_p.argmax())], LABELS[int(probability.argmax())], decide_label(probability, threshold['min_confidence'], threshold['min_margin']), *probability.tolist()])
    print('FINAL', json.dumps({'promote': promote, 'raw_accuracy': raw['accuracy_overall'], 'macro_f1': raw['macro_f1'], 'app_accuracy_answered': app['accuracy_answered'], 'coverage': app['coverage'], 'threshold': [threshold['min_confidence'], threshold['min_margin']]}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
