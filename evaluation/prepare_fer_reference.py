"""Fetch pinned public data as Parquet; never execute a dataset loader or pickle."""
import hashlib
import io
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime/fer-benchmark-deps'))
import httpx
import pyarrow.parquet as pq
from PIL import Image

REPO = 'Aaryan333/fer2013_train_publicTest_privateTest'
REVISION = 'da8d1643649c86bfc2cffba5b744b51fdf329212'
LABELS = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
root = Path(__file__).resolve().parents[1]
destination = root / '.runtime/fer2013-benchmark-data'
destination.mkdir(exist_ok=True)
for name in ('httpx', 'httpx2'):
    logging.getLogger(name).setLevel(logging.WARNING)
manifest = {'repo': REPO, 'revision': REVISION, 'source': 'https://huggingface.co/datasets/' + REPO, 'label_mapping': dict(enumerate(LABELS)), 'splits': {}}
with httpx.Client(timeout=120, follow_redirects=True) as client:
    response = client.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data')
    response.raise_for_status()
    entries = response.json()
    for split in ('publicTest', 'privateTest'):
        entry = next(e for e in entries if e['path'].startswith('data/' + split + '-'))
        path = destination / (split + '.parquet')
        expected = (entry.get('lfs') or {}).get('oid')
        if not path.exists() or (expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            print('Downloading ' + split, flush=True)
            partial = path.with_suffix('.part')
            with client.stream('GET', f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/' + entry['path']) as download:
                download.raise_for_status()
                with partial.open('wb') as out:
                    for chunk in download.iter_bytes(1024 * 1024):
                        out.write(chunk)
            actual = hashlib.sha256(partial.read_bytes()).hexdigest()
            if expected and actual != expected:
                raise ValueError('Dataset SHA256 mismatch')
            partial.replace(path)
        table = pq.read_table(path)
        rows = table.to_pylist()
        assert len(rows) == 3589, f'Unexpected split size: {split} {len(rows)}'
        counts, hashes = {}, []
        for index, row in enumerate(rows):
            label = LABELS[row['label']]
            image_data = row['image']['bytes']
            with Image.open(io.BytesIO(image_data)) as image:
                assert image.size == (48, 48), image.size
                image = image.convert('L')
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                folder = destination / split / label
                folder.mkdir(parents=True, exist_ok=True)
                image.save(folder / f'{index:05d}.png')
            hashes.append(pixel_hash)
            counts[label] = counts.get(label, 0) + 1
        manifest['splits'][split] = {'count': len(rows), 'class_counts': counts, 'parquet_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'upstream_lfs_sha256': expected, 'file': entry['path'], 'exact_duplicate_pixels': len(hashes) - len(set(hashes)), 'pixel_hashes': hashes}
        print(json.dumps({'split': split, 'count': len(rows), 'classes': counts}), flush=True)
manifest['public_private_exact_overlap'] = len(set(manifest['splits']['publicTest']['pixel_hashes']) & set(manifest['splits']['privateTest']['pixel_hashes']))
(destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Prepared; exact pixel overlap: ' + str(manifest['public_private_exact_overlap']), flush=True)
