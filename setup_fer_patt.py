"""Install the bundled, checksum-pinned PAtt-Lite RAF independent-photo export."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEFAULT_PATH=ROOT/'.models/fer2013/patt-lite-raf.onnx'


def prepare(target:Path=DEFAULT_PATH)->Path:
    from fer_model import LABELS,model_metadata
    config=json.loads((ROOT/'fer_patt_config.json').read_text(encoding='utf-8'))
    if config.get('format')!=1 or set(config.get('labels',[]))!=set(LABELS):
        raise ValueError('Invalid PAtt-Lite configuration')
    source=ROOT/'model_assets/patt-lite/patt-lite-raf.onnx'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=config['onnx_sha256']:
        raise ValueError('Bundled PAtt-Lite checksum mismatch')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=config['onnx_sha256']:
        temporary=target.with_suffix('.installing');shutil.copyfile(source,temporary);temporary.replace(target)
    metadata={'format':1,'architecture':'patt_lite','input_size':224,
              'mean':[0.]*3,'std':[1.]*3,'labels':config['labels'],'tta':'none',
              'ensemble_legacy_weight':0.,'min_confidence':config.get('min_confidence',.5),
              'min_margin':config.get('min_margin',.12),'sha256':config['onnx_sha256'],
              'source':config['source'],'evaluation':'RAF checkpoint; independent batch-one inference; experimental'}
    Path(str(target)+'.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    model_metadata.cache_clear();model_metadata(str(target.resolve()))
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_PATH)
    print(prepare(parser.parse_args().output))
