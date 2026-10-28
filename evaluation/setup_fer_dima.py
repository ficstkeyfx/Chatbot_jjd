"""Prepare the pinned alternate ViT with public-selected FER calibration."""
from __future__ import annotations
import argparse
import sys
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPOSITORY='Xenova/facial_emotions_image_detection'
REVISION='e039a363ab5d9e1b82eb08b3af90b4648cb8f696'
SHA256='3aac38bf9d6119956282fb5dfab1919ffcd4b72422e0be6436e3ca2a084265d4'
DEFAULT_PATH=ROOT/'.models/fer2013/vit-dima-fp32.onnx'
URL=f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/onnx/model.onnx'


def prepare(target:Path=DEFAULT_PATH,config_path:Path|None=None)->Path:
    from fer_model import model_metadata
    config_path=config_path or ROOT/'fer_dima_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    source=config.get('source',{})
    if config.get('format')!=1 or source.get('sha256')!=SHA256 or source.get('repository')!=REPOSITORY or source.get('revision')!=REVISION:
        raise ValueError('FER configuration does not match pinned weights')
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=SHA256:
        temporary=target.with_suffix('.download')
        urllib.request.urlretrieve(URL,temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SHA256:
            raise ValueError('FER model checksum mismatch')
        temporary.replace(target)
    metadata={'format':1,'architecture':'vit','input_size':224,'mean':[.5]*3,'std':[.5]*3,
              'labels':['sad','disgust','angry','neutral','fear','surprise','happy'],'tta':'flip','ensemble_legacy_weight':0.,
              'repository':REPOSITORY,'revision':REVISION,'sha256':SHA256,'upstream':'https://huggingface.co/dima806/facial_emotions_image_detection',
              'license':'Apache-2.0 (upstream model)','min_confidence':config['min_confidence'],'min_margin':config['min_margin'],
              'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'evaluation':'FER2013 reference + fixed external RAF-DB test; upstream FER split contamination possible'}
    if config.get('fusion'):
        metadata['fusion']=config['fusion']
        from setup_fer_model import convert
        legacy=target.with_name('webapp.onnx')
        if not legacy.is_file():convert(legacy)
    sidecar=Path(str(target)+'.json')
    # Validate before replacing an existing working sidecar.
    temporary=Path(str(target)+'.validation.onnx')
    validation=Path(str(temporary)+'.json')
    validation.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    try:
        model_metadata.cache_clear()
        model_metadata(str(temporary.resolve()))
        validation.replace(sidecar)
    finally:
        validation.unlink(missing_ok=True)
        model_metadata.cache_clear()
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_PATH)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    print(prepare(args.output,args.config))
