"""Install a frozen EfficientNet joint-domain expert for the explicit camera profile."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REVISION='af833487321c3efdcb1768a91a6c656a1986fdf6'
SOURCE_SHA='7863032d8c0c0bc259aa0dd7b8ebf6b607af454e7a0ea82be2b37767914ecd0f'
DEFAULT_PATH=ROOT/'.models/fer2013/fer-camera.onnx'


def prepare(target:Path=DEFAULT_PATH)->Path:
    import onnx
    from fer_model import LABELS,model_metadata
    from setup_fer_vit import prepare as prepare_vit
    config_path=ROOT/'fer_camera_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    head=config['head']
    mean=np.asarray(head['mean'],np.float64);scale=np.asarray(head['scale'],np.float64)
    weights=np.asarray(head['coefficients'],np.float64);bias=np.asarray(head['intercept'],np.float64)
    if config.get('format')!=1 or head['labels']!=list(LABELS) or mean.shape!=(1408,) or scale.shape!=(1408,) or weights.shape!=(7,1408) or bias.shape!=(7,) or not all(np.isfinite(a).all() for a in (mean,scale,weights,bias)) or (scale<=0).any() or not 0<=config['efficient_weight']<=1:
        raise ValueError('Invalid camera expert configuration')
    if hashlib.sha256((ROOT/'fer_vit_fusion.json').read_bytes()).hexdigest()!=config['vit_fusion_sha256']:
        raise ValueError('Camera companion calibration version mismatch')
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    source=target.with_name('enet_b2_7.onnx')
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:
        temporary=source.with_suffix('.download')
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/{REVISION}/models/affectnet_emotions/onnx/enet_b2_7.onnx',temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('Camera backbone checksum mismatch')
        temporary.replace(source)
    model=onnx.load(str(source));last=model.graph.node[-1]
    if last.op_type!='Gemm' or last.input[1]!='classifier.weight':raise ValueError('Unexpected camera source graph')
    feature=last.input[0];del model.graph.node[-1]
    kept=[x for x in model.graph.initializer if x.name not in {'classifier.weight','classifier.bias'}]
    del model.graph.initializer[:];model.graph.initializer.extend(kept)
    affine=weights/scale;intercept=bias-affine@mean
    model.graph.initializer.extend([onnx.numpy_helper.from_array(affine.astype(np.float32),'camera.weight'),onnx.numpy_helper.from_array(intercept.astype(np.float32),'camera.bias')])
    model.graph.node.append(onnx.helper.make_node('Gemm',[feature,'camera.weight','camera.bias'],['camera_logits'],transB=1))
    del model.graph.output[:];model.graph.output.append(onnx.helper.make_tensor_value_info('camera_logits',onnx.TensorProto.FLOAT,['batch_size',7]))
    onnx.checker.check_model(model)
    temporary=target.with_suffix('.tmp');onnx.save(model,str(temporary));temporary.replace(target)
    prepare_vit(target.with_name('vit-fer2013-fp32.onnx'))
    metadata={'format':1,'architecture':'efficientnet_b2','input_size':260,'mean':[.485,.456,.406],'std':[.229,.224,.225],'labels':list(LABELS),'tta':'flip','tta_reduction':'mean_logits','ensemble_legacy_weight':0.,'min_confidence':config['min_confidence'],'min_margin':config['min_margin'],'companion':{'filename':'vit-fer2013-fp32.onnx','weight':1-config['efficient_weight']},'source_revision':REVISION,'source_sha256':SOURCE_SHA,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'training':config['training']}
    sidecar=Path(str(target)+'.json');temporary=sidecar.with_suffix('.tmp');temporary.write_text(json.dumps(metadata,indent=2),encoding='utf-8');temporary.replace(sidecar)
    model_metadata.cache_clear();model_metadata(str(target.resolve()))
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_PATH);print(prepare(parser.parse_args().output))
