"""Equivalent ONNX rewrites for a trainable, pickle-free PyTorch conversion."""
import onnx
from onnx2torch import convert
from onnx2torch.node_converters.pad import OnnxPadDynamic,OnnxPadStatic


def convert_trainable(model,static_padding=False):
    # For a 4D face tensor GlobalMaxPool is exactly ReduceMax over H and W.
    # onnx2torch supports ReduceMax, while its GlobalMaxPool converter is absent.
    for node in model.graph.node:
        if node.op_type=='GlobalMaxPool':
            node.CopyFrom(onnx.helper.make_node('ReduceMax',list(node.input),list(node.output),axes=[2,3],keepdims=1,name=node.name))
    onnx.checker.check_model(model)
    result=convert(model)
    if static_padding:
        def attribute(name):
            value=result
            for part in name.split('.'):value=getattr(value,part)
            return value
        replaced=0
        for node in result.graph.nodes:
            if node.op!='call_module':continue
            module=result.get_submodule(node.target)
            if not isinstance(module,OnnxPadDynamic):continue
            pads_node=node.args[1]
            if getattr(pads_node,'op',None)!='get_attr':raise ValueError('Padding must already be specialized')
            pads=attribute(pads_node.target).tolist()
            fill=node.args[2] if len(node.args)>2 else 0.
            if getattr(fill,'op',None)=='get_attr':fill=float(attribute(fill.target))
            replacement=OnnxPadStatic.create_from_onnx_params(pads,module.mode,float(fill))
            while len(replacement.pads)>4 and replacement.pads[-2:]==[0,0]:replacement.pads=replacement.pads[:-2]
            parent,_,name=node.target.rpartition('.')
            setattr(result.get_submodule(parent) if parent else result,name,replacement)
            node.args=(node.args[0],)
            replaced+=1
        result.graph.eliminate_dead_code();result.recompile()
        result.static_pads_replaced=replaced
    return result
