"""Specialize shape-only face padding at fixed 260x260, with exact ORT checks."""
import copy
import onnx
import numpy as np
import onnxruntime as ort


def specialize_padding(model):
    pads=[n for n in model.graph.node if n.op_type=='Pad']
    probe=copy.deepcopy(model)
    names=[]
    for node in pads:
        for i,name in enumerate(node.input[1:],1):
            if name and name not in names:
                names.append(name)
                probe.graph.output.append(onnx.helper.make_tensor_value_info(name,onnx.TensorProto.INT64 if i==1 else onnx.TensorProto.FLOAT,[8] if i==1 else []))
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(probe.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
    values={}
    for batch in (1,2,8):
        x=np.random.default_rng(42+batch).normal(size=(batch,3,260,260)).astype('float32')
        actual=session.run(names,{session.get_inputs()[0].name:x})
        for name,value in zip(names,actual):
            if name in values:np.testing.assert_array_equal(value,values[name])
            else:values[name]=value
    output=copy.deepcopy(model)
    for index,node in enumerate(n for n in output.graph.node if n.op_type=='Pad'):
        pad=values[node.input[1]]
        assert len(pad)==8 and np.all(pad[[0,1,4,5]]==0)
        for i,name in enumerate(list(node.input[1:]),1):
            if not name:continue
            replacement=f'fixed_face_pad_{index}_{i}'
            output.graph.initializer.append(onnx.numpy_helper.from_array(values[name],replacement));node.input[i]=replacement
    needed={x.name for x in output.graph.output};kept=[]
    for node in reversed(output.graph.node):
        if any(name in needed for name in node.output):
            kept.append(node);needed.update(name for name in node.input if name)
    initializers=[x for x in output.graph.initializer if x.name in needed]
    del output.graph.node[:];output.graph.node.extend(reversed(kept))
    del output.graph.initializer[:];output.graph.initializer.extend(initializers)
    infos=[x for x in output.graph.value_info if x.name in needed]
    del output.graph.value_info[:];output.graph.value_info.extend(infos)
    onnx.checker.check_model(output)
    return output
