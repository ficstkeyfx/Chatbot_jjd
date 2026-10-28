"""Convert the authors' TensorFlow.js five-layer CNN to ONNX once.

Source: https://github.com/amilkh/cs230-fer/tree/master/webapp/public/jsmodel
License: https://github.com/amilkh/cs230-fer/blob/master/LICENSE (MIT)
"""

from __future__ import annotations

import json
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fer_model import DEFAULT_MODEL_PATH


SOURCE = "https://raw.githubusercontent.com/amilkh/cs230-fer/678566372d6d4972691bbd539f5e410bb864a083/webapp/public/jsmodel/"
SHA256 = {
    "model.json": "4635bd58414d903af1bdf722134fd9f55c2bc27a36a779c1a77345f2d4fb9a2d",
    "group1-shard1of3.bin": "1736b9953eaf60a8b7e062ae7fdcb394e64729702947382fd30a001917c5703f",
    "group1-shard2of3.bin": "ae82d6da0ec66fed9933110c2328ca3d53eb8cd09f0bc164d9436a1bc1b2ac02",
    "group1-shard3of3.bin": "89a4a856535db5bef5962cb17e05fe78738ba1a238f6600414d2197dd0cc599a",
}


def _asset(directory: Path, filename: str) -> bytes:
    if filename not in SHA256:
        raise ValueError("Unexpected model asset")
    path = directory / filename
    if not path.exists():
        urllib.request.urlretrieve(SOURCE + filename, path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA256[filename]:
        raise ValueError(f"Model asset checksum mismatch: {filename}")
    return content


def _read_weights(directory: Path) -> dict[str, np.ndarray]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(_asset(directory, "model.json"))["weightsManifest"][0]
    shards = []
    for filename in manifest["paths"]:
        shards.append(_asset(directory, filename))
    blob = b"".join(shards)
    weights = {}
    offset = 0
    for entry in manifest["weights"]:
        if entry["dtype"] != "float32":
            raise ValueError("Unexpected weight dtype")
        shape = tuple(entry["shape"])
        size = int(np.prod(shape)) * 4
        weights[entry["name"]] = np.frombuffer(blob[offset:offset + size], dtype="<f4").reshape(shape).copy()
        offset += size
    if offset != len(blob):
        raise ValueError("Weight shards do not match the manifest")
    return weights


def convert(target: Path = DEFAULT_MODEL_PATH) -> Path:
    weights = _read_weights(target.parent)
    nodes = []
    initializers = []

    def add(name: str, array: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(np.asarray(array, dtype=np.float32), name))
        return name

    nodes.append(helper.make_node("Transpose", ["face"], ["nchw"], perm=[0, 3, 1, 2]))
    current = "nchw"
    for index, (kernel, channels, padding) in enumerate(((5, 32, [2, 2, 2, 2]),
                                                         (4, 32, [1, 1, 2, 2]),
                                                         (5, 64, [2, 2, 2, 2])), start=1):
        prefix = f"conv{index}"
        conv_kernel = add(prefix + "_kernel", weights[prefix + "/kernel"].transpose(3, 2, 0, 1))
        conv_bias = add(prefix + "_bias", weights[prefix + "/bias"])
        nodes.append(helper.make_node("Conv", [current, conv_kernel, conv_bias],
                                      [prefix], pads=padding, kernel_shape=[kernel, kernel]))
        nodes.append(helper.make_node("Relu", [prefix], [prefix + "_relu"]))
        bn = f"batch_normalization_{index + 8}"
        bn_inputs = [prefix + "_relu"]
        for suffix in ("gamma", "beta", "moving_mean", "moving_variance"):
            bn_inputs.append(add(bn + "_" + suffix, weights[bn + "/" + suffix]))
        nodes.append(helper.make_node("BatchNormalization", bn_inputs,
                                      [prefix + "_bn"], epsilon=0.001))
        nodes.append(helper.make_node("MaxPool", [prefix + "_bn"],
                                      [prefix + "_pool"], kernel_shape=[2, 2], strides=[2, 2]))
        current = prefix + "_pool"
    nodes.append(helper.make_node("Transpose", [current], ["nhwc"], perm=[0, 2, 3, 1]))
    nodes.append(helper.make_node("Flatten", ["nhwc"], ["flat"], axis=1))
    current = "flat"
    for prefix in ("fc1", "fcsoftmax"):
        kernel = add(prefix + "_kernel", weights[prefix + "/kernel"])
        bias = add(prefix + "_bias", weights[prefix + "/bias"])
        nodes.append(helper.make_node("Gemm", [current, kernel, bias], [prefix]))
        current = prefix
        if prefix == "fc1":
            nodes.append(helper.make_node("Relu", [current], ["fc1_relu"]))
            current = "fc1_relu"
    nodes.append(helper.make_node("Softmax", [current], ["probabilities"], axis=1))
    graph = helper.make_graph(
        nodes, "cs230-fer-webapp", [helper.make_tensor_value_info("face", TensorProto.FLOAT, [1, 48, 48, 1])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 7])], initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 11
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    return target


if __name__ == "__main__":
    print(convert())
