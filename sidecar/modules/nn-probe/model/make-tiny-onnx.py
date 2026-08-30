"""Writes tiny.onnx, the graph nn-probe runs.

The result is in git - it is a few hundred bytes - so a bare checkout needs
neither python nor this script. Run it only to regenerate the file:

    pip install onnx
    python make-tiny-onnx.py

The graph is y = a @ w + b over fp32, with a of shape [1, 4] and y of shape
[1, 2]. The weights are small integers so the expected output of any input is
arithmetic anyone can do on paper, which is what the tests assert against.
"""

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]], dtype=np.float32)
B = np.array([0.5, -0.5], dtype=np.float32)

graph = helper.make_graph(
    nodes=[
        helper.make_node("MatMul", ["a", "w"], ["p"], name="matmul"),
        helper.make_node("Add", ["p", "b"], ["y"], name="add"),
    ],
    name="tiny",
    inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 4])],
    outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    initializer=[numpy_helper.from_array(W, "w"), numpy_helper.from_array(B, "b")],
)

model = helper.make_model(
    graph,
    producer_name="ffrwd",
    opset_imports=[helper.make_opsetid("", 13)],
)
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "tiny.onnx")
print("tiny.onnx:", len(model.SerializeToString()), "bytes")
