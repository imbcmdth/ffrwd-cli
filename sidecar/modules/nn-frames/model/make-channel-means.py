"""Writes the three graphs nn-frames runs: each channel's mean and largest
value of an image tensor, or of each image of a batch.

The results are in git - each is a few hundred bytes - so a bare checkout
needs neither python nor this script. Run it only to regenerate them:

    uvx --from onnx --with numpy python make-channel-means.py

channel-means.onnx takes `image`, fp32 [1, 3, h, w] (NCHW), and
channel-means-nhwc.onnx takes `image`, fp32 [1, h, w, 3] (NHWC); h and w are
free, so one graph takes a tensor at any size. Both return `mean` and `max`,
fp32 [1, 3]: per channel, the mean over every pixel and the largest value.
Six numbers a test can work out from the frame on its own.

channel-means-batch.onnx takes `image`, fp32 [n, 3, h, w], n images of a batch
(crops of one frame, say), and returns `mean` and `max`, fp32 [n, 3]: the same
six numbers for each image, in order.
"""

import onnx
from onnx import TensorProto, helper


def graph(name: str, shape: list, axes: list, batch: int | str) -> onnx.ModelProto:
    built = helper.make_graph(
        nodes=[
            helper.make_node("ReduceMean", ["image"], ["mean"], axes=axes, keepdims=0),
            helper.make_node("ReduceMax", ["image"], ["max"], axes=axes, keepdims=0),
        ],
        name=name,
        inputs=[helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
        outputs=[
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [batch, 3]),
            helper.make_tensor_value_info("max", TensorProto.FLOAT, [batch, 3]),
        ],
    )
    model = helper.make_model(
        built,
        producer_name="ffrwd",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


for file, shape, axes, batch in [
    ("channel-means.onnx", [1, 3, "h", "w"], [2, 3], 1),
    ("channel-means-nhwc.onnx", [1, "h", "w", 3], [1, 2], 1),
    ("channel-means-batch.onnx", ["n", 3, "h", "w"], [2, 3], "n"),
]:
    model = graph(file.removesuffix(".onnx"), shape, axes, batch)
    onnx.save(model, file)
    print(f"{file}:", len(model.SerializeToString()), "bytes")
