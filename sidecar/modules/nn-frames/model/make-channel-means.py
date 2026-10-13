"""Writes the four graphs nn-frames runs: each channel's mean and largest
value of an image tensor, or of each image of a batch or a clip.

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

channel-means-clip.onnx takes `image`, fp32 [1, n, 3, h, w], a clip of n
frames as a video model takes one, and returns `mean` and `max`, fp32
[1, n, 3]: the six numbers of each frame, in order.
"""

import onnx
from onnx import TensorProto, helper


def graph(name: str, shape: list, axes: list, out: list) -> onnx.ModelProto:
    built = helper.make_graph(
        nodes=[
            helper.make_node("ReduceMean", ["image"], ["mean"], axes=axes, keepdims=0),
            helper.make_node("ReduceMax", ["image"], ["max"], axes=axes, keepdims=0),
        ],
        name=name,
        inputs=[helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
        outputs=[
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, out),
            helper.make_tensor_value_info("max", TensorProto.FLOAT, out),
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


for file, shape, axes, out in [
    ("channel-means.onnx", [1, 3, "h", "w"], [2, 3], [1, 3]),
    ("channel-means-nhwc.onnx", [1, "h", "w", 3], [1, 2], [1, 3]),
    ("channel-means-batch.onnx", ["n", 3, "h", "w"], [2, 3], ["n", 3]),
    ("channel-means-clip.onnx", [1, "n", 3, "h", "w"], [3, 4], [1, "n", 3]),
]:
    model = graph(file.removesuffix(".onnx"), shape, axes, out)
    onnx.save(model, file)
    print(f"{file}:", len(model.SerializeToString()), "bytes")
