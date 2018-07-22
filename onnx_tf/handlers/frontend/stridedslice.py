from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Slice")
@tf_op("StridedSlice")
class StridedSlice(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[1] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[1], node.op_type)
    if node.inputs[2] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[2], node.op_type)
    if node.inputs[3] not in kwargs["consts"]:
      exception.CONST_NOT_FOUND_EXCEPT(node.inputs[3], node.op_type)
    consts = kwargs["consts"]
    strides = consts[node.inputs[3]]
    input_num = len(node.inputs)
    if input_num  > 4:
        exception.OP_UNSUPPORTED_EXCEPT("input_num > 4", node.op_type)
    for stride in strides:
        if stride != 1:
            exception.OP_UNSUPPORTED_EXCEPT("stride = {}".format(stride), node.op_type)

  @classmethod
  def version_1(cls, node, **kwargs):
    consts = kwargs["consts"]
    node_dict = kwargs["node_dict"]

    # convert NHWC to NCHW
    # TODO(wwcai): if data_fmt
    old_starts = consts[node.inputs[1]]
    old_ends = consts[node.inputs[2]]
    if old_starts[0] == 1 and old_ends[0] == 2:
        starts = [2]
        ends = [3]
    if old_starts[0] == 2 and old_ends[0] == 3:
        starts = [3]
        ends = [4]

    return cls.make_node_from_tf_node(
        node, [node.inputs[0]],
        starts=consts[node.inputs[1]],
        ends=consts[node.inputs[2]],
        axes=list(range(len(node_dict[node.inputs[0]].attr["_output_shapes"]))))
