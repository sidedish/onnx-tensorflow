from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Split")
@tf_op("Split")
class Split(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    if node.inputs[0] not in kwargs["consts"]: #axis or split_dim
        exception.CONST_NOT_FOUND_EXCEPT(node.inputs[0], node.op_type)

  @classmethod
  def version_2(cls, node, **kwargs):
    consts = kwargs["consts"]
    axis = int(consts[node.inputs[0]])
    # FIXME(wwcai): NHWC->NCHW
    if axis == 3:
        axis = 1
    elif axis == 1 or axis == 2:
        axis = axis + 1

    return cls.make_node_from_tf_node(
        node, [node.inputs[1]],
        cls.get_outputs_names(node, num=node.attr["num_split"]),
        axis=axis)
