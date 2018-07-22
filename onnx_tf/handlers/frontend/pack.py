from onnx_tf.common import exception
from onnx_tf.handlers.frontend_handler import FrontendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_op


@onnx_op("Stack")
@tf_op("Pack")
class Pack(FrontendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    input_num = len(node.inputs)
    if input_num < 2:
        exception.OP_UNSUPPORTED_EXCEPT("input_num < 2", node.op_type)

  @classmethod
  def version_4(cls, node, **kwargs):
    axis = node.attr.get("axis", 0)
    return cls.make_node_from_tf_node(node, node.inputs, axis=axis)
