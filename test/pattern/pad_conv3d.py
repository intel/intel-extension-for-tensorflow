import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
try:
    from intel_extension_for_tensorflow.python.test_func import test as test_lib
except ImportError:
    from tensorflow.python.platform import test as test_lib
import numpy as np
import time


@test_util.run_all_in_native_and_block_format
@test_util.run_all_in_graph_and_eager_modes
class FusedConv3DTest(test_lib.TestCase):
  @test_util.run_deprecated_v1
  def testFusePadConv3d(self):
    if test_lib.is_gpu_available():
      self.skipTest("Skip on GPU due to the pattern not supported")
    tf.compat.v1.disable_eager_execution()
    x = constant_op.constant(np.random.rand(1, 5, 8, 7, 1),
        dtype=dtypes.float32)
    w = constant_op.constant(np.random.rand(1, 2, 3, 1, 1),
        dtype=dtypes.float32)
    pad_value = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
    p = constant_op.constant(pad_value, dtype=dtypes.int32)

    x_pad = array_ops.pad(x, p)
    x_min = constant_op.constant(0, dtype=dtypes.float32)
    x_max = constant_op.constant(1, dtype=dtypes.float32)
    x_int8, x_min, x_max = array_ops.quantize(
      x_pad, x_min, x_max, T=dtypes.quint8, mode="SCALED",
      round_mode="HALF_TO_EVEN", narrow_range=True)
    x_pad_fp = array_ops.dequantize(x_int8, x_min, x_max,
      mode="SCALED")

    conv3d = nn_ops.Conv3D(
      input=x_pad_fp, filter=w, strides=[1, 1, 1, 1, 1],
      padding='SAME', data_format='NDHWC')
    fused = array_ops.identity(conv3d)
    fused = array_ops.identity(fused)
    fused = array_ops.identity(fused)

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    # fused pattern output value from gpu side
    # with self.session(use_gpu=False) as sess:
    with self.session() as sess:
      start_time = time.time()
      ret = sess.run(fused, options=run_options,
                 run_metadata=metadata)
      duration = time.time() - start_time
      print("end to end duration is : {}".format(duration))

      # Graph should contain fused op.
      graph = metadata.partition_graphs[0]
      found_fused_op = False
      pad_exist = False
      pad_val_flat = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
      for node in graph.node:
        if node.op in ('_ITEXConv3D'):
          if node.attr['padding'].s == b"EXPLICIT" and \
              node.attr['explicit_paddings'].list.i == pad_val_flat:
            found_fused_op = True

        if node.op in ('Pad'):
          pad_exist = True
      self.assertTrue((found_fused_op and not pad_exist),
              "this pattern has fusion issue!!")

    # reference value which is no fusion
    with self.session(use_gpu=True) as sess:
      ret_ref = sess.run(fused, options=run_options,
                 run_metadata=metadata)

    self.assertAllClose(ret_ref, ret)


if __name__ == '__main__':
  test.main()
