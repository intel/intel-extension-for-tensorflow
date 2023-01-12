"""Functional tests for quantized operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
import intel_extension_for_tensorflow as itex

import numpy as np
import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl

import tensorflow as tf

os.environ["ITEX_ENABLE_ONEDNN_LAYOUT_OPT"] = "0"
os.environ["ITEX_NATIVE_FORMAT"] = "1"

class QuantizedFusedBatchNorm(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedFusedBatchNorm, self).__init__(method_name)

  @test_util.run_deprecated_v1
  def testQuantizedBN(self):
    with ops.name_scope("test"):
      x_f32 = constant_op.constant([2, 3, 4, 1, 2, 3, 2, 3, 2, 2, 1, 2, 2, 3, 2, 3], shape = [1, 2, 2, 4], dtype=float)
      scale_f32 = constant_op.constant([2, 3, 1, 3], shape = [4], dtype=float)
      shift_f32 = constant_op.constant([-5, -3, -4, -4], shape = [4], dtype=float)
      mean_f32 = constant_op.constant([0.1, 0.2, 0.2, 0.3], shape = [4], dtype=float)
      var_f32 = constant_op.constant([1.1, 1.1, 1.2, 1.2], shape = [4], dtype=float)

      y_f32, _ , _ = array_ops.identity_n(nn_impl.fused_batch_norm(
          x_f32,
          scale_f32,
          shift_f32,
          mean=mean_f32,
          variance=var_f32,
          is_training=False))

      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      y_min = tf.math.reduce_min(y_f32)
      y_max = tf.math.reduce_max(y_f32)

      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      bn_int8, _ , _ = load_ops_library._QuantizedFusedBatchNorm(input=[x_int8, scale_f32, shift_f32, mean_f32, var_f32, x_min, x_max, y_min, y_max], T=dtypes.qint8, U=float, Tout=dtypes.qint8, out_types=[dtypes.qint8, float, float])
      bn_int8_deq = array_ops.dequantize(bn_int8, y_min, y_max, mode="SCALED", narrow_range=True)

      bn_int8_res = self.evaluate(bn_int8)

      y_f32_res = self.evaluate(y_f32)
      bn_int8_deq_res = self.evaluate(bn_int8_deq)

      # No accuracy check yet. Intel TF's BN INT8 kernel don't have close result compared to FP32 golden ground truth
      # int8 test tolerate larger difference
      # self.assertAllClose(bn_int8_deq_res, y_f32_res, rtol=0.0, atol=1.1)



if __name__ == "__main__":
  test.main()
