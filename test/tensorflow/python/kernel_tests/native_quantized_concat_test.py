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

import tensorflow as tf

os.environ["ITEX_ENABLE_ONEDNN_LAYOUT_OPT"] = "0"
os.environ["ITEX_NATIVE_FORMAT"] = "1"

class QuantizedConcat(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedConcat, self).__init__(method_name)

  @test_util.run_deprecated_v1
  def testFull(self):
    if not test_lib.is_gpu_available():
      self.skipTest("Skip on CPU due to PreCI reports the issue which cannot be reproduced locally")

    test_shape = [[1,2], [3,4], [1,2,3], [2,2,3], [2,3,4], [3,4,5], [2,2,2,3], [2,2,2,2,3]]
    test_start = [[0, 10], [20, 60], [-10, 10]]
    test_range = [[0, 0],[-5, 5]]

    for shape in test_shape:
      for start in test_start:
        for qrange in test_range:
          for axis in range(0, len(shape)):
            x_np = np.arange(start[0], start[0] + np.prod(shape), dtype='f4').reshape(shape)
            y_np = np.arange(start[1], start[1] + np.prod(shape), dtype='f4').reshape(shape)
            z_np = np.concatenate([x_np, y_np], axis=axis)
            min = np.min(z_np) + qrange[0]
            max = np.max(z_np) + qrange[1]
            self.ExecuteConcatAndVerify(x_np, y_np, z_np, axis, min, max, dtypes.qint8)
            if (min >= 0):
              self.ExecuteConcatAndVerify(x_np, y_np, z_np, axis, min, max, dtypes.quint8)

  def ExecuteConcatAndVerify(self, x_np, y_np, z_np, axis, min, max, dtype):
    with ops.name_scope("test"):
      x_f32 = constant_op.constant(x_np)
      y_f32 = constant_op.constant(y_np)
      z_f32 = constant_op.constant(z_np)

      x_int8, x_min, x_max = array_ops.quantize(x_f32, min, max, T=dtype, mode="SCALED")
      y_int8, y_min, y_max = array_ops.quantize(y_f32, min, max, T=dtype, mode="SCALED")
      z_int8, z_min, z_max = load_ops_library.QuantizedConcatV2(values=[x_int8, y_int8],axis=axis,input_mins=[x_min, y_min], input_maxes=[x_max, y_max]) 
      result_int8 = array_ops.dequantize(z_int8, z_min, z_max, mode="SCALED")

      tol = (max - min + 1) / 127
      self.assertAllClose(result_int8, z_f32, rtol=tol, atol=tol, msg="Concate {0} with {1} at axis:{3} to {2}. range({4}, {5})".format(x_np.shape, y_np.shape, z_np.shape, axis, min, max))



if __name__ == "__main__":
  test.main()
