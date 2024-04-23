# TODO(itex): Turn on this UT, once fix bugs

# """Functional tests for quantized operations."""

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

os.environ["ITEX_LAYOUT_OPT"] = "0"
os.environ["ITEX_NATIVE_FORMAT"] = "1"

class QuantizedPoolingTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedPoolingTest, self).__init__(method_name)
    self.TestCaseArgs = [
      #seed,     shape, min, max,      ksize,   strides, padding,rtol, atol
      [   1, [2,3,3,3],  -100, 100, [1,1,1,1], [1,1,1,1], "VALID", 0.2, 0.2],
      [   2, [2,3,3,3],    0, 200, [1,1,1,1], [1,2,2,1], "VALID", 0.2, 0.2],
      [   3, [2,3,3,3],  -50,  50, [1,2,2,1], [1,1,1,1], "VALID", 0.2, 0.2],
      [   4, [2,2,2,3],  -50,   0, [1,2,2,1], [1,2,2,1], "SAME",  0.2, 0.2],
      [   5, [2,2,4,3],    0, 100, [1,2,2,1], [1,1,1,1], "SAME",  0.2, 0.2],
      [   6, [2,6,6,3], -100,   0, [1,2,2,1], [1,1,1,1], "SAME",  0.2, 0.2],
    ]
  
  @test_util.run_deprecated_v1
  def testQuantizedAvgPool_qint8(self):
    with ops.name_scope("test"):
      for testcase_args in self.TestCaseArgs:
        self._Verify(nn_ops.avg_pool, 
                     load_ops_library.QuantizedAvgPool, 
                     dtypes.qint8, 
                     testcase_args[0], 
                     testcase_args[1], 
                     testcase_args[2], 
                     testcase_args[3],
                     testcase_args[4],
                     testcase_args[5],
                     testcase_args[6],
                     testcase_args[7],
                     testcase_args[8]
        )

      for testcase_args in self.TestCaseArgs:
        self._Verify(nn_ops.avg_pool, 
                     load_ops_library.ITEXQuantizedAvgPool, 
                     dtypes.qint8, 
                     testcase_args[0], 
                     testcase_args[1], 
                     testcase_args[2], 
                     testcase_args[3],
                     testcase_args[4],
                     testcase_args[5],
                     testcase_args[6],
                     testcase_args[7],
                     testcase_args[8]
        )

  # TODO(itex): turn on this flag, once GPU asymmetric quantization is supported
  # @test_util.run_deprecated_v1
  # def testQuantizedAvgPool_quint8(self):
  #   with ops.name_scope("test"):
  #     for testcase_args in self.TestCaseArgs:
  #       self._Verify(nn_ops.avg_pool, 
  #                    load_ops_library.QuantizedAvgPool, 
  #                    dtypes.quint8, 
  #                    testcase_args[0], 
  #                    testcase_args[1], 
  #                    testcase_args[2], 
  #                    testcase_args[3],
  #                    testcase_args[4],
  #                    testcase_args[5],
  #                    testcase_args[6],
  #                    testcase_args[7],
  #                    testcase_args[8]
  #       )
  
  @test_util.run_deprecated_v1
  def testQuantizedMaxPool_qint8(self):
    with ops.name_scope("test"):
      for testcase_args in self.TestCaseArgs:
        self._Verify(nn_ops.max_pool, 
                     load_ops_library.QuantizedMaxPool, 
                     dtypes.qint8, 
                     testcase_args[0], 
                     testcase_args[1], 
                     testcase_args[2], 
                     testcase_args[3],
                     testcase_args[4],
                     testcase_args[5],
                     testcase_args[6],
                     testcase_args[7],
                     testcase_args[8]
        )

  # TODO(itex): turn on this flag, once GPU asymmetric quantization is supported
  # @test_util.run_deprecated_v1
  # def testQuantizedMaxPool_quint8(self):
  #   with ops.name_scope("test"):
  #     for testcase_args in self.TestCaseArgs:
  #       self._Verify(nn_ops.max_pool, 
  #                    load_ops_library.QuantizedMaxPool, 
  #                    dtypes.quint8, 
  #                    testcase_args[0], 
  #                    testcase_args[1], 
  #                    testcase_args[2], 
  #                    testcase_args[3],
  #                    testcase_args[4],
  #                    testcase_args[5],
  #                    testcase_args[6],
  #                    testcase_args[7],
  #                    testcase_args[8]
  #       )

  def _CreateInputData(self, input_shape, seed=1, min_value=-100, max_value=100):
    """Build tensor data spreading the range [min_value, max_value)."""
    np.random.seed(seed)
    value = (max_value - min_value) * np.random.random_sample(input_shape) + min_value
    return np.dtype(np.float32).type(value) 

  def _Verify(self, ref_func, test_func, test_dtype, input_seed, input_shape, min, max, ksize, strides, padding, rtol, atol):
    input_values = self._CreateInputData(input_shape, input_seed, min, max)

    input_min = tf.math.reduce_min(input_values)
    input_max = tf.math.reduce_max(input_values)

    if test_dtype == dtypes.qint8:
      q_int8, q_min, q_max = array_ops.quantize(input_values, input_min, input_max, T=test_dtype, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)
    elif test_dtype == dtypes.quint8:
      q_int8, q_min, q_max = array_ops.quantize(input_values, input_min, input_max, T=test_dtype, mode="MIN_FIRST", narrow_range=True)
    qo_int8, qo_min, qo_max = test_func(input=q_int8, min_input=q_min, max_input=q_max, ksize=ksize, strides=strides, padding=padding)

    if test_dtype == dtypes.qint8:
      output_int8 = array_ops.dequantize(qo_int8, qo_min, qo_max, mode="SCALED", narrow_range=True)
    elif test_dtype == dtypes.quint8:
      output_int8 = array_ops.dequantize(qo_int8, qo_min, qo_max, mode="MIN_FIRST", narrow_range=True)

    output_f32 = ref_func(input_values, ksize=ksize, strides=strides, padding=padding)

    output_int8_res = self.evaluate(output_int8)
    output_f32_res = self.evaluate(output_f32)

    self.assertAllClose(output_int8_res, output_f32_res, rtol=rtol, atol=atol)

if __name__ == "__main__":
  test.main()
