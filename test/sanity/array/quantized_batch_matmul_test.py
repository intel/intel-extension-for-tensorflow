# Copyright (c) 2022 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



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

@test_util.run_all_in_native_and_block_format
class QuantizedBatchMatMul(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedBatchMatMul, self).__init__(method_name)
  
  # single BMM INT8 test
  @test_util.run_deprecated_v1
  def _testQuantizeBMM(self, api_mode):
    with ops.name_scope("test"):
      x_f32 = constant_op.constant([[[-0.33,  1.33,  1.44],
                                     [-0.23,  0.60, -0.08]],
                                    [[ 0.97,  0.70,  0.90],
                                     [-0.56, -1.84, -1.27]]])
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      y_f32 = constant_op.constant([[[ 0.14,  0.78],
                                     [ 1.51, -1.36],
                                     [ 0.56, -0.29]],

                                    [[ 1.05,  1.82],
                                     [ 1.72,  0.36],
                                     [ 0.16, -0.10]]])
      y_min = tf.math.reduce_min(y_f32)
      y_max = tf.math.reduce_max(y_f32)
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)
      
      if api_mode == "V1":
        matmul_int8 = load_ops_library._QuantizedBatchMatMulV2AndDequantize(x=x_int8, y=y_int8, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max)
        matmul_int8 = array_ops.identity(matmul_int8)
      elif api_mode == "V2":
        matmul_int8 = load_ops_library._QuantizedBatchMatMul(device_inputs=[x_int8, y_int8], 
                                                             host_inputs=[x_min, x_max, y_min, y_max],
                                                             T1=dtypes.qint8, T2=dtypes.qint8,
                                                             fused_ops=["Dequantize"],
                                                             Tdevice_outputs=[dtypes.float32])
                                                                       
        matmul_int8 = array_ops.identity(matmul_int8[0])[0]
      
      
      matmul_f32 = tf.linalg.matmul(x_f32, y_f32)
      matmul_f32 = array_ops.identity(matmul_f32)

      matmul_int8_res = self.evaluate(matmul_int8)
      matmul_f32_res = self.evaluate(matmul_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.3, atol=0.3)

  # single BMM + Mul fusion INT8 test
  @test_util.run_deprecated_v1
  def _testQuantizeBMMMul(self, api_mode):
    with ops.name_scope("test"):
      x_f32 = constant_op.constant([[[-0.33,  1.33,  1.44],
                                     [-0.23,  0.60, -0.08]],
                                    [[ 0.97,  0.70,  0.90],
                                     [-0.56, -1.84, -1.27]]])
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      y_f32 = constant_op.constant([[[ 0.14,  0.78],
                                     [ 1.51, -1.36],
                                     [ 0.56, -0.29]],

                                    [[ 1.05,  1.82],
                                     [ 1.72,  0.36],
                                     [ 0.16, -0.10]]])
      y_min = tf.math.reduce_min(y_f32)
      y_max = tf.math.reduce_max(y_f32)
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      mul_tensor = constant_op.constant(0.5)

      if api_mode == "V1":
        matmul_int8 = load_ops_library._QuantizedFusedBatchMatMulV2AndDequantize(x=x_int8, y=y_int8, min_x=x_min, max_x=x_max, min_y=y_min, max_y=y_max, 
                                                                                args=[mul_tensor], fused_ops=["Mul"])
        matmul_int8 = array_ops.identity(matmul_int8)
      elif api_mode == "V2":
        matmul_int8 = load_ops_library._QuantizedBatchMatMul(device_inputs=[x_int8, y_int8, mul_tensor], 
                                                             host_inputs=[x_min, x_max, y_min, y_max],
                                                             T1=dtypes.qint8, T2=dtypes.qint8,
                                                             fused_ops=["Mul", "Dequantize"],
                                                             Tdevice_outputs=[dtypes.float32])
                                                                       
        matmul_int8 = array_ops.identity(matmul_int8[0])[0]
      
      
      matmul_f32 = tf.math.multiply(tf.linalg.matmul(x_f32, y_f32), mul_tensor)
      matmul_f32 = array_ops.identity(matmul_f32)

      matmul_int8_res = self.evaluate(matmul_int8)
      matmul_f32_res = self.evaluate(matmul_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.3, atol=0.3)

  @test_util.run_deprecated_v1
  def testMatMulINT8(self):
    self._testQuantizeBMM("V1")
    self._testQuantizeBMM("V2")
    self._testQuantizeBMMMul("V1")
    self._testQuantizeBMMMul("V2")


if __name__ == "__main__":
  test.main()
