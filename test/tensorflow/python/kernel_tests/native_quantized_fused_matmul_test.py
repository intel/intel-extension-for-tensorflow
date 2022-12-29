# Copyright (C) 2022 Intel Corporation
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
# http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#  
# 
# SPDX-License-Identifier: Apache-2.0
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

os.environ["ITEX_ENABLE_ONEDNN_LAYOUT_OPT"] = "0"
os.environ["ITEX_NATIVE_FORMAT"] = "1"

# TODO(itex): Turn on this UT, once onednn fix gpu accuracy issue

class QuantizedFusedMatMul(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedFusedMatMul, self).__init__(method_name)

  #  matmul + bias + gelu per tensor test
  @test_util.run_deprecated_v1
  def _testGeluPerTensorPostop(self, api_mode):

    with ops.name_scope("test"):
      x_f32 = constant_op.constant([[1.51, 1.13, -0.66],
                                    [0.79, 0.98, 1.88]])
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.quint8, mode="MIN_FIRST", narrow_range=False)

      y_f32 = constant_op.constant([[-1.71, 3.05, 2.05, 0.64],
                                    [ 0.73, 0.23, 1.42, 3.57],
                                    [-0.61, 2.24, 2.69, 2.90]])
      y_min = tf.math.reduce_min(y_f32)
      y_max = tf.math.reduce_max(y_f32)
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      bias_f32 = constant_op.constant([-1.42, 2.05, -1.00, 0.94], dtype=dtypes.float32)

      if api_mode == "V1":
        matmul_int8 = load_ops_library._QuantizedFusedMatMulAndDequantize(a=x_int8, b=y_int8, args=[bias_f32], min_a=x_min, max_a=x_max,
                                                                          min_b=y_min, max_b=y_max,
                                                                          Toutput=tf.float32, fused_ops=["BiasAdd", "GeluApproximate"],
                                                                          input_quant_mode="MIN_FIRST") 
        matmul_int8 = array_ops.identity(matmul_int8)

      elif api_mode == "V2":
        matmul_int8 = load_ops_library._QuantizedMatMul(device_inputs=[x_int8, y_int8, bias_f32], 
                                                                       host_inputs=[x_min, x_max, y_min, y_max],
                                                                       T1=dtypes.quint8, T2=dtypes.qint8,
                                                                       Tdevice_outputs=[dtypes.float32],
                                                                       fused_ops=["BiasAdd", "GeluApproximate", "Dequantize"],
                                                                       input_quant_mode="MIN_FIRST") 
        matmul_int8 = array_ops.identity(matmul_int8[0])[0]
      
      matmul_f32 = itex.ops.gelu(tf.nn.bias_add(tf.linalg.matmul(x_f32, y_f32), bias_f32), approximate=True)
      matmul_f32 = array_ops.identity(matmul_f32)

      with self.session(use_gpu=True) as sess:
        # Run twice to check the funtionality of object cache
        for _ in range(2):
          matmul_int8_res = sess.run(matmul_int8)
          matmul_f32_res = sess.run(matmul_f32)

          # int8 test tolerate larger difference
          self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.2, atol=0.2)

  #  matmul + bias + gelu per channel test
  @test_util.run_deprecated_v1
  def _testGeluPerchannelPostop(self, api_mode):
    
    with ops.name_scope("test"):
      x_f32 = constant_op.constant([[1.51, 1.13, -0.66],
                                    [0.79, 0.98, 1.88]])
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.quint8, mode="MIN_FIRST", narrow_range=False)

      y_f32 = constant_op.constant([[-1.71, 3.05, 2.05, 0.64],
                                    [ 0.73, 0.23, 1.42, 3.57],
                                    [-0.61, 2.24, 2.69, 2.90]])
      y_min = tf.math.reduce_min(y_f32, axis=0)
      y_max = tf.math.reduce_max(y_f32, axis=0)
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=1)

      bias_f32 = constant_op.constant([-1.42, 2.05, -1.00, 0.94], dtype=dtypes.float32)
      if api_mode == "V1":
        matmul_int8 = load_ops_library._QuantizedFusedMatMulAndDequantize(a=x_int8, b=y_int8, args=[bias_f32], min_a=x_min, max_a=x_max,
                                                                          min_b=y_min, max_b=y_max,
                                                                          Toutput=tf.float32, fused_ops=["BiasAdd", "GeluApproximate"],
                                                                          input_quant_mode="MIN_FIRST") 
        matmul_int8 = array_ops.identity(matmul_int8)
      elif api_mode == "V2":
        matmul_int8 = load_ops_library._QuantizedMatMul(device_inputs=[x_int8, y_int8, bias_f32], 
                                                                       host_inputs=[x_min, x_max, y_min, y_max],
                                                                       T1=dtypes.quint8, T2=dtypes.qint8,
                                                                       Tdevice_outputs=[dtypes.float32],
                                                                       fused_ops=["BiasAdd", "GeluApproximate", "Dequantize"],
                                                                       input_quant_mode="MIN_FIRST") 
        matmul_int8 = array_ops.identity(matmul_int8[0])[0]
      
      
      matmul_f32 = itex.ops.gelu(tf.nn.bias_add(tf.linalg.matmul(x_f32, y_f32), bias_f32), approximate=True)
      matmul_f32 = array_ops.identity(matmul_f32)

      with self.session(use_gpu=True) as sess:
        # Run twice to check the funtionality of object cache
        for _ in range(2):
          matmul_int8_res = sess.run(matmul_int8)
          matmul_f32_res = sess.run(matmul_f32)

          # int8 test tolerate larger difference
          self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.2, atol=0.2)

  # matmul + bias + add per tensor
  @test_util.run_deprecated_v1
  def _testAddPostop(self, api_mode):

    with ops.name_scope("test"):
      x_f32 = constant_op.constant([[1.51, 1.13, -0.66],
                                    [0.79, 0.98, 1.88]])
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.quint8, mode="MIN_FIRST", narrow_range=False)
      
      y_f32 = constant_op.constant([[-1.71, 3.05, 2.05, 0.64],
                                    [ 0.73, 0.23, 1.42, 3.57],
                                    [-0.61, 2.24, 2.69, 2.90]])
      y_min = tf.math.reduce_min(y_f32)
      y_max = tf.math.reduce_max(y_f32)
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      z_f32 = constant_op.constant([[ 0.92, 1.43, 1.84,-1.01],
                                    [-1.83,-1.39,-0.50, 0.32]])
      
      bias_f32 = constant_op.constant([-1.42, 2.05, -1.00, 0.94], dtype=dtypes.float32)

      if api_mode == "V1":
        matmul_int8 = load_ops_library._QuantizedFusedMatMulAndDequantize(a=x_int8, b=y_int8, args=[bias_f32, z_f32], min_a=x_min, max_a=x_max,
                                                                          min_b=y_min, max_b=y_max,
                                                                          Toutput=tf.float32, fused_ops=["BiasAdd", "Add"],
                                                                          input_quant_mode="MIN_FIRST") 

        matmul_int8 = array_ops.identity(matmul_int8)

      elif api_mode == "V2":
        matmul_int8 = load_ops_library._QuantizedMatMul(device_inputs=[x_int8, y_int8, bias_f32, z_f32], 
                                                                       host_inputs=[x_min, x_max, y_min, y_max],
                                                                       T1=dtypes.quint8, T2=dtypes.qint8,
                                                                       Tdevice_outputs=[dtypes.float32],
                                                                       fused_ops=["BiasAdd", "Add", "Dequantize"],
                                                                       input_quant_mode="MIN_FIRST") 
        matmul_int8 = array_ops.identity(matmul_int8[0])[0]
      
      
      
      matmul_f32 = tf.math.add(tf.nn.bias_add(tf.linalg.matmul(x_f32, y_f32), bias_f32), z_f32)
      matmul_f32 = array_ops.identity(matmul_f32)

      with self.session(use_gpu=True) as sess:
        # Run twice to check the funtionality of object cache
        for _ in range(2):
          matmul_int8_res = sess.run(matmul_int8)
          matmul_f32_res = sess.run(matmul_f32)

          # int8 test tolerate larger difference
          self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.2, atol=0.2)

  def testMatMulINT8(self):
    self._testGeluPerTensorPostop("V1")
    self._testGeluPerTensorPostop("V2")
    self._testGeluPerchannelPostop("V1")
    self._testGeluPerchannelPostop("V2")
    self._testAddPostop("V1")
    self._testAddPostop("V2")

if __name__ == "__main__":
  test.main()
