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

"""Functional tests for quantized matmul operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import tensorflow as tf

@test_util.run_all_in_native_and_block_format
class QuantizedMatMul(test.TestCase):
    
    def __init__(self, method_name="runTest"):
        super(QuantizedMatMul, self).__init__(method_name)
        
    @test_util.run_deprecated_v1
    def testQuantizedMatMulWithBiasAndReluAndRequantize(self):
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

            matmul_f32 = tf.nn.relu(tf.nn.bias_add(tf.linalg.matmul(x_f32, y_f32), bias_f32))
            z_freezed_min = tf.math.reduce_min(matmul_f32)
            z_freezed_max = tf.math.reduce_max(matmul_f32)

            
            matmul_int8, matmul_min, matmul_max = tf.raw_ops.QuantizedMatMulWithBiasAndReluAndRequantize(
                a=x_int8,
                b=y_int8,
                bias=bias_f32,
                min_a=x_min,
                max_a=x_max,
                min_b=y_min,
                max_b=y_max,
                min_freezed_output=z_freezed_min,
                max_freezed_output=z_freezed_max,
                Toutput=tf.dtypes.quint8,
                input_quant_mode='MIN_FIRST',
            )
            

            matmul_int8_deq = array_ops.dequantize(matmul_int8, matmul_min, matmul_max, mode="SCALED", narrow_range=True)
            matmul_int8_deq = array_ops.identity(matmul_int8_deq)
            matmul_f32 = array_ops.identity(matmul_f32)

            matmul_int8_res = self.evaluate(matmul_int8_deq)
            matmul_f32_res = self.evaluate(matmul_f32)

            # int8 test tolerate larger difference
            self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.2, atol=0.2)
            
    @test_util.run_deprecated_v1
    def testQuantizedMatMulWithBiasAndRelu(self):
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

            matmul_f32 = tf.nn.relu(tf.nn.bias_add(tf.linalg.matmul(x_f32, y_f32), bias_f32))
            matmul_f32_min = tf.math.reduce_min(matmul_f32)
            matmul_f32_max = tf.math.reduce_max(matmul_f32)
            matmul_int32, matmul_int32_min, matmul_int32_max = tf.raw_ops.QuantizedMatMulWithBiasAndRelu(
                a=x_int8,
                b=y_int8,
                bias=bias_f32,
                min_a=x_min,
                max_a=x_max,
                min_b=y_min,
                max_b=y_max,
                Toutput=tf.dtypes.qint32,
                transpose_a=False,
                transpose_b=False,
                input_quant_mode='MIN_FIRST',
                name=None
            )

            matmul_int8_req, matmul_int8_req_min, matmul_int8_req_max = tf.raw_ops.Requantize(input=matmul_int32, input_min=matmul_int32_min, input_max=matmul_int32_max,
                                                                                                requested_output_min=matmul_f32_min, requested_output_max=matmul_f32_max,
                                                                                                out_type=dtypes.qint8)
            matmul_int8_deq = array_ops.dequantize(matmul_int8_req, matmul_int8_req_min, matmul_int8_req_max, mode="MIN_FIRST", narrow_range=False)
            matmul_int8_deq = array_ops.identity(matmul_int8_deq)
            matmul_f32 = array_ops.identity(matmul_f32)

            matmul_int8_res = self.evaluate(matmul_int8_deq)
            matmul_f32_res = self.evaluate(matmul_f32)

            # int8 test tolerate larger difference
            self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.2, atol=0.2)
    
    @test_util.run_deprecated_v1
    def testQuantizedMatMulWithBias(self):
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

            matmul_f32 = tf.nn.bias_add(tf.linalg.matmul(x_f32, y_f32), bias_f32)
            matmul_f32_min = tf.math.reduce_min(matmul_f32)
            matmul_f32_max = tf.math.reduce_max(matmul_f32)

            matmul_int32, matmul_int32_min, matmul_int32_max = tf.raw_ops.QuantizedMatMulWithBias(
                a=x_int8,
                b=y_int8,
                bias=bias_f32,
                min_a=x_min,
                max_a=x_max,
                min_b=y_min,
                max_b=y_max,
                Toutput=tf.dtypes.qint32,
                transpose_a=False,
                transpose_b=False,
                input_quant_mode='MIN_FIRST',
                name=None
            )
           
            matmul_int8_req, matmul_int8_req_min, matmul_int8_req_max = tf.raw_ops.Requantize(input=matmul_int32, input_min=matmul_int32_min, input_max=matmul_int32_max,
                                                                                                requested_output_min=matmul_f32_min, requested_output_max=matmul_f32_max,
                                                                                                out_type=dtypes.qint8)
            matmul_int8_deq = array_ops.dequantize(matmul_int8_req, matmul_int8_req_min, matmul_int8_req_max, mode="MIN_FIRST", narrow_range=False)
            matmul_int8_deq = array_ops.identity(matmul_int8_deq)
            matmul_f32 = array_ops.identity(matmul_f32)

            matmul_int8_res = self.evaluate(matmul_int8_deq)
            matmul_f32_res = self.evaluate(matmul_f32)
            
            # int8 test tolerate larger difference
            self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.2, atol=0.2)

        
    @test_util.run_deprecated_v1
    def testQuantizedMatMulWithBiasAndDequantize(self):
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
            matmul_f32 = tf.nn.bias_add(tf.linalg.matmul(x_f32, y_f32), bias_f32)
            z_freezed_min = tf.math.reduce_min(matmul_f32)
            z_freezed_max = tf.math.reduce_max(matmul_f32)
    
            matmul_int8 = tf.raw_ops.QuantizedMatMulWithBiasAndDequantize(
                a=x_int8,
                b=y_int8,
                bias=bias_f32,
                min_a=x_min,
                max_a=x_max,
                min_b=y_min,
                max_b=y_max,
                min_freezed_output=z_freezed_min,
                max_freezed_output=z_freezed_max,
                Toutput=tf.float32,
                transpose_a=False,
                transpose_b=False,
                input_quant_mode='MIN_FIRST',
                name=None
            )
           
            matmul_int8 = array_ops.identity(matmul_int8)
            matmul_f32 = array_ops.identity(matmul_f32)

            matmul_int8_res = self.evaluate(matmul_int8)
            matmul_f32_res = self.evaluate(matmul_f32)

            # int8 test tolerate larger difference
            self.assertAllClose(matmul_int8_res, matmul_f32_res, rtol=0.2, atol=0.2)
            
            
    
if __name__ == "__main__":
  test.main()
