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
class QuantizedReshape(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedReshape, self).__init__(method_name)

  @test_util.run_deprecated_v1
  def testReshapeINT8(self):
    with ops.name_scope("test"):
      x_f32 = constant_op.constant([[1.51, 1.13, -0.66],
                                    [0.79, 0.98, 1.88]])
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      x_reshape_int8, x_reshape_min, x_reshape_max = load_ops_library.quantized_reshape(tensor=x_int8, shape=[3, 2], input_min=x_min, input_max=x_max)
      x_reshape_int8 = array_ops.dequantize(x_reshape_int8, x_reshape_min, x_reshape_max, mode="SCALED", narrow_range=True)
      reshape_int8 = array_ops.identity(x_reshape_int8)
      
      
      reshape_f32 = tf.reshape(x_f32, [3, 2])
      reshape_f32 = array_ops.identity(reshape_f32)

      reshape_int8_res = self.evaluate(reshape_int8)
      reshape_f32_res = self.evaluate(reshape_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(reshape_int8_res, reshape_f32_res, rtol=0.2, atol=0.2)

  @test_util.run_deprecated_v1
  def testTransposeINT8(self):
    with ops.name_scope("test"):
      x_f32 = constant_op.constant([[1.51, 1.13, -0.66],
                                    [0.79, 0.98, 1.88]])
      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

      x_tran_int8, x_tran_min, x_tran_max = load_ops_library._QuantizedTranspose(x=x_int8, perm=[1, 0], min_x=x_min, max_x=x_max)
      x_tran_int8 = array_ops.dequantize(x_tran_int8, x_tran_min, x_tran_max, mode="SCALED", narrow_range=True)
      tran_int8 = array_ops.identity(x_tran_int8)
      
      
      tran_f32 = tf.transpose(x_f32, [1, 0])
      tran_f32 = array_ops.identity(tran_f32)

      tran_int8_res = self.evaluate(tran_int8)
      tran_f32_res = self.evaluate(tran_f32)

      # int8 test tolerate larger difference
      self.assertAllClose(tran_int8_res, tran_f32_res, rtol=0.2, atol=0.2)

if __name__ == "__main__":
  test.main()
