# Copyright (c) 2022 Intel Corporation
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
"""Functional tests for QuantizedDepthwiseConv2D operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import os

os.environ["ITEX_ONEDNN_GRAPH"] = "1"


@test_util.run_all_in_native_and_block_format
class DepthwiseConvINT8Test(test.TestCase):

  @test_util.run_deprecated_v1
  def testDepthwiseConv2DINT8(self):
    x_f32_np = np.random.uniform(low=-3.0, high=3.0, size=(1, 4, 4, 3)).astype(np.float32)
    y_f32_np = np.random.uniform(low=-3.0, high=3.0, size=(3, 3, 3, 1)).astype(np.float32)
    with ops.name_scope("test"):
      x_f32 = constant_op.constant(x_f32_np)
      x_min = constant_op.constant([-3.0], shape=[]) 
      x_max = constant_op.constant([3.0], shape=[]) 
      x_int8, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max,
                                              T=dtypes.qint8,
                                              mode="SCALED",
                                              round_mode="HALF_TO_EVEN",
                                              narrow_range=True)
      x_deq = array_ops.dequantize(x_int8, x_min, x_max, mode="SCALED")
                    
      y_f32 = constant_op.constant(y_f32_np)
      y_min = constant_op.constant([-3.0] * 3) 
      y_max = constant_op.constant([3.0] * 3) 
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max,
                                              T=dtypes.qint8,
                                              mode="SCALED",
                                              round_mode="HALF_TO_EVEN",
                                              narrow_range=True, axis=2)
      y_deq = array_ops.dequantize(y_int8, y_min, y_max, mode="SCALED", axis=2)
                    

      output = load_ops_library.DepthwiseConv2dNative(input=x_deq, filter=y_deq, strides=[1, 1, 1, 1], \
                                                      padding="SAME", explicit_paddings=[], data_format="NHWC", \
                                                      dilations=[1, 1, 1, 1], name="DepthwiseConv2dNative")
      output = array_ops.identity(output)
       
      res = self.evaluate(output)
      print(res)


if __name__ == "__main__":
  test.main()
