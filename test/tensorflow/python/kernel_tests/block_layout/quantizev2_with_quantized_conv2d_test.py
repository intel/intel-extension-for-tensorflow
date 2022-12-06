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
"""Functional tests for _ITEXQuantizeV2WithQuantizedConv2D operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

import numpy as np
import os
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import tensorflow as tf

os.environ["ITEX_LAYOUT_OPT"] = "1"
os.environ["ITEX_NATIVE_FORMAT"] = "0"

class QuantizeV2WithQuantizedConv2DTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizeV2WithQuantizedConv2DTest, self).__init__(method_name)

  @test_util.run_deprecated_v1
  def test_ITEXQuantizeV2WithQuantizedConv2DOp(self):
    # Only XPU kernel is registered for _ITEXQuantizeV2WithQuantizedConv2D
    if not test.is_gpu_available():
      return
    with ops.name_scope("test"):
      x_f32_np = np.random.uniform(low=-3.0, high=3.0, size=(1, 6, 6, 4)).astype(np.float32)
      x_f32 = constant_op.constant(x_f32_np)

      x_min = tf.math.reduce_min(x_f32)
      x_max = tf.math.reduce_max(x_f32)

      y_f32_np = np.random.uniform(low=-3.0, high=3.0, size=(3, 3, 4, 4)).astype(np.float32)
      y_f32 = constant_op.constant(y_f32_np)

      y_min = tf.math.reduce_min(y_f32, axis=(0, 1, 2))
      y_max = tf.math.reduce_max(y_f32, axis=(0, 1, 2))
      y_int8, y_min, y_max = array_ops.quantize(y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=3)

      bias_f32_np = np.random.uniform(low=-1.0, high=1.0, size=(4)).astype(np.float32)
      bias_f32 = constant_op.constant(bias_f32_np)

      conv_f32 = tf.nn.bias_add(tf.nn.conv2d(x_f32, y_f32, [1,1,1,1], padding="SAME"), bias_f32)
      # The output of _ITEXQuantizeV2WithQuantizedConv2D is UINT8, so we should set negative value to 0
      conv_f32 = tf.nn.relu(conv_f32) 

      z_freezed_min = tf.math.reduce_min(conv_f32)
      z_freezed_max = tf.math.reduce_max(conv_f32)

      conv_int8_req, conv_freezed_min, conv_freezed_max = load_ops_library._ITEXQuantizeV2WithQuantizedConv2D(
        input=x_f32, filter=y_int8, bias=bias_f32, 
        min_input=x_min, max_input=x_max, min_filter=y_min, max_filter=y_max, 
        min_freezed_output=z_freezed_min, max_freezed_output=z_freezed_max,
        strides=[1, 1, 1, 1], padding="SAME", out_type=dtypes.quint8)  
    
      conv_int8 = array_ops.dequantize(conv_int8_req, conv_freezed_min, conv_freezed_max, mode="SCALED", narrow_range=True)

      conv_f32 = array_ops.identity(conv_f32)
      conv_int8 = array_ops.identity(conv_int8)
      
      conv_int8_res = self.evaluate(conv_int8)
      conv_f32_res = self.evaluate(conv_f32)
      
      self.assertAllClose(conv_int8_res, conv_f32_res, rtol=0.2, atol=0.2)



if __name__ == "__main__":
  test.main()
