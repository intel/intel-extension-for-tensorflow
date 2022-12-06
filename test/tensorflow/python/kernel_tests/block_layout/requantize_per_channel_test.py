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
"""Functional tests for RequantizePerChannel and RequantizationRangePerChannel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

os.environ["ITEX_LAYOUT_OPT"] = "1"
os.environ["ITEX_NATIVE_FORMAT"] = "0"

class RequantizePerChannelTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testRequantizePerChannel(self):
    # Only XPU kernel is registered for RequantizePerChannel
    if not test.is_gpu_available():
      return
    with ops.name_scope("test"):
      x_f32_np = np.random.uniform(low=-3.0, high=3.0,
                                  size=(1, 4, 4, 3)).astype(np.float32)
      x_f32 = constant_op.constant(x_f32_np)
      x_min = tf.math.reduce_min(x_f32, axis=(0, 1, 2))
      x_max = tf.math.reduce_max(x_f32, axis=(0, 1, 2))

      x_qint32, x_min, x_max = array_ops.quantize(x_f32, x_min, x_max,
                                              T=dtypes.qint32,
                                              mode="SCALED",
                                              round_mode="HALF_TO_EVEN",
                                              narrow_range=True,
                                              axis=3)

      range_op_output_min = float(-3.1415926)
      range_op_output_max = -range_op_output_min
      x_qint8_res = array_ops.dequantize(x_qint32, x_min, x_max,
                                        mode="SCALED",
                                        narrow_range=True, axis=3)
      x_qint8_res, unused_min, unused_max = \
        array_ops.quantize(x_qint8_res,
                          range_op_output_min, range_op_output_max,
                          T=dtypes.qint8,
                          mode="SCALED",
                          round_mode="HALF_TO_EVEN",
                          narrow_range=True)

      x_qint8, output_min, output_max = \
        math_ops.RequantizePerChannel(input = x_qint32,
                                    input_min = x_min, input_max = x_max,
                                    requested_output_min = range_op_output_min,
                                    requested_output_max = range_op_output_max,
                                    out_type=dtypes.qint8)

      x_qint8_res = array_ops.identity(x_qint8_res)
      x_qint8 = array_ops.identity(x_qint8)
      output_min = array_ops.identity(output_min)
      output_max = array_ops.identity(output_max)

      x_qint8_res = self.evaluate(x_qint8_res)
      x_qint8 = self.evaluate(x_qint8)
      output_min = self.evaluate(output_min)
      output_max = self.evaluate(output_max)

      self.assertAllClose(x_qint8, x_qint8_res, rtol=0.2, atol=0.2)
      self.assertAllClose(output_min, range_op_output_min, rtol=2e-3, atol=2e-3)
      self.assertAllClose(output_max, range_op_output_max, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
  test.main()
