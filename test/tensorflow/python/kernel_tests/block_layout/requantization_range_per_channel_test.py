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
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

os.environ["ITEX_LAYOUT_OPT"] = "1"
os.environ["ITEX_NATIVE_FORMAT"] = "0"

class RequantizationRangePerChannelTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testRequantizationRangePerChannel(self):
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

      clip_value_max = float(6)
      x_min_res = tf.math.reduce_min(x_min, axis=(0,))
      x_max_res = tf.math.reduce_max(x_max, axis=(0,))

      x_min_res = tf.math.maximum(x_min_res, -clip_value_max)
      x_max_res = tf.math.minimum(x_max_res, clip_value_max)

      output_min, output_max = \
        load_ops_library.RequantizationRangePerChannel(input = x_qint32,
                                              input_min = x_min,
                                              input_max = x_max,
                                              clip_value_max = clip_value_max)

      output_min = array_ops.identity(output_min)
      output_max = array_ops.identity(output_max)

      x_min_res = array_ops.identity(x_min_res)
      x_max_res = array_ops.identity(x_max_res)

      output_min = self.evaluate(output_min)
      output_max = self.evaluate(output_max)

      x_min_res = self.evaluate(x_min_res)
      x_max_res = self.evaluate(x_max_res)

      self.assertAllClose(output_min, x_min_res, rtol=0.002, atol=0.002)
      self.assertAllClose(output_max, x_max_res, rtol=0.002, atol=0.002)


if __name__ == "__main__":
  test.main()
