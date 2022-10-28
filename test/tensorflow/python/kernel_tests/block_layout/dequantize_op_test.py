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
"""Tests for Dequantize Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

# TODO(itex): Test Deuantize op in eager mode (non-block-layout pass),
# when we support it

# TODO(itex): For intel-tf proper, the Quantize and Dequantize op only have narrow_range
# implementation, regardless the `narrow_range` flag is True or False. Currently, ITEX also 
# follow such logic.

@test_util.run_all_in_native_and_block_format
class DequantizeOpTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(DequantizeOpTest, self).__init__(method_name)

  @test_util.run_deprecated_v1
  def _testDequantizeOp(self, inputs, min_range, max_range, dtype,
                        mode="MIN_COMBINED", narrow_range=False):
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        input_op = constant_op.constant(inputs, shape=[len(inputs)], dtype=dtype)
        dequantized = array_ops.identity(array_ops.dequantize(input_op, min_range, max_range,
                                          mode=mode, narrow_range=narrow_range))
        tf_ans = self.evaluate(dequantized)

    # TODO(vrv): Add support for DT_QINT32 quantization if needed.
    type_dict = {
        dtypes.quint8: np.uint8,
        dtypes.qint8: np.int8,
        dtypes.quint16: np.uint16,
        dtypes.qint16: np.int16
    }
    self.assertIn(dtype, type_dict.keys())
    v_max = np.iinfo(type_dict[dtype]).max
    v_min = np.iinfo(type_dict[dtype]).min
    self.assertGreaterEqual(min_range, v_min)
    self.assertLessEqual(max_range, v_max)
    type_range = v_max - v_min

    if mode == "MIN_COMBINED":
      if v_min < 0:
        half_range = (type_range + 1) / 2
      else:
        half_range = 0.0
      np_ans = ((inputs.astype(np.float32) + half_range) *
                (max_range - min_range) / type_range) + min_range
    elif mode == "SCALED":
      if narrow_range:
        v_min += 1
      scale_factor = max(min_range / v_min, max_range / v_max)
      np_ans = inputs.astype(np.float32) * scale_factor

    self.assertAllClose(tf_ans, np_ans, rtol=1e-2, atol=1e-2)

  # MIN_COMBINED Quantized mode is not supported
  # @test_util.run_deprecated_v1
  # def testBasicQuint8(self):
  #   self._testDequantizeOp(np.array([0, 128, 255]), 0.0, 6.0, dtypes.quint8)
  #   self._testDequantizeOp(np.array([0, 128, 255]), 0.0, 123.456, dtypes.quint8)
  #   self._testDequantizeOp(
  #       np.array([0, 4, 42, 108, 243]), 5.0, 200.2, dtypes.quint8)

  # MIN_COMBINED Quantized mode is not supported
  # @test_util.run_deprecated_v1
  # def testBasicQint8(self):
  #   self._testDequantizeOp(np.array([-128, 0, 127]), -1.0, 2.0, dtypes.qint8)
  #   self._testDequantizeOp(np.array([-2, 4, -17]), -5.0, -3.0, dtypes.qint8)
  #   self._testDequantizeOp(np.array([0, -4, 42, -108]), 5.0, 40.0, dtypes.qint8)

  @test_util.run_deprecated_v1
  def testScaledMode(self):
    self._testDequantizeOp(np.array([-128, 0, 127]), -1.0, 2.0, dtypes.qint8,
                           mode="SCALED")
    self._testDequantizeOp(np.array([-2, 4, -17]), -5.0, -3.0, dtypes.qint8,
                           mode="SCALED")
    self._testDequantizeOp(np.array([0, -4, 42, -108]), 5.0, 40.0, dtypes.qint8,
                           mode="SCALED")

  @test_util.run_deprecated_v1
  def testNarrowRange(self):
    self._testDequantizeOp(np.array([-128, 0, 127]), -1.0, 2.0, dtypes.qint8,
                           mode="SCALED", narrow_range=True)
    self._testDequantizeOp(np.array([-2, 4, -17]), -5.0, -3.0, dtypes.qint8,
                           mode="SCALED", narrow_range=True)
    self._testDequantizeOp(np.array([0, -4, 42, -108]), 5.0, 40.0, dtypes.qint8,
                           mode="SCALED", narrow_range=True)

  @test_util.run_deprecated_v1
  def _testAxis(self, use_gpu):
    # Generates a tensor of the specified `shape` using values from `values`
    # scaled by (slice_idx + 1) along `axis` dimension.
    def scale_per_slice(shape, axis, values):
      # Note: repeats the values if the shape is larger than values.
      out = np.take(values, np.remainder(np.arange(np.prod(shape)),
                                         len(values))).reshape(shape)
      if axis is not None:
        scale_shape = [1] * len(shape)
        scale_shape[axis] = shape[axis]
        out *= np.arange(1, shape[axis] + 1).reshape(scale_shape)
      return out

    shape = np.array([2, 3, 4, 5])
    values = np.array([-128, -64, 0, 38, 102, 71, 64], dtype=np.int32)
    # TODO(itex): Here we use the results of intel-tf proper as golden ground truth
    # The original eigen broad_range dequantize result is below, intel-tf only supports narraw_range dequantization
    # dequant_values = [-2, -1.0, 0, 0.59375, 1.59375, 1.109375, 1.0]             
    dequant_values = np.array([-2.015748, -1.007874, 0, 0.5984252, 1.6062992, 1.1181102, 1.007874],
                              dtype=np.float32)
    with self.cached_session(use_gpu=use_gpu):
      for axis in [0, 1, 2, 3, None]:
        inputs = constant_op.constant(
            scale_per_slice(shape, None, values), dtype=dtypes.qint8)
        
        # TODO(itex): remove this useless relu node, when we support
        # "Dequantize" with GPU kernels
        # Here relu node doesn't participate the actual calculuation. 
        # Its existence is to guarantee there is at least one GPU node in the graph,
        # then the GPU graph optimizer will be executed
        relu_useless = nn_ops.relu(math_ops.cast(inputs, dtype=dtypes.float32))
        
        expected_dequantized = scale_per_slice(shape, axis, dequant_values)
        if axis is None:
          min_range, max_range = -2.0, 1.6
        else:
          num_slices = shape[axis]
          min_range, max_range = [], []
          for slice_idx in range(num_slices):
            min_range.append(-2.0 * (slice_idx + 1))
            max_range.append(1.6 * (slice_idx + 1))
        dequantized = self.evaluate(
            array_ops.identity(array_ops.dequantize(
                inputs, min_range, max_range, mode="SCALED", narrow_range=True, axis=axis)))
        self.assertAllClose(dequantized, expected_dequantized, rtol=1e-5, atol=1e-5)
        if axis is not None:
          dequantized = self.evaluate(
              array_ops.identity(array_ops.dequantize(
                  inputs, min_range, max_range, mode="SCALED", narrow_range=True, axis=(axis - 4))))
          self.assertAllClose(dequantized, expected_dequantized, rtol=1e-5, atol=1e-5)

  # TODO(itex): Temporarily disable CPU test, since currently GPU and CPU
  # build are separately
  @test_util.run_deprecated_v1
  def testAxisCPU(self):
    self._testAxis(use_gpu=False)

  @test_util.run_deprecated_v1
  def testAxisGPU(self):
    self._testAxis(use_gpu=True)

if __name__ == "__main__":
  test.main()
