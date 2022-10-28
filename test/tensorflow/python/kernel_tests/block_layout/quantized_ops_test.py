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

import numpy as np
import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

# TODO(itex): Test Quantize op in eager mode (non-block-layout pass),
# when we support it

# TODO(itex): For intel-tf proper, the Quantize and Dequantize op only have narrow_range
# implementation, regardless the `narrow_range` flag is True or False. Currently, ITEX also 
# follow such logic.

@test_util.run_all_in_native_and_block_format
class QuantizedOpsTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedOpsTest, self).__init__(method_name)

  @test_util.run_deprecated_v1
  def testQuantizeOp(self):
    expected_output = [1, 1, 2, 127, 255, 255]
    for use_gpu in [False, True]:
      with self.session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(
            [1.0, 1.25, 1.75, 127.0, 255.0, 500.0],
            shape=[6],
            dtype=dtypes.float32)
        x_min = 0.0
        x_max = 255.0
        quantize_op = array_ops.quantize(x, x_min, x_max, dtypes.quint8, mode="MIN_FIRST")
     
      # Very special case. Since we don't rewrite the last node of the graph. We need to add a "Identity" node. 
      # "Identity" op only accepts single type of all inputs. However, "QuantizeV2" output datatypes are different.
      # (float,int8,int8)
      # So we only take the first tensor as the input of identity op
      identity_op = array_ops.identity(quantize_op[0])
      value = self.evaluate(identity_op)
      self.assertArrayNear(expected_output, value, 0.1)

  @test_util.run_deprecated_v1
  def testQuantizeAsymmetricOp(self):
    # Actually Intel-TF-2.6.0 QuantizeV2 result is different from public TF. 
    # Here we regard Intel TF result as golden data. public TF result is left 
    # here, maybe useful for debugging.
    public_tf_output = [0, 144, 176, 223, 239, 255]
    intel_tf_output = [0, 143, 175, 223, 239, 255]
    # We only test the condition use_gpu=True. If testing use_gpu=False, it will fall back to 
    # the implementation stock TF Quantize, which has different result
    for use_gpu in [True]:
      with self.session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(
            [-1.0, 1.25, 1.75, 2.5, 2.75, 3.0],
            shape=[6],
            dtype=dtypes.float32)

        # For some reasons, intel-tf doesn't rewrite QuantizeV2 with input node is "Const" op. 
        # ITEX actually doesn't have such restrictions. Anyway, for convenient debugging, 
        # just add an Identity op here. Not sure why identity(x) not work, while identity([x]) works
        # maybe investigate in the future 
        x = array_ops.identity([x])

        x_min = -1.0
        x_max = 3.0
        quantize_op = array_ops.quantize(x, x_min, x_max, dtypes.quint8, narrow_range=False, mode="MIN_FIRST")
     
      # Very special case. Since we don't rewrite the last node of the graph. We need to add a "Identity" node. 
      # "Identity" op only accepts single type of all inputs. However, "QuantizeV2" output datatypes are different.
      # (float,int8,int8)
      # So we only take the first tensor as the input of identity op
      identity_op = array_ops.identity(quantize_op[0])
      value = self.evaluate(identity_op)
      self.assertArrayNear(intel_tf_output, value[0], 0.1)

  @test_util.run_deprecated_v1
  def testDequantizeOp(self):
    expected_output = [1.0, 2.0, 4.0, 8.0, 16.0, 255.0]
    inp = np.array([1, 2, 4, 8, 16, 255]).astype(np.uint8)

    for use_gpu in [False, True]:
      with self.session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(inp, shape=[6], dtype=dtypes.quint8)
        x_min = 0.0
        x_max = 255.0
        op = array_ops.identity(array_ops.dequantize(x, x_min, x_max, mode="MIN_FIRST"))
        value = self.evaluate(op)
        self.assertArrayNear(expected_output, value, 0.1)

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
    values = np.array([-1, -0.5, 0, 0.3, 0.8, 0.555, 0.5], dtype=np.float32)
    # TODO(itex): intel TF implementaion is always narrow range [-127, 127]
    # regardless the flag "narrow_range"
    # However public TF uses the broad range [-128, 127], 
    # when "narrow_range" = False
    # Below is the result for original public TF standard in broad range
    # [-128, -64, 0, 38, 102, 71, 64]
    quant_values = np.array([-127, -64, 0, 38, 102, 70, 64], dtype=np.int32)
    with self.cached_session(use_gpu=use_gpu):
      for axis in [1, 2, 3, None]:
        inputs = constant_op.constant(scale_per_slice(shape, axis, values))
        print("inputs", inputs)
        
        # TODO(itex): remove this useless relu node, when we support
        # "QuantizeV2" with GPU kernels
        # Here relu node doesn't participate the actual calculuation. 
        # Its existence is to guarantee there is at least one GPU node in the graph,
        # then the GPU graph optimizer will be executed
        relu_useless = nn_ops.relu(inputs)
        
        expected_quantized = scale_per_slice(shape, None, quant_values)
        if axis is None:
          min_range, max_range = -1.0, 0.8
        else:
          num_slices = shape[axis]
          min_range, max_range = [], []
          for slice_idx in range(num_slices):
            min_range.append(-1.0 * (slice_idx + 1))
            max_range.append(0.8 * (slice_idx + 1))
        quantize_op = array_ops.quantize(
                inputs,
                min_range,
                max_range,
                T=dtypes.qint8,
                mode="SCALED",
                round_mode="HALF_TO_EVEN",
                narrow_range=True,
                axis=axis)
        
        # Reasons for using quantize_op[0] instead of quantize_op is explainedn above.
        # All inputs' datatype of "Identity" should be the same
        identity_op = array_ops.identity(quantize_op[0])
        quantized = self.evaluate(identity_op)
        self.assertAllEqual(quantized, expected_quantized)
        
        if axis is not None:
          quantize_op = array_ops.quantize(
                    inputs,
                    min_range,
                    max_range,
                    T=dtypes.qint8,
                    mode="SCALED",
                    round_mode="HALF_TO_EVEN",
                    narrow_range=True,
                    axis=(axis - 4))
          identity_op = array_ops.identity(quantize_op[0])
          quantized = self.evaluate(identity_op)
          self.assertAllClose(quantized, expected_quantized)

  @test_util.run_deprecated_v1
  def testAxisCPU(self):
    self._testAxis(use_gpu=False)

  @test_util.run_deprecated_v1
  def testAxisGPU(self):
    self._testAxis(use_gpu=True)

if __name__ == "__main__":
  test.main()
