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
from tensorflow.python.ops.gen_nn_ops import quantized_max_pool

os.environ["ITEX_LAYOUT_OPT"] = "1"

# TODO(itex): Test Quantize op in eager mode (non-block-layout pass),
# when we support it

class QuantizedOpsTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(QuantizedOpsTest, self).__init__(method_name)

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
    with self.cached_session(use_gpu=use_gpu) as sess:
      for axis in [None, None]:
        inputs = constant_op.constant(scale_per_slice(shape, axis, values))
        
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
                T=dtypes.quint8,
                mode="SCALED",
                round_mode="HALF_TO_EVEN",
                narrow_range=True,
                axis=axis)
        
        # Reasons for using quantize_op[0] instead of quantize_op is explainedn above.
        # All inputs' datatype of "Identity" should be the same
        res, min_res, max_res = quantized_max_pool(quantize_op[0], quantize_op[1], quantize_op[2], ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="VALID")
        identity_op = array_ops.identity(res)
        print(sess.run(identity_op))
        #self.assertAllEqual(quantized, expected_quantized)
 
  @test_util.run_deprecated_v1
  def testAxisGPU(self):
    self._testAxis(use_gpu=True)

if __name__ == "__main__":
  test.main()
