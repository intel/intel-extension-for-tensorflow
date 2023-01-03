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
"""Tests for tensorflow.ops.tf.gather_v1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow import raw_ops

_TEST_TYPES = (dtypes.int64, dtypes.float32)

class GatherV1Test(test.TestCase, parameterized.TestCase):

  def _buildParams(self, data, dtype):
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testGatherV1(self):
    with self.session(use_gpu=True):
      data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8,
                       9, 10, 11, 12, 13, 14])
      for dtype in _TEST_TYPES:
        params_np = self._buildParams(data, dtype)
        params = constant_op.constant(params_np)
        indices = constant_op.constant([0, 1, 0, 2])
        gather_t = raw_ops.Gather(params=params, indices=indices)
        gather_val = self.evaluate(gather_t)
        self.assertAllEqual(np.take(params_np, [0, 1, 0, 2]),
                            gather_val)
        expected_shape = data.shape[1:] + indices.shape
        self.assertEqual(expected_shape, gather_t.get_shape())

if __name__ == "__main__":
  test.main()
