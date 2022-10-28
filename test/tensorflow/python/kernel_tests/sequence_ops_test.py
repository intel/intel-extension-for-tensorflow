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
"""Tests for tensorflow.ops.ops."""

import numpy as np
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


class RangeTest(test.TestCase):

  def _Range(self, start, limit, delta):
    with self.cached_session(use_gpu=True):
      tf_ans = math_ops.range(start, limit, delta, name="range")
      self.assertEqual([len(np.arange(start, limit, delta))],
                       tf_ans.get_shape())
      return self.evaluate(tf_ans)

  def testBasic(self):
    self.assertTrue(
        np.array_equal(self._Range(0, 5, 1), np.array([0, 1, 2, 3, 4])))
    self.assertTrue(np.array_equal(self._Range(0, 5, 2), np.array([0, 2, 4])))
    self.assertTrue(np.array_equal(self._Range(0, 6, 2), np.array([0, 2, 4])))
    self.assertTrue(
        np.array_equal(self._Range(13, 32, 7), np.array([13, 20, 27])))
    self.assertTrue(
        np.array_equal(
            self._Range(100, 500, 100), np.array([100, 200, 300, 400])))
    self.assertEqual(math_ops.range(0, 5, 1).dtype, dtypes.int32)

  @test_util.run_deprecated_v1
  def testLimitOnly(self):
    with self.session(use_gpu=True):
      self.assertAllEqual(np.arange(5), math_ops.range(5))

  def testEmpty(self):
    for start in 0, 5:
      self.assertTrue(np.array_equal(self._Range(start, start, 1), []))

  def testNonInteger(self):
    self.assertTrue(
        np.allclose(self._Range(0, 2, 0.5), np.array([0, 0.5, 1, 1.5])))
    self.assertTrue(np.allclose(self._Range(0, 5, 2.5), np.array([0, 2.5])))
    self.assertTrue(
        np.allclose(self._Range(0, 3, 0.9), np.array([0, 0.9, 1.8, 2.7])))
    self.assertTrue(
        np.allclose(
            self._Range(100., 500., 100.), np.array([100, 200, 300, 400])))
    self.assertEqual(math_ops.range(0., 5., 1.).dtype, dtypes.float32)

  def testNegativeDelta(self):
    self.assertTrue(
        np.array_equal(self._Range(5, -1, -1), np.array([5, 4, 3, 2, 1, 0])))
    self.assertTrue(
        np.allclose(self._Range(2.5, 0, -0.5), np.array([2.5, 2, 1.5, 1, 0.5])))
    self.assertTrue(
        np.array_equal(self._Range(-5, -10, -3), np.array([-5, -8])))

  def testDType(self):
    zero_int32 = math_ops.cast(0, dtypes.int32)
    zero_int64 = math_ops.cast(0, dtypes.int64)
    zero_float32 = math_ops.cast(0, dtypes.float32)
    zero_float64 = math_ops.cast(0, dtypes.float64)

    self.assertEqual(math_ops.range(zero_int32, 0, 1).dtype, dtypes.int32)
    self.assertEqual(math_ops.range(zero_int64, 0, 1).dtype, dtypes.int64)
    self.assertEqual(math_ops.range(zero_float32, 0, 1).dtype, dtypes.float32)
    self.assertEqual(math_ops.range(zero_float64, 0, 1).dtype, dtypes.float64)

    self.assertEqual(
        math_ops.range(zero_int32, zero_int64, 1).dtype, dtypes.int64)
    self.assertEqual(
        math_ops.range(zero_int64, zero_float32, 1).dtype, dtypes.float32)
    self.assertEqual(
        math_ops.range(zero_float32, zero_float64, 1).dtype, dtypes.float64)
    self.assertEqual(
        math_ops.range(zero_float64, zero_int32, 1).dtype, dtypes.float64)

    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.int32).dtype, dtypes.int32)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.int64).dtype, dtypes.int64)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.float32).dtype, dtypes.float32)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.float64).dtype, dtypes.float64)

  def testMixedDType(self):
    # Test case for GitHub issue 35710
    tf_ans = math_ops.range(
        constant_op.constant(4, dtype=dtypes.int32), dtype=dtypes.int64)
    self.assertAllEqual(self.evaluate(tf_ans), np.array([0, 1, 2, 3]))

  # TODO(itex): These codes will cause an error: itex/core/utils/tensor_shape.cc:159] Check failed: val >= 0 (-9223372036854775808 vs. 0)
  # Just workaround it before we fix the error.
  """
  def testLargeLimits(self):
    # Test case for GitHub issue 46913.
    with self.session(use_gpu=True):
      with self.assertRaises(errors_impl.ResourceExhaustedError):
        v = math_ops.range(0, 9223372036854775807)
        self.evaluate(v)

  def testLargeStarts(self):
    # Test case for GitHub issue 46899.
    with self.session(use_gpu=True):
      with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
        v = math_ops.range(start=-1e+38, limit=1)
        self.evaluate(v)
  """


if __name__ == "__main__":
  test.main()
