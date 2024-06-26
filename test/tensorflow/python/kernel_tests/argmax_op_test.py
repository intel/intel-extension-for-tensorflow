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
"""Tests for tensorflow.ops.argmax_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import functools

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

@test_util.run_all_in_graph_and_eager_modes
class ArgMaxTest(test.TestCase):

  def _testArg(self,
               method,
               x,
               axis,
               expected_values,
               use_gpu=False,
               expected_err_re=None):
    with self.session(use_gpu=use_gpu):
      ans = method(x, axis=axis)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        # Defaults to int64 output.
        self.assertEqual(np.int64, tf_ans.dtype)
        self.assertAllEqual(tf_ans, expected_values)
        self.assertShapeEqual(expected_values, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothArg(self,
                   method,
                   x,
                   axis,
                   expected_values,
                   expected_err_re=None):
    self._testArg(method, x, axis, expected_values, True, expected_err_re)
    # Compilation time is too large with XLA/CPU autojit.
    if not test_util.is_xla_enabled():
      self._testArg(method, x, axis, expected_values, False, expected_err_re)

  def _testBasic(self, dtype):
    x = np.arange(200, dtype=np.float32).astype(np.bool_).astype(dtype)
    np.random.shuffle(x)

    # Check that argmin and argmax match numpy along the primary axis
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())


  def testOpt(self):
      shapes = [
            [1024],
            [1024*64],
            [1024*256],
            [1024,262144],
            [1024, 32],
            [1024, 1024],
            [1024, 2048],
            [32, 16384],
            [2048, 1024],
            [256, 16],
            [4*320*512,16],
            [4, 256, 16],
            [2,128,128,128,32],
            [3,4,640,1024,3]]

      axises = [
              0,
              0,
              0,
              0,
              1,
              1,
              1,
              1,
              0,
              0,
              0,
              1,
              2,
              4]

      for shape, axis in zip(shapes, axises):
          x = np.random.rand(*shape).astype(np.float32)
          self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis=axis))

  def _testTieBreaking(self, dtype):
    x = np.zeros(200, dtype=dtype)

    # Check that argmin and argmax match numpy along the primary axis for
    # breaking ties.
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testDim(self, dtype):
    shape = (3, 2, 4, 5, 6, 3, 7)
    x = np.arange(
        functools.reduce(lambda x, y: x * y, shape),
        dtype=np.float32).astype(dtype)
    np.random.shuffle(x)
    x = x.reshape(shape)

    # Check that argmin and argmax match numpy along all axes
    for axis in range(-7, 7):
      self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis))
      self._testBothArg(math_ops.argmin, x, axis, x.argmin(axis))

  def testFloat(self):
    self._testBasic(np.float32)
    self._testTieBreaking(np.float32)
    self._testDim(np.float32)

  def testFloatInt32Output(self):
    x = np.asarray(100 * np.random.randn(200), dtype=np.float32)
    expected_values = x.argmax()
    with self.session(use_gpu=True):
      ans = math_ops.argmax(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      # The values are equal when comparing int32 to int64 because
      # the values don't have a range that exceeds 32-bit integers.
      self.assertAllEqual(tf_ans, expected_values)
    expected_values = x.argmin()
    with self.session(use_gpu=True):
      ans = math_ops.argmin(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      self.assertAllEqual(tf_ans, expected_values)

  def testInt32(self):
    self._testBasic(np.int32)
    self._testTieBreaking(np.int32)
    self._testDim(np.int32)

  def testInt64(self):
    self._testBasic(np.int64)
    self._testTieBreaking(np.int64)
    self._testDim(np.int64)

  def testDouble(self):
    self._testBasic(np.float64)
    self._testTieBreaking(np.float64)
    self._testDim(np.float64)


  #TODO(itex): Bool dtype has accuracy issue [TFDO-4329].
  #def testBool(self):
  #  self._testBasic(np.bool_)
  #  self._testTieBreaking(np.bool_)
  #  self._testDim(np.bool_)

  def testEmpty(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        with self.assertRaisesOpError(
            r"Reduction axis 0 is empty in shape \[0\]"):
          op([], 0).eval()

  @test_util.run_deprecated_v1
  def testDefaultAxis(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ans = op([1]).eval()
        self.assertAllEqual(ans, 0)

  @test_util.run_deprecated_v1
  def testOutputEmpty(self):
    with self.cached_session():
      for op in math_ops.argmin, math_ops.argmax:
        ret = op(array_ops.zeros(shape=[1, 0, 2]), axis=-1).eval()
        self.assertEqual(ret.shape, (1, 0))


if __name__ == "__main__":
  test.main()
