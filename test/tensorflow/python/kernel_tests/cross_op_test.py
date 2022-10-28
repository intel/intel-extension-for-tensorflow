"""Tests for the cross operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops

class CrossTest(test.TestCase):

  def _testZero(self, dtypes_to_test):
    for dtype in dtypes_to_test:
      inx = constant_op.constant([0, 0, 0], dtype=dtype)
      iny = constant_op.constant([0, 0, 0], dtype=dtype)
      out = math_ops.cross(inx, iny)
      expected_out = constant_op.constant([0, 0, 0], dtype=dtype)

      with self.session(use_gpu=False):
        self.assertAllClose(expected_out, self.evaluate(out))
      if test.is_gpu_available():
        with self.session(use_gpu=True):
          self.assertAllClose(expected_out, self.evaluate(out))

  def _testRightHandRule(self, dtypes_to_test):
    for dtype in dtypes_to_test:
      inx = constant_op.constant([1, 0, 0, 0, 1, 0], shape=[2,3], dtype=dtype)
      iny = constant_op.constant([0, 1, 0, 1, 0, 0], shape=[2,3], dtype=dtype)
      out = math_ops.cross(inx, iny)
      expected_out = constant_op.constant([0, 0, 1, 0, 0, -1], shape=[2,3], dtype=dtype)

      with self.session(use_gpu=False):
        self.assertAllClose(expected_out, self.evaluate(out))
      if test.is_gpu_available():
        with self.session(use_gpu=True):
          self.assertAllClose(expected_out, self.evaluate(out))

  def _testArbitraryNonintegral(self, dtypes_to_test):
    for dtype in dtypes_to_test:
      np_dtype = 'float32' if dtype == dtypes.float32 else 'float64'
      npx = np.array([ -0.669, -0.509, 0.125], dtype=np_dtype)
      npy = np.array([ -0.477, 0.592, -0.110], dtype=np_dtype)
      tfx = constant_op.constant(npx, dtype=dtype)
      tfy = constant_op.constant(npy, dtype=dtype)
      np_out = np.cross(npx, npy)
      tf_out = math_ops.cross(tfx, tfy)

      with self.session(use_gpu=False):
        self.assertAllClose(np_out, self.evaluate(tf_out))

      if test.is_gpu_available():
        with self.session(use_gpu=True):
          self.assertAllClose(np_out, self.evaluate(tf_out))

  @test_util.run_deprecated_v1
  def testFloatTypes(self):
    dtypes_to_test = [dtypes.float32, dtypes.float64]
    self._testZero(dtypes_to_test)
    self._testRightHandRule(dtypes_to_test)
    self._testArbitraryNonintegral(dtypes_to_test)

  @test_util.run_deprecated_v1
  def testIntTypes(self):
    for dtype in [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]:
      inx = constant_op.constant([2, 0, 0, 0, 2, 0], shape=[2,3], dtype=dtype)
      iny = constant_op.constant([0, 2, 0, 2, 0, 0], shape=[2,3], dtype=dtype)
      out = math_ops.cross(inx, iny)
      expected_out = constant_op.constant([0, 0, 4, 0, 0, -4], shape=[2,3], dtype=dtype)

      with self.session(use_gpu=False):
        self.assertAllClose(expected_out, self.evaluate(out))
      if test.is_gpu_available():
        with self.session(use_gpu=True):
          self.assertAllClose(expected_out, self.evaluate(out))

if __name__ == '__main__':
  test.main()
