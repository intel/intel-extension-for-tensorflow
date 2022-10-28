from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops


class ClipTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testClipByValueGradient(self):
    '''tf.raw_ops.ClipByValue has no gradient, so use clip_ops.clip_by_value'''
    inputs = constant_op.constant([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32)
    outputs_1 = clip_ops.clip_by_value(inputs, 0.5, 3.5)
    min_val = constant_op.constant([0.5, 0.5, 0.5, 0.5], dtype=dtypes.float32)
    max_val = constant_op.constant([3.5, 3.5, 3.5, 3.5], dtype=dtypes.float32)
    outputs_2 = clip_ops.clip_by_value(inputs, min_val, max_val)
    with self.cached_session():
      error_1 = gradient_checker.compute_gradient_error(inputs, [4], outputs_1,
                                                        [4])
      self.assertLess(error_1, 1e-4)

      error_2 = gradient_checker.compute_gradient_error(inputs, [4], outputs_2,
                                                        [4])
      self.assertLess(error_2, 1e-4)

  # ClipByValue test
  def testClipByValue(self):
    with self.session():
      x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
      np_ans = [[-4.4, 2.0, 3.0], [4.0, 4.4, 4.4]]
      clip_value = 4.4
      ans = tf.raw_ops.ClipByValue(t=x, clip_value_min=-clip_value, clip_value_max=clip_value)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Scalar, Scalar]
  def testClipByValue0Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.bfloat16,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [4, 4, 4]]
        clip_value_min = 2
        clip_value_max = 4
        ans = tf.raw_ops.ClipByValue(t=x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Tensor, Scalar]
  def testClipByValue1Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.bfloat16,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [4, 4, 4]]
        clip_value_min = constant_op.constant(
            [2, 2, 2, 3, 3, 3], shape=[2, 3], dtype=dtype)
        clip_value_max = 4
        ans = tf.raw_ops.ClipByValue(t=x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Scalar, Tensor]
  def testClipByValue2Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.bfloat16,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[4, 4, 4], [4, 5, 6]]
        clip_value_min = 4
        clip_value_max = constant_op.constant(
            [6, 6, 6, 6, 6, 6], shape=[2, 3], dtype=dtype)
        ans = tf.raw_ops.ClipByValue(t=x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  # [Tensor, Tensor, Tensor]
  def testClipByValue3Type(self):
    for dtype in [
        dtypes.float16,
        dtypes.float32,
        dtypes.bfloat16,
    ]:
      with self.cached_session():
        x = constant_op.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=dtype)
        np_ans = [[2, 2, 3], [5, 5, 6]]
        clip_value_min = constant_op.constant(
            [2, 2, 2, 5, 5, 5], shape=[2, 3], dtype=dtype)
        clip_value_max = constant_op.constant(
            [5, 5, 5, 7, 7, 7], shape=[2, 3], dtype=dtype)
        ans = tf.raw_ops.ClipByValue(t=x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
        tf_ans = self.evaluate(ans)

      self.assertAllClose(np_ans, tf_ans)

  def testClipByValueBadShape(self):
    with self.session():
      x = constant_op.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3, 1])
      # Use a nonsensical shape.
      clip = constant_op.constant([1.0, 2.0])
      with self.assertRaisesRegex(errors.InvalidArgumentError,
         "clip_value_min and clip_value_max must be either of the same shape as input, or a scalar."):
        _ = tf.raw_ops.ClipByValue(t=x, clip_value_min=-clip, clip_value_max=clip)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
         "clip_value_min and clip_value_max must be either of the same shape as input, or a scalar."):
        _ = tf.raw_ops.ClipByValue(t=x, clip_value_min=1.0, clip_value_max=clip)
        
  '''
  def testClipByValueNonFinite(self):
    # TODO(b/78016351): Enable test on GPU once the bug is fixed.
    with self.cached_session():
      x = constant_op.constant([float('NaN'), float('Inf'), -float('Inf')])
      np_ans = [float('NaN'), 4.0, -4.0]
      clip_value = 4.0
      ans = clip_ops.clip_by_value(x, -clip_value, clip_value)
      tf_ans = self.evaluate(ans)

    self.assertAllClose(np_ans, tf_ans)
  '''

  def _testClipIndexedSlicesByValue(self, values, indices, shape,
                                    clip_value_min, clip_value_max, expected):
    '''
    tf.raw_ops.ClipByValue cannot take IndexedSlices as input, so use clip_ops.clip_by_value
    '''
    with self.session():
      values = constant_op.constant(values)
      indices = constant_op.constant(indices)
      shape = constant_op.constant(shape)
      # IndexedSlices mode
      indexed_slices = ops.IndexedSlices(values, indices, shape)
      clipped = clip_ops.clip_by_value(indexed_slices, clip_value_min,
                                       clip_value_max)
      # clipped should be IndexedSlices
      self.assertIsInstance(clipped, ops.IndexedSlices)

    self.assertAllClose(clipped.values, expected)

  def testClipByValueWithIndexedSlicesClipped(self):
    values = [[[-3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
              [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]]
    indices = [2, 6]
    shape = [10, 2, 3]
    # [-2.0, 2.0]
    self._testClipIndexedSlicesByValue(values, indices, shape, -2.0, 2.0,
                                       [[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                                        [[0.0, 2.0, 0.0], [0.0, 0.0, -1.0]]])
    # [1.0, 2.0]
    self._testClipIndexedSlicesByValue(values, indices, shape, 1.0, 2.0,
                                       [[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                                        [[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]])
    # [-2.0, -1.0]
    self._testClipIndexedSlicesByValue(
        values, indices, shape, -2.0, -1.0,
        [[[-2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
         [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]])

  @test_util.run_deprecated_v1
  def testClipByValueEmptyTensor(self):
    # Test case for GitHub issue 19337
    zero = array_ops.placeholder(dtype=dtypes.float32, shape=None)
    x = tf.raw_ops.ClipByValue(t=zero, clip_value_min=zero, clip_value_max=zero)
    y = tf.raw_ops.ClipByValue(t=zero, clip_value_min=1.0, clip_value_max=1.0)
    z = tf.raw_ops.ClipByValue(t=zero, clip_value_min=zero, clip_value_max=1.0)
    w = tf.raw_ops.ClipByValue(t=zero, clip_value_min=1.0, clip_value_max=zero)
    with self.session() as sess:
      sess.run([x, y, z, w], feed_dict={zero: np.zeros((7, 0))})


if __name__ == '__main__':
  test.main()
