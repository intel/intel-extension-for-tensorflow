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

"""Functional tests for reduction ops."""
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import itertools
import numbers

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

# The maximum input rank to test.
_MAX_RANK = 5

def _powerset(iterable):
    """Helper for generating all possible reduction_axes arguments.

    Example:
    powerset([0,1,2]): () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)

    Args:
      iterable: An iterable of items to generate the powerset of.

    Returns:
      The powerset of all items in iterable.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1))



class BaseReductionTest(test.TestCase):

    def _tf_reduce(self, x, reduction_axes, keepdims):
        raise NotImplementedError()

    def _np_reduce(self, x, reduction_axes, keepdims):
        raise NotImplementedError()

    def _makeIncremental(self, shape, dtype):
        data = np.arange(np.prod(shape)).reshape(shape).astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
            data -= 2j * data
        return data

    def _makeRandom(self, shape, dtype):
        data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
            data -= 2j * data
        return data

    def _compare(self, x, reduction_axes, keepdims, feed_dict=None):
        np_ans = self._np_reduce(x, reduction_axes, keepdims)
        with self.cached_session(use_gpu=True) as sess:
            tf_ans = self._tf_reduce(x, reduction_axes, keepdims)
            out = sess.run(tf_ans, feed_dict)
        self.assertAllClose(np_ans, out)
        self.assertShapeEqual(np_ans, tf_ans)

    def _compareAll(self, x, reduction_axes, feed_dict=None):
        if reduction_axes is not None and np.shape(reduction_axes) == (1,):
            # Test scalar reduction_axes argument
            self._compareAll(x, reduction_axes[0])
        self._compare(x, reduction_axes, keepdims=False, feed_dict=feed_dict)
        self._compare(x, reduction_axes, keepdims=True, feed_dict=feed_dict)

    def _compareAllAxes(self, x, feed_dict=None):
        self._compareAll(x, None)
        for axes in _powerset(range(x.ndim)):
            self._compareAll(x, axes, feed_dict)

    def _compareGradient(self, x, reduction_axes, rtol=1e-8, atol=1e-8):
        if reduction_axes is not None and np.shape(reduction_axes) == (1,):
            # Test scalar reduction_axes argument
            self._compareGradient(x, reduction_axes[0], rtol=rtol, atol=atol)
        with self.cached_session(use_gpu=True):
            t = ops.convert_to_tensor(x)
            su = self._tf_reduce(t, reduction_axes, False)
            jacob_t, jacob_n = gradient_checker.compute_gradient(
                t, x.shape, su, su.get_shape().as_list(), x_init_value=x, delta=1)
        self.assertAllClose(jacob_t, jacob_n, rtol=rtol, atol=atol)

    def _compareGradientAxes(self, x, rtol=1e-8, atol=1e-8):
        self._compareGradient(x, None, rtol=rtol, atol=atol)
        self._compareGradient(x, [], rtol=rtol, atol=atol)
        self._compareGradient(x, 0, rtol=rtol, atol=atol)
        self._compareGradient(x, [1], rtol=rtol, atol=atol)
        self._compareGradient(x, [2], rtol=rtol, atol=atol)
        self._compareGradient(x, [1, 2], rtol=rtol, atol=atol)
        self._compareGradient(x, [0, 1, 2, 3], rtol=rtol, atol=atol)


class SumReductionTest(BaseReductionTest):

    def _tf_reduce(self, x, reduction_axes, keepdims):
        return math_ops.reduce_sum(x, reduction_axes, keepdims)

    def _np_reduce(self, x, reduction_axes, keepdims):
        if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                          np.ndarray):
            reduction_axes = tuple(reduction_axes)
        return np.sum(x, axis=reduction_axes, keepdims=keepdims)
        
    @test_util.run_deprecated_v1
    def testVecColReduciton(self):
        arr = np.ones([2, 128 * 128 * 128, 32], dtype=np.float32)
        row_sum = np.sum(arr, axis=1)

        with self.session(graph=ops.Graph(), use_gpu=True) as sess:
            tf_row_sum = self._tf_reduce(arr, 1, False)
            tf_out_row = self.evaluate(tf_row_sum)
        self.assertAllClose(row_sum, tf_out_row)        

    @test_util.run_deprecated_v1
    def testRowReduciton(self):
      for size in [32, 33, 256 * 8 * 4, 256 * 8 * 4 + 1]:
        arr = np.ones([2, size], dtype=np.float32)
        row_sum = np.sum(arr, axis=1)

        with self.session(graph=ops.Graph(), use_gpu=True) as sess:
            tf_row_sum = self._tf_reduce(arr, 1, False)
            tf_out_row = self.evaluate(tf_row_sum)
        self.assertAllClose(row_sum, tf_out_row)      
        
    @test_util.run_deprecated_v1
    def testReducitonBF16(self):
      for axis in [None, 0, 1]:
        arr = np.ones([2, 1025], dtype=dtypes.bfloat16.as_numpy_dtype)
        row_sum = np.sum(arr.astype(np.float32), axis=axis)

        with self.session(graph=ops.Graph(), use_gpu=True) as sess:
            tf_row_sum = self._tf_reduce(arr, axis, False)
            tf_out_row = self.evaluate(tf_row_sum)
        self.assertAllClose(row_sum, tf_out_row, 1e-2, 1e-2) 
           

    @test_util.run_deprecated_v1
    def testFloat32(self):
        for rank in range(1, _MAX_RANK + 1):
            np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
            self._compareAllAxes(np_arr)

        for _ in range(10):
            size_x = int(2**np.random.uniform(0, 15))
            size_y = int(2**np.random.uniform(0, 15))

            if size_x * size_y > 1e7:
                size_y = int(1e7 / size_x)

            arr = np.ones([size_x, size_y], dtype=np.float32)
            col_sum = np.sum(arr, axis=0)
            row_sum = np.sum(arr, axis=1)

            with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                tf_row_sum = self._tf_reduce(arr, 1, False)
                tf_col_sum = self._tf_reduce(arr, 0, False)
                tf_out_row, tf_out_col = self.evaluate([tf_row_sum, tf_col_sum])
            self.assertAllClose(col_sum, tf_out_col)
            self.assertAllClose(row_sum, tf_out_row)

        for size_x in [1, 3, 16, 33]:
            for size_y in [1, 3, 16, 33]:
                for size_z in [1, 3, 16, 33]:
                    arr = np.ones([size_x, size_y, size_z], dtype=np.float32)
                    sum_y = np.sum(arr, axis=1)
                    sum_xz = np.sum(arr, axis=(0, 2))

                    with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                        tf_sum_xz = self._tf_reduce(arr, [0, 2], False)
                        tf_sum_y = self._tf_reduce(arr, 1, False)
                        tf_out_sum_xz, tf_out_sum_y = self.evaluate([tf_sum_xz, tf_sum_y])
                    self.assertAllClose(sum_y, tf_out_sum_y)
                    self.assertAllClose(sum_xz, tf_out_sum_xz)

    @test_util.run_deprecated_v1
    def testDouble(self):
        for rank in range(1, _MAX_RANK + 1):
            np_arr = self._makeIncremental((2,) * rank, dtypes.float64)
            self._compareAllAxes(np_arr)

        for _ in range(10):
            size_x = int(2**np.random.uniform(0, 15))
            size_y = int(2**np.random.uniform(0, 15))

            if size_x * size_y > 1e7:
                size_y = int(1e7 / size_x)

            arr = np.ones([size_x, size_y], dtype=np.float64)
            col_sum = np.sum(arr, axis=0)
            row_sum = np.sum(arr, axis=1)

            with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                tf_row_sum = self._tf_reduce(arr, 1, False)
                tf_col_sum = self._tf_reduce(arr, 0, False)
                tf_out_row, tf_out_col = self.evaluate([tf_row_sum, tf_col_sum])
            self.assertAllClose(col_sum, tf_out_col)
            self.assertAllClose(row_sum, tf_out_row)

        for size_x in [1, 3, 16, 33]:
            for size_y in [1, 3, 16, 33]:
                for size_z in [1, 3, 16, 33]:
                    arr = np.ones([size_x, size_y, size_z], dtype=np.float64)
                    sum_y = np.sum(arr, axis=1)
                    sum_xz = np.sum(arr, axis=(0, 2))

                    with self.session(graph=ops.Graph(), use_gpu=True) as sess:
                        tf_sum_xz = self._tf_reduce(arr, [0, 2], False)
                        tf_sum_y = self._tf_reduce(arr, 1, False)
                        tf_out_sum_xz, tf_out_sum_y = self.evaluate([tf_sum_xz, tf_sum_y])
                    self.assertAllClose(sum_y, tf_out_sum_y)
                    self.assertAllClose(sum_xz, tf_out_sum_xz)
    
    @test_util.run_deprecated_v1
    def testGradient(self):
        for dtype in [
            dtypes.float32, dtypes.float64
        ]:
            x = self._makeIncremental([2, 3, 4, 2], dtype)
            self._compareGradientAxes(x)


class MeanReductionTest(BaseReductionTest):

    def _tf_reduce(self, x, reduction_axes, keepdims):
        return math_ops.reduce_mean(x, reduction_axes, keepdims)

    def _np_reduce(self, x, reduction_axes, keepdims):
        if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                          np.ndarray):
            reduction_axes = tuple(reduction_axes)
        elif isinstance(reduction_axes, numbers.Integral):
            reduction_axes = (reduction_axes,)

        if reduction_axes is None:
            count = np.prod(x.shape)
        else:
            count = np.prod([x.shape[ax] for ax in reduction_axes])
        # np.mean automatically converts integer inputs to float, while TensorFlow's
        # reduce_mean does not. For integer inputs, we emulate TensorFlow's behavior
        # using np.sum and truncating division.
        np_sum = np.sum(x, axis=reduction_axes, keepdims=keepdims)
        if np.issubdtype(x.dtype, np.integer):
            return np_sum // count
        return np_sum / count

    @test_util.run_deprecated_v1
    def testFloat32(self):
        for rank in range(1, _MAX_RANK + 1):
            np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
            self._compareAllAxes(np_arr)

    @test_util.run_deprecated_v1
    def testGradient(self):
        s = [2, 3, 4, 2]
        for dtype in [dtypes.float32]:
            x = self._makeIncremental(s, dtype)
            self._compareGradientAxes(x, rtol=1e-3, atol=1e-3)

class MaxReductionTest(test.TestCase):

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False):
    np_ans = x
    if reduction_axes is None:
      np_ans = np.amax(np_ans, keepdims=keepdims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.amax(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_max(x, reduction_axes, keepdims)
      out = self.evaluate(tf_ans)
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=True)

  def testAxesType(self):
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session(use_gpu=True) as sess:
        v = math_ops.reduce_max([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  # TODO(itex): Enable it when Eigen::PropagateNaN is implemented.
  # def testSpecialValues(self):
  #   for dtype in [np.float32, np.float64]:
  #     for size in range(1, 4):
  #       for arr in itertools.product([-np.inf, 1., np.nan, np.inf],
  #                                    repeat=size):
  #         self._compareAll(np.array(arr, dtype=dtype), None)

  def testInt64Reduce3D(self):
    # Create a 3D array of int64s and reduce across all possible
    # dimensions
    np_arr = np.arange(-31, -1).reshape([2, 3, 5]).astype(np.int64)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  def testFloatReduce3D(self):
    # Create a 3D array of floats and reduce across all possible
    # dimensions
    np_arr = np.arange(-31, -1).reshape([2, 3, 5]).astype(np.float32)
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

  @test_util.run_deprecated_v1
  def testGradient(self):
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [1, 2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient2(self):
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [1])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 4, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient3(self):
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t, [2])
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [2, 3, 2], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient4(self):
    s = [2, 3, 4, 2]
    x = np.arange(-49.0, -1.0).reshape(s).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      su = math_ops.reduce_max(t)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, s, su, [1], x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testEmptyGradients(self):
    with self.cached_session():
      x = array_ops.zeros([0, 3])
      y = math_ops.reduce_max(x, [1])
      error = gradient_checker.compute_gradient_error(x, [0, 3], y, [0])
      self.assertEqual(error, 0)

class AllReductionTest(test.TestCase):

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False):
    np_ans = x
    if reduction_axes is None:
      np_ans = np.all(np_ans, keepdims=keepdims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.all(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_all(x, reduction_axes, keepdims)
      out = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testAll3D(self):
    # Create a 3D array of bools and reduce across all possible
    # dimensions
    np_arr = (np.random.uniform(0, 1, 30) > 0.1).reshape([2, 3, 5])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

class AnyReductionTest(test.TestCase):

  def _compare(self, x, reduction_axes, keepdims, use_gpu=False):
    np_ans = x
    if reduction_axes is None:
      np_ans = np.any(np_ans, keepdims=keepdims)
    else:
      for ra in reduction_axes[::-1]:
        np_ans = np.any(np_ans, axis=ra, keepdims=keepdims)
    with self.cached_session(use_gpu=use_gpu):
      if reduction_axes is not None:
        reduction_axes = np.array(reduction_axes).astype(np.int32)
      tf_ans = math_ops.reduce_any(x, reduction_axes, keepdims)
      out = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, x, reduction_axes):
    self._compare(x, reduction_axes, False, use_gpu=True)
    self._compare(x, reduction_axes, False, use_gpu=False)
    self._compare(x, reduction_axes, True, use_gpu=True)
    self._compare(x, reduction_axes, True, use_gpu=False)

  def testAll3D(self):
    # Create a 3D array of bools and reduce across all possible
    # dimensions
    np_arr = (np.random.uniform(0, 1, 30) > 0.9).reshape([2, 3, 5])
    self._compareAll(np_arr, None)
    self._compareAll(np_arr, [])
    self._compareAll(np_arr, [0])
    self._compareAll(np_arr, [1])
    self._compareAll(np_arr, [2])
    self._compareAll(np_arr, [0, 1])
    self._compareAll(np_arr, [1, 2])
    self._compareAll(np_arr, [0, 2])
    self._compareAll(np_arr, [0, 1, 2])

class ProdReductionTest(BaseReductionTest):
  
  def _tf_reduce(self, x, reduction_axes, keepdims):
    return math_ops.reduce_prod(x, reduction_axes, keepdims)

  def _np_reduce(self, x, reduction_axes, keepdims):
    if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                      np.ndarray):
      reduction_axes = tuple(reduction_axes)
    return np.prod(x, axis=reduction_axes, keepdims=keepdims)

  @test_util.run_deprecated_v1
  def testFloat32(self):
    for rank in range(1, _MAX_RANK + 1):
      np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
      self._compareAllAxes(np_arr)

  @test_util.run_deprecated_v1
  def testGradientWithZeros(self):
    s = [2, 3, 4, 2]
    x = self._makeIncremental(s, dtypes.float32) / 20.
    # No zeros in input
    self._compareGradientAxes(x, rtol=1e-3, atol=1e-3)
    # Zero at beginning
    x1 = x.copy()
    x1[:, :, 0, :] = 0
    self._compareGradientAxes(x1, rtol=1e-3, atol=1e-3)
    # Zero at end
    x2 = x.copy()
    x2[:, :, -1, :] = 0
    self._compareGradientAxes(x2, rtol=1e-3, atol=1e-3)
    # Zero in middle
    x3 = x.copy()
    x3[:, :, 2, :] = 0
    self._compareGradientAxes(x3, rtol=1e-3, atol=1e-3)
    # All zeros
    x4 = x.copy()
    x4[:, :, :, :] = 0
    self._compareGradientAxes(x4, rtol=1e-3, atol=1e-3)

class ArgMinTest(test_util.TensorFlowTestCase):
  def testArgmin(self):
    np_array = np.array([[1, 1, 2, 2],[2, 1, 2, 1],[1, 2, 1, 1]])
    with test_util.use_gpu():
      tf_values = constant_op.constant(np_array)
      np_min = np.argmin(np_array, axis=0)
      tf_min = math_ops.argmin(tf_values, axis=0)
      self.assertAllEqual(np_min, tf_min)
  
if __name__ == "__main__":
    test.main()
