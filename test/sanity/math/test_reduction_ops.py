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

import itertools
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import tensorflow as tf

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


class BaseReductionTest(test_util.TensorFlowTestCase):
    """test reduction ops"""

    def _tf_reduce(self, x, reduction_axes, keepdims):
        raise NotImplementedError()

    def _np_reduce(self, x, reduction_axes, keepdims):
        raise NotImplementedError()

    def _makeIncremental(self, shape, dtype):
        data = np.arange(np.prod(shape)).reshape(
            shape).astype(dtype.as_numpy_dtype)
        if dtype.is_complex:
            data -= 2j * data
        return data

    def _compare(self, x, reduction_axes, keepdims, feed_dict=None):
        np_ans = self._np_reduce(x, reduction_axes, keepdims)
        with self.cached_session(use_gpu=True) as sess:
            dtype = tf.bfloat16
            tf_ans = self._tf_reduce(
                tf.cast(x, dtype=dtype), reduction_axes, keepdims)
            tf_ans = tf.cast(tf_ans, tf.float32)
            out = sess.run(tf_ans, feed_dict)
        self.assertAllClose(np_ans, out, rtol=1e-2, atol=1e-2)
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


class SumAndMeanReductionTest(BaseReductionTest):

    def _tf_reduce(self, x, reduction_axes, keepdims):
        return math_ops.reduce_sum(x, reduction_axes, keepdims)

    def _np_reduce(self, x, reduction_axes, keepdims):
        if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                          np.ndarray):
            reduction_axes = tuple(reduction_axes)
        return np.sum(x, axis=reduction_axes, keepdims=keepdims)

    @test_util.run_deprecated_v1
    def testSumBf16(self):
        arr = np.ones([68000], dtype=np.float32)
        dtype = tf.bfloat16

        with self.session(graph=ops.Graph(), use_gpu=True):
            tf_arr = variables.Variable(arr)
            variables.global_variables_initializer().run()
            tf_mean = math_ops.reduce_mean(
                tf.cast(tf_arr, dtype=dtype), 0, False)
            tf_out_mean = self.evaluate(tf_mean)
        self.assertAllClose(tf_out_mean, 1., rtol=1e-2, atol=1e-2)

    @test_util.run_deprecated_v1
    def testSumBf16(self):
        arr = np.ones([68000], dtype=np.float32)
        dtype = tf.bfloat16

        with self.session(graph=ops.Graph(), use_gpu=True):
            tf_arr = variables.Variable(arr)
            variables.global_variables_initializer().run()
            tf_mean = math_ops.reduce_mean(
                tf.cast(tf_arr, dtype=dtype), 0, False)
            tf_out_mean = self.evaluate(tf_mean)
        self.assertAllClose(tf_out_mean, 1., rtol=1e-2, atol=1e-2)

    @test_util.run_deprecated_v1
    def testMeanBf16(self):
        for rank in range(1, _MAX_RANK + 1):
            np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
            self._compareAllAxes(np_arr)


# TODO(yifeng): Remove the device check after ITEX-CPU SumReduction is supported
class SumReductionPerformanceTest(BaseReductionTest):
    def perf_evaluate(self, tf_mean, ans):
        for i in range(10):
            tf_out_mean = self.evaluate(tf_mean)
        self.assertAllClose(tf_out_mean, ans, rtol=1e-2, atol=1e-2)

    @test_util.run_deprecated_v1
    def testGenericReducerPath(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")

        arr = np.ones([2, 128, 128, 128, 2], dtype=np.float32)
        reduction_axes = [1, 2, 3]
        dtype = tf.bfloat16
        ans = np.ones([2, 2], dtype=np.float32)

        with self.session(graph=ops.Graph(), use_gpu=True):
            tf_arr = variables.Variable(arr)
            variables.global_variables_initializer().run()
            tf_mean = math_ops.reduce_mean(
                tf.cast(tf_arr, dtype=dtype), reduction_axes, False)
            tf_out_mean = self.perf_evaluate(tf_mean, ans)

    @test_util.run_deprecated_v1
    def testFullReductionPath(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")

        arr = np.ones([2, 128, 128, 128, 2], dtype=np.float32)
        reduction_axes = [0, 1, 2, 3, 4]
        dtype = tf.bfloat16

        with self.session(graph=ops.Graph(), use_gpu=True):
            tf_arr = variables.Variable(arr)
            variables.global_variables_initializer().run()
            tf_mean = math_ops.reduce_mean(
                tf.cast(tf_arr, dtype=dtype), reduction_axes, False)
            tf_out_mean = self.perf_evaluate(tf_mean, 1.)

    @test_util.run_deprecated_v1
    def testInnerMostDimsReductionPath(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")

        arr = np.ones([2, 128, 128, 128, 2], dtype=np.float32)
        reduction_axes = [1, 2, 3, 4]
        dtype = tf.bfloat16

        with self.session(graph=ops.Graph(), use_gpu=True):
            tf_arr = variables.Variable(arr)
            variables.global_variables_initializer().run()
            tf_mean = math_ops.reduce_mean(
                tf.cast(tf_arr, dtype=dtype), reduction_axes, False)
            tf_out_mean = self.perf_evaluate(tf_mean, [1., 1.])
       
    @test_util.run_deprecated_v1
    def testPreservingInnerDimsReductionPath(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")

        arr = np.ones([2, 128, 128, 128, 2], dtype=np.float32)
        reduction_axes = [0, 1, 2, 3]
        dtype = tf.bfloat16
        ans = np.ones([2], dtype=np.float32)

        with self.session(graph=ops.Graph(), use_gpu=True):
            tf_arr = variables.Variable(arr)
            variables.global_variables_initializer().run()
            tf_mean = math_ops.reduce_mean(
                tf.cast(tf_arr, dtype=dtype), reduction_axes, False)
            tf_out_mean = self.perf_evaluate(tf_mean, ans)
        

class ProdReductionTest(BaseReductionTest):
    def _tf_reduce(self, x, reduction_axes, keepdims):
        return math_ops.reduce_prod(x, reduction_axes, keepdims)

    def _np_reduce(self, x, reduction_axes, keepdims):
        if isinstance(reduction_axes, list) or isinstance(reduction_axes,
                                                          np.ndarray):
            reduction_axes = tuple(reduction_axes)
        return np.prod(x, axis=reduction_axes, keepdims=keepdims)

    @test_util.run_deprecated_v1
    def testProdBf16(self):
        for rank in range(1, _MAX_RANK + 1):
            np_arr = self._makeIncremental((2,) * rank, dtypes.float32)
            self._compareAllAxes(np_arr)


if __name__ == '__main__':
    test.main()
