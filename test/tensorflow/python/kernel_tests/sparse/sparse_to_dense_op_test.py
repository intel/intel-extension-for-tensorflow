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
"""Tests for tensorflow.kernels.sparse_op."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

@test_util.with_eager_op_as_function
class SparseToDenseTest(test.TestCase, parameterized.TestCase):

  def testInt(self):
    with self.session(use_gpu=True):
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], 1, 0)
        np_ans = np.array([0, 1, 0, 1, 0]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

  @parameterized.parameters(
      dtypes.bfloat16, dtypes.float16, dtypes.float32, dtypes.float64
  )
  def testFloatTypes(self, dtype):
    with self.session(use_gpu=True):
        tf_ans = sparse_ops.sparse_to_dense(
            [1, 3], [5], array_ops.constant(1.0, dtype=dtype), 0.0
        )
        np_ans = np.array([0, 1, 0, 1, 0]).astype(dtype.as_numpy_dtype)
        self.assertAllClose(np_ans, tf_ans)

  def testComplex(self):
    with self.session(use_gpu=True):
        for dtype in [dtypes.complex64, dtypes.complex128]:
            tf_val = math_ops.cast(
                constant_op.constant([1.0 + 1.0j, 2.0 - 2.0j]), dtypes.complex128)
            tf_ans = sparse_ops.sparse_tensor_to_dense(sparse_ops.from_dense(tf_val))
        self.assertAllClose(tf_val, tf_ans)

  def testEmptyNonZeros(self):
    with self.session(use_gpu=True):
        indices = array_ops.constant([], dtype=dtypes.int32)
        values = array_ops.constant([], dtype=dtypes.float32)
        tf_ans = sparse_ops.sparse_to_dense(indices, [5], values, 0.0)
        np_ans = np.array([0, 0, 0, 0, 0]).astype(np.float32)
        self.assertAllClose(np_ans, tf_ans)

  def testString(self):
    with self.session(use_gpu=True):
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], "a", "b")
        np_ans = np.array(["b", "a", "b", "a", "b"]).astype(np.string_)
        self.assertAllEqual(np_ans, tf_ans)

  def testSetValue(self):
    with self.session(use_gpu=True):
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], [1, 2], -1)
        np_ans = np.array([-1, 1, -1, 2, -1]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

  def testSetSingleValue(self):
    with self.session(use_gpu=True):
        tf_ans = sparse_ops.sparse_to_dense([1, 3], [5], 1, -1)
        np_ans = np.array([-1, 1, -1, 1, -1]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

  def test2d(self):
    with self.session(use_gpu=True):
        tf_ans = sparse_ops.sparse_to_dense([[1, 3], [2, 0]], [3, 4], 1, -1)
        np_ans = np.array([[-1, -1, -1, -1],
                        [-1, -1, -1, 1],
                        [1, -1, -1, -1]]).astype(np.int32)
        self.assertAllClose(np_ans, tf_ans)

  def testZeroDefault(self):
    with self.session(use_gpu=True):
        x = sparse_ops.sparse_to_dense(2, [4], 7)
        self.assertAllEqual(x, [0, 0, 7, 0])

  def test3d(self):
    with self.session(use_gpu=True):
        tf_ans = sparse_ops.sparse_to_dense([[1, 3, 0], [2, 0, 1]], [3, 4, 2], 1,
                                            -1)
        np_ans = np.ones((3, 4, 2), dtype=np.int32) * -1
        np_ans[1, 3, 0] = 1
        np_ans[2, 0, 1] = 1
        self.assertAllClose(np_ans, tf_ans)
  def testBadShape(self):
    with self.session(use_gpu=True):
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                    "must be rank 1"):
            sparse_ops.sparse_to_dense([1, 3], [[5], [3]], 1, -1)

  def testBadValue(self):
    with self.session(use_gpu=True):
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                    r"sparse_values has incorrect shape \[2,1\], "
                                    r"should be \[\] or \[2\]"):
            self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [[5], [3]], -1))

  def testBadNumValues(self):
    with self.session(use_gpu=True):
        with self.assertRaisesRegex(
            (ValueError, errors.InvalidArgumentError),
            r"sparse_values has incorrect shape \[3\], should be \[\] or \[2\]"):
            self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [1, 2, 3], -1))

  def testBadDefault(self):
    with self.session(use_gpu=True):
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                    "default_value should be a scalar"):
            self.evaluate(sparse_ops.sparse_to_dense([1, 3], [5], [1, 2], [0]))

  def testShapeInferenceKnownShape(self):
    with self.session(use_gpu=True):
        with ops.Graph().as_default():
            indices = array_ops.placeholder(dtypes.int64)

            shape = [4, 5, 6]
            output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
            self.assertEqual(output.get_shape(), [4, 5, 6])

            shape = array_ops.placeholder(dtypes.int64, shape=(3,))
            output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
            self.assertEqual(output.get_shape().as_list(), [None, None, None])

  def testShapeInferenceUnknownShape(self):
    with self.session(use_gpu=True):
        with ops.Graph().as_default():
            indices = array_ops.placeholder(dtypes.int64)
            shape = array_ops.placeholder(dtypes.int64)
            output = sparse_ops.sparse_to_dense(indices, shape, 1, 0)
            self.assertIsNone(output.get_shape().ndims)


if __name__ == "__main__":
  test.main()