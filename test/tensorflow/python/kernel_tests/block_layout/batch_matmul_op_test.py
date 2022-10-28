# Copyright (c) 2022 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# 
#Licensed under the Apache License, Version 2.0 (the "License");
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
"""Tests for tensorflow.ops.tf.BatchMatMul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark


def GetRandomNormalInput(shape, dtype):
  # float16 has limited range so we reduce the variance of the scalars.
  scale = 10.0 if dtype != np.float16 else 0.1
  loc = -10.0 if dtype != np.float16 else 0.1
  vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
  if dtype in (np.complex64, np.complex128):
    imag = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    vals += 1j * imag
  return vals.reshape(shape)

@test_util.run_all_in_native_and_block_format
class BatchMatmulOpTest(test.TestCase):

  # Uses numpy to compute batch_matmul(x, y, adjoint_a, adjoint_b).
  def _npBatchMatmul(self, x, y, adjoint_a, adjoint_b):
    # output's shape depends on adj[0] and adj[1]
    if adjoint_a:
      x = np.conjugate(np.swapaxes(x, -1, -2))
    if adjoint_b:
      y = np.conjugate(np.swapaxes(y, -1, -2))
    return np.matmul(x, y)

  # Compares TensorFlow BatchMatmul with NumPy's matmul.
  @test_util.run_deprecated_v1
  def _compare(self, x_in, y_in, adjoint_a, adjoint_b):
    x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
    y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
    x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
    y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
    is_floating = x.dtype != np.int32
    tol = 100 * np.finfo(x.dtype).eps if is_floating else 0
    with self.cached_session(use_gpu=is_floating) as sess:
      x_ph = array_ops.placeholder(x.dtype)
      y_ph = array_ops.placeholder(y.dtype)
      z0 = array_ops.identity(math_ops.matmul(
          x_ph, y_ph, adjoint_a=adjoint_a, adjoint_b=adjoint_b))
      z0_val = sess.run(z0, feed_dict={x_ph: x, y_ph: y})
      z1 = self._npBatchMatmul(x, y, adjoint_a, adjoint_b)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def _testNonEmpty(self, dtype, adjoint_a, adjoint_b):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b)

    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
    CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
    CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
    CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
    CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])
    CompareNonEmpty(self, [5, 7, 2, 3], [5, 7, 3, 5])

  @test_util.run_deprecated_v1
  def _testBroadcasting(self, dtype, adjoint_a, adjoint_b):

    def CompareNonEmpty(self, a_shape, b_shape):
      self._compare(
          GetRandomNormalInput(a_shape, dtype),
          GetRandomNormalInput(b_shape, dtype),
          adjoint_a,
          adjoint_b)

    CompareNonEmpty(self, [2, 3], [1, 3, 5])
    CompareNonEmpty(self, [1, 2, 3], [3, 5])
    CompareNonEmpty(self, [5, 1, 2, 3], [1, 7, 3, 5])
    CompareNonEmpty(self, [5, 2, 2, 3], [3, 5])
    CompareNonEmpty(self, [2, 3], [5, 2, 3, 5])
    CompareNonEmpty(self, [4, 5, 1, 2, 3], [1, 1, 3, 5])
    CompareNonEmpty(self, [1, 2, 1, 4, 2, 1, 3, 4], [3, 2, 1, 1, 1, 2, 4, 2])

  @test_util.run_deprecated_v1
  def _testEmpty(self, dtype, adjoint_a, adjoint_b):

    def CompareEmpty(self, a_shape, b_shape):
      self._compare(
          np.zeros(a_shape).astype(dtype),
          np.zeros(b_shape).astype(dtype),
          adjoint_a,
          adjoint_b)

    CompareEmpty(self, [0, 3, 2], [0, 2, 4])
    CompareEmpty(self, [3, 0, 2], [3, 2, 5])
    CompareEmpty(self, [3, 3, 2], [3, 2, 0])

  def _CreateNumpyTensor(self, shape):
    total_size = 1
    for s in shape:
      total_size *= s
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  @test_util.run_deprecated_v1
  def _testBlocked(self, dtype):

    def CompareBlocked(self):
      tensor_in_sizes = [2, 6, 7, 1]
      filter_in_sizes = [3, 3, 1, 1]
      x1 = self._CreateNumpyTensor(tensor_in_sizes)
      x2 = self._CreateNumpyTensor(filter_in_sizes)
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      phl_in = GetRandomNormalInput([2, 4, 3, 5], dtype)
      phr_in = GetRandomNormalInput([2, 4, 1, 6], dtype)

      with self.cached_session() as sess:
        phl = array_ops.placeholder(np.float32)
        phr = array_ops.placeholder(np.float32)
        conv1 = array_ops.identity(nn_ops.conv2d(
            t1,
            t2,
            dilations=(1, 1),
            strides=[1, 1],
            padding="VALID",
            data_format="NHWC"))
        matmul_l = array_ops.identity(math_ops.matmul(
            phl, conv1, adjoint_a=False, adjoint_b=False))
        matmul_r = array_ops.identity(math_ops.matmul(
            conv1, phr, adjoint_a=False, adjoint_b=False))
        mml_out, mmr_out = sess.run([matmul_l, matmul_r],
            feed_dict={phl: phl_in, phr: phr_in})

      self.assertEqual(mml_out.shape, (2, 4, 3, 1))
      self.assertEqual(mmr_out.shape, (2, 4, 5, 6))

      with self.cached_session() as sess:
        conv2 = array_ops.identity(nn_ops.conv2d(
            t1,
            t2,
            dilations=(1, 1),
            strides=[1, 1],
            padding="VALID",
            data_format="NHWC"))
        conv_out = sess.run(conv2)

      gtl = np.matmul(phl_in, conv_out)
      gtr = np.matmul(conv_out, phr_in)

      is_floating = dtype != np.int32
      tol = 100 * np.finfo(dtype).eps if is_floating else 0

      self.assertAllClose(mml_out, gtl, rtol=tol, atol=tol)
      self.assertAllClose(mmr_out, gtr, rtol=tol, atol=tol)

    CompareBlocked(self)

@test_util.run_deprecated_v1
def _GetBatchMatmulOpTest(dtype, adjoint_a, adjoint_b):

  @test_util.run_without_tensor_float_32("Tests batch matmul")
  def Test(self):
    np.random.seed(42)
    self._testNonEmpty(dtype, adjoint_a, adjoint_b)
    self._testEmpty(dtype, adjoint_a, adjoint_b)

  return Test


@test_util.run_deprecated_v1
def _GetBatchMatmulOpBroadcastingTest(dtype, adjoint_a, adjoint_b):

  @test_util.run_without_tensor_float_32("Tests batch matmul")
  def Test(self):
    np.random.seed(42)
    self._testBroadcasting(dtype, adjoint_a, adjoint_b)

  return Test


@test_util.run_deprecated_v1
def _GetBatchMatmulOpWithBlockedInputTest(dtype, adjoint_a, adjoint_b):

  @test_util.run_without_tensor_float_32("Tests batch matmul")
  def Test(self):
    np.random.seed(42)
    self._testBlocked(dtype)

  return Test

if __name__ == "__main__":
  dtypes_to_test = [
      np.float16, np.float32
  ]

  for dtype_ in dtypes_to_test:
    for adjoint_a_ in False, True:
      for adjoint_b_ in False, True:
        name = "%s_%s_%s" % (dtype_.__name__, adjoint_a_, adjoint_b_)
        setattr(
            BatchMatmulOpTest,
            "testBatchMatmulOp_" + name,
                _GetBatchMatmulOpTest(dtype_, adjoint_a_, adjoint_b_))

  setattr(BatchMatmulOpTest,
          "testBatchMatMulWithBlockedInput",
          _GetBatchMatmulOpWithBlockedInputTest(np.float32, False, False))

  test.main()
