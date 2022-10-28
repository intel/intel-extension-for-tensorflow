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

"""Functional tests for Transpose op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gradient_checker

@test_util.run_all_in_native_and_block_format
class TransposeTest(test.TestCase):

  def _np_transpose(self, x, perm):
    ret = np.copy(x)
    ret = ret.transpose(perm)
    return ret

  @test_util.run_deprecated_v1
  def _compareCpu(self, x, p, conjugate=False):
    if p is None:
      rank = x.ndim
      perm = (rank - 1) - np.arange(rank)
    else:
      perm = p
    np_ans = self._np_transpose(x, perm)
    if conjugate:
      np_ans = np.conj(np_ans)
    with self.cached_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      y = array_ops.identity(array_ops.transpose(inx, p, conjugate=conjugate))
      tf_ans = self.evaluate(y)
      self.assertShapeEqual(np_ans, y)
      self.assertAllEqual(np_ans, tf_ans)

      jacob_t = None
      # Gradient check on CPU.
      xs = list(np.shape(x))
      ys = list(np.shape(tf_ans))
      if x.dtype in [np.float32, np.complex64]:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype in [np.float64, np.complex128]:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  @test_util.run_deprecated_v1
  def _compareGpu(self, x, p, conjugate=False):
    if p is None:
      rank = x.ndim
      perm = (rank - 1) - np.arange(rank)
    else:
      perm = p
    np_ans = self._np_transpose(x, perm)
    if conjugate:
      np_ans = np.conj(np_ans)
    with self.cached_session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      y = array_ops.identity(array_ops.transpose(inx, p, conjugate=conjugate))
      tf_ans = self.evaluate(y)

      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

      jacob_t = None
      # Gradient check on GPU.
      xs = list(np.shape(x))
      ys = list(np.shape(tf_ans))
      if x.dtype == np.float32:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        jacob_t, jacob_n = gradient_checker.compute_gradient(inx, xs, y, ys, x,
                                                             1e-2)
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  @test_util.run_deprecated_v1
  def _compare(self, x, use_gpu=False):
    n = np.ndim(x)
    # generate all permutations of [0, 1, ... n-1] in random order.
    all_perm = np.random.permutation(
        [p for p in itertools.permutations(range(n))]).astype(np.int32)
    cs = [False, True] if x.dtype in [np.complex64, np.complex128] else [False]
    for c in cs:
      for p in all_perm[:2]:
        self._compareCpu(x, p, conjugate=c)
        if use_gpu:
          self._compareGpu(x, p, conjugate=c)
    # Test with an empty permutation
    for c in cs:
      self._compareCpu(x, None, conjugate=c)
      if use_gpu:
        self._compareGpu(x, None, conjugate=c)

  @test_util.run_deprecated_v1
  def _compare_cpu_gpu(self, x):
    n = np.ndim(x)
    # generate all permutation of [0, 1, ... n-1] in random order,
    # choose the first two.
    perms = itertools.permutations(range(n))
    for _ in range(2):
      p = np.random.permutation(next(perms)).astype(np.int32)
      tf_a_cpu, tf_g_cpu = self._compareCpu(x, p)
      tf_a_gpu, tf_g_gpu = self._compareGpu(x, p)
      assert tf_g_cpu is not None
      assert tf_g_gpu is not None
      if x.dtype == np.float32:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-3, 1e-3)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-6, 1e-6)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-6, 1e-6)

  @test_util.run_deprecated_v1
  def _testBoth(self, x):
    self._compare(x, use_gpu=False)
    self._compare(x, use_gpu=True)

  @test_util.run_v1_only("b/120545219")
  def testRank1(self):
    self._compareCpu(np.arange(0., 2), [0])

  @test_util.run_deprecated_v1
  def test1D(self):
    vector = np.arange(0, 2).reshape((1, 1, 1, 2, 1))
    self._compare(vector, use_gpu=False)
    self._compare(vector, use_gpu=True)

  @test_util.run_deprecated_v1
  def test5DGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return
    large_shapes = [[4, 10, 10, 10, 3], [4, 10, 10, 10, 8], [4, 10, 10, 10, 13],
                    [4, 3, 10, 10, 10], [4, 8, 10, 10, 10], [4, 13, 10, 10,
                                                             10]] * 3
    perms = [[0, 4, 1, 2, 3]] * 3 + [[0, 2, 3, 4, 1]] * 3 + [[
        4, 1, 2, 3, 0
    ]] * 6 + [[1, 2, 3, 4, 0]] * 6

    datatypes = [np.int8, np.float16, np.float32, np.float64, np.complex128]
    for datatype in datatypes:
      for input_shape, perm in zip(large_shapes, perms):
        with self.subTest(
            datatype=datatype, input_shape=input_shape, perm=perm):
          total_size = np.prod(input_shape)
          inp = np.arange(
              1, total_size + 1, dtype=datatype).reshape(input_shape)
          np_ans = self._np_transpose(inp, perm)
          with self.cached_session(use_gpu=True):
            inx = ops.convert_to_tensor(inp)
            y = array_ops.identity(array_ops.transpose(inx, perm))
            tf_ans = self.evaluate(y)
          self.assertAllEqual(np_ans, tf_ans)
          self.assertShapeEqual(np_ans, y)

  @test_util.run_deprecated_v1
  def test4DGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return
    large_shapes = [[4, 10, 10, 3], [4, 10, 10, 8], [4, 10, 10, 13],
                    [4, 3, 10, 10], [4, 8, 10, 10], [4, 13, 10, 10]] * 3
    perms = [[0, 3, 1, 2]] * 3 + [[0, 2, 3, 1]] * 3 + [[3, 1, 2, 0]] * 6 + [[
        1, 2, 3, 0
    ]] * 3 + [[2, 3, 0, 1]] * 3

    for input_shape, perm in zip(large_shapes, perms):
      with self.subTest(input_shape=input_shape, perm=perm):
        total_size = np.prod(input_shape)
        inp = np.arange(
            1, total_size + 1, dtype=np.float32).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session(use_gpu=True):
          inx = ops.convert_to_tensor(inp)
          y = array_ops.identity(array_ops.transpose(inx, perm))
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

    # shapes related to Inception (taken from conv_ops_test.py)
    inception_shapes = [[4, 5, 5, 124], [4, 8, 8, 38], [4, 8, 8, 38], [
        4, 8, 8, 204
    ], [4, 8, 8, 44], [4, 8, 8, 204], [4, 8, 8, 204], [4, 8, 8, 204], [
        4, 8, 8, 176
    ], [4, 8, 8, 176], [4, 8, 8, 176], [4, 8, 8, 176], [4, 17, 17, 19], [
        4, 17, 17, 19
    ], [4, 17, 17, 124], [4, 17, 17, 12], [4, 17, 17, 124], [4, 17, 17, 22], [
        4, 17, 17, 19
    ], [4, 17, 17, 19], [4, 17, 17, 121], [4, 17, 17, 121], [4, 17, 17, 22], [
        4, 17, 17, 19
    ], [4, 17, 17, 19], [4, 17, 17, 115], [4, 17, 17, 115], [4, 17, 17, 19], [
        4, 17, 17, 16
    ], [4, 17, 17, 115], [4, 17, 17, 102], [4, 17, 17, 12], [4, 17, 17, 102], [
        4, 17, 17, 12
    ], [4, 17, 17, 102], [4, 17, 17, 12], [4, 17, 17, 76], [4, 17, 17, 12], [
        4, 17, 17, 12
    ], [4, 17, 17, 76], [4, 17, 17, 76], [4, 35, 35, 9], [4, 35, 35, 28], [
        4, 35, 35, 6
    ], [4, 35, 35, 28], [4, 35, 35, 25], [4, 35, 35, 4], [4, 35, 35, 25],
                        [4, 35, 35, 9], [4, 35, 35, 19], [4, 35, 35, 19],
                        [4, 35, 35, 19], [4, 73, 73, 6], [4, 73, 73,
                                                          6], [4, 147, 147, 2]]
    for input_shape in inception_shapes:
      with self.subTest(input_shape=input_shape):
        perm = [0, 3, 1, 2]
        total_size = np.prod(input_shape)
        inp = np.arange(
            1, total_size + 1, dtype=np.float32).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session(use_gpu=True):
          inx = ops.convert_to_tensor(inp)
          y = array_ops.identity(array_ops.transpose(inx, perm))
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

  @test_util.run_deprecated_v1
  def test3DGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    datatypes = [np.int8, np.float16, np.float32, np.float64, np.complex128]
    large_shapes = [[4, 1000, 3], [4, 1000, 8], [4, 1000, 13], [4, 3, 1000],
                    [4, 8, 1000], [4, 13, 1000]] * 3
    perms = [[0, 2, 1]] * 6 + [[2, 1, 0]] * 6 + [[1, 2, 0]] * 3 + [[2, 0, 1]
                                                                  ] * 3
    for datatype in datatypes:
      for input_shape, perm in zip(large_shapes, perms):
        with self.subTest(
            datatype=datatype, input_shape=input_shape, perm=perm):
          total_size = np.prod(input_shape)
          inp = np.arange(
              1, total_size + 1, dtype=datatype).reshape(input_shape)
          np_ans = self._np_transpose(inp, perm)
          with self.cached_session(use_gpu=True):
            inx = ops.convert_to_tensor(inp)
            y = array_ops.identity(array_ops.transpose(inx, perm))
            tf_ans = self.evaluate(y)
          self.assertAllEqual(np_ans, tf_ans)
          self.assertShapeEqual(np_ans, y)

  @test_util.run_deprecated_v1
  def testLargeSizeGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    large_shapes = [[1000000, 31, 3], [3, 1000000, 31], [3, 31, 1000000],
                    [10000, 310, 3], [3, 10000, 310], [3, 310, 10000],
                    [2, 1000, 1000], [1000, 2, 1000], [1000, 1000, 2]]
    perms = [[0, 2, 1]] * 9

    for input_shape, perm in zip(large_shapes, perms):
      with self.subTest(input_shape=input_shape, perm=perm):
        total_size = np.prod(input_shape)
        inp = np.arange(
            1, total_size + 1, dtype=np.float32).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session(use_gpu=True):
          inx = ops.convert_to_tensor(inp)
          y = array_ops.identity(array_ops.transpose(inx, perm))
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

  @test_util.run_deprecated_v1
  def testRandomizedSmallDimLargeSizeGPU(self):
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    # Draw 10 random shapes with large dimension sizes.
    # 40% prob to generate dim[0] size within [1, 2047]
    # 40% prob to generate dim[0] size within [2048, 4095]
    # 20% prob to generate dim[0] size within [4096, 100000]
    # 50% prob to use dim[1] as the small dim (<16)
    num_samples = 10
    total_size = 500000
    small_size_limit = 2048
    large_size_limit = 95905
    small_size_percentage = 0.4
    medium_size_percentage = 0.4
    large_size_percentage = 0.2
    perms = [[0, 2, 1]] * num_samples
    dim_zero_sizes = []
    dim_zero_sizes += list(
        np.random.randint(
            small_size_limit, size=int(small_size_percentage * num_samples)) +
        1)
    dim_zero_sizes += list(
        np.random.randint(
            small_size_limit, size=int(medium_size_percentage * num_samples)) +
        small_size_limit)
    dim_zero_sizes += list(
        np.random.randint(
            large_size_limit, size=int(large_size_percentage * num_samples)) +
        small_size_limit * 2)
    input_shapes = []
    small_dim_limit = 16
    for dim_zero_size in dim_zero_sizes:
      small_dim_size = np.random.randint(small_dim_limit - 1) + 1
      large_dim_size = int(
          total_size / dim_zero_size / small_dim_size) + small_dim_limit
      input_shapes += ([[dim_zero_size, small_dim_size, large_dim_size]]
                       if np.random.randint(2) else
                       [[dim_zero_size, large_dim_size, small_dim_size]])

    for input_shape, perm in zip(input_shapes, perms):
      # generate input data with random ints from 0 to 9.
      with self.subTest(input_shape=input_shape, perm=perm):
        inp = np.random.randint(10, size=input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session(use_gpu=True):
          inx = ops.convert_to_tensor(inp)
          y = array_ops.identity(array_ops.transpose(inx, perm))
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)
        self._ClearCachedSession()

  @test_util.run_v1_only("b/120545219")
  def testNop(self):
    self._compareCpu(np.arange(0, 6).reshape([3, 2]).astype(np.float32), [0, 1])

  @test_util.run_v1_only("b/120545219")
  def testSimple(self):
    self._compareCpu(
        np.arange(0, 8).reshape([2, 4]).astype(np.float32),
        np.array([1, 0]).astype(np.int32))

  @test_util.run_deprecated_v1
  def testPermType(self):
    for perm_dtype in [np.int64, np.int32]:
      with self.subTest(perm_dtype=perm_dtype):
        x = np.arange(0, 8).reshape([2, 4]).astype(np.float32)
        p = np.array([1, 0]).astype(perm_dtype)
        np_ans = np.copy(x).transpose(p)
        with self.cached_session(use_gpu=True):
          inx = ops.convert_to_tensor(x)
          inp = constant_op.constant(p)
          y = array_ops.identity(array_ops.transpose(inx, inp))
          tf_ans = self.evaluate(y)
          self.assertShapeEqual(np_ans, y)
          self.assertAllEqual(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testHalf(self):
    self._compare(np.arange(0, 21).reshape([3, 7]).astype(np.float16))
    self._compare(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float16))
    self._compare(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float16))

  @test_util.run_v1_only("b/120545219")
  def testFloat(self):
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float32))

  @test_util.run_v1_only("b/120545219")
  def testDouble(self):
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float64))

  @test_util.run_v1_only("b/120545219")
  def testComplex64(self):
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 21).reshape([3, 7]).astype(np.complex64))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex64))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex64))

  @test_util.run_v1_only("b/120545219")
  def testComplex128(self):
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 21).reshape([3, 7]).astype(np.complex128))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex128))
    self._testBoth(
        np.complex(1, 2) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex128))

  @test_util.run_deprecated_v1
  def testInt8(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int8))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int8))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int8))

  @test_util.run_deprecated_v1
  def testInt16(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int16))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int16))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int16))

  @test_util.run_deprecated_v1
  def testInt32(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int32))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int32))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int32))

  @test_util.run_deprecated_v1
  def testInt64(self):
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int64))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int64))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int64))

  @test_util.run_deprecated_v1
  def testSubGroupTransposeGPU(self):
    # In current impl for transpose, use oneDNN transpose is dims <=6, otherwise use Eigen impl: shuffle functor
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return
    datatypes = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.complex128]

    MAXDIMS = 0
    large_shapes = [[1 for i in range(MAXDIMS)] for j in range(5)]
    shapes = [[4,1024, 1024], [1024, 1024]] + [[4, 32, 32, 32,128]] * 3
    perms = [[i for i in range(MAXDIMS)] for j in range(5)]
    perms_for_transpose = [[0,2,1], [1,0], [0,1,2,4,3], [0,2,3,4,1], [4,3,2,1,0]]
    for i in range(len(perms_for_transpose)):
      for j in range(len(perms_for_transpose[i])):
        perms_for_transpose[i][j] += MAXDIMS

    for i in range(len(large_shapes)):
      large_shapes[i] += shapes[i]
      perms[i] += perms_for_transpose[i]

    for datatype in datatypes:
      for input_shape, perm in zip(large_shapes, perms):
        with self.subTest(
            datatype=datatype, input_shape=input_shape, perm=perm):
          total_size = np.prod(input_shape)
          inp = np.arange(
              1, total_size + 1, dtype=datatype).reshape(input_shape)
          np_ans = self._np_transpose(inp, perm)
          with self.cached_session(use_gpu=True):
            inx = ops.convert_to_tensor(inp)
            y = array_ops.transpose(inx, perm)
            tf_ans = self.evaluate(y)
          self.assertAllEqual(np_ans, tf_ans)
          self.assertShapeEqual(np_ans, y)

  @test_util.run_v1_only("b/120545219")
  def testTranspose2DAuto(self):
    x_np = [[1, 2, 3], [4, 5, 6]]
    for use_gpu in [False, True]:
      with self.subTest(use_gpu=use_gpu):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = array_ops.identity(array_ops.transpose(x_np)).eval()
          self.assertAllEqual(x_tf, [[1, 4], [2, 5], [3, 6]])

  @test_util.run_v1_only("b/120545219")
  def testSingletonDims(self):
    # A singleton dimension is a dimension i with shape[i] == 1. Such dimensions
    # can be collapsed and expanded using reshape without changing the
    # underlying data storage. If all non-singleton dimensions remain in
    # ascending order, the shuffled singletons will be transposed by a reshape,
    # saving a memory allocation & copy. Since this gets a special code-path in
    # transpose_op.cc, we test that the codepath is exercised and the results
    # are as expected; we do not test that we save the memory allocation and
    # copy here.
    for shape in [[2, 1, 2], [2, 1, 2, 1, 1, 2], [1, 2, 2, 1, 1, 1],
                  [1, 1, 1, 2, 2, 2], [2, 2, 1, 1, 1]]:
      with self.subTest(shape=shape):
        self._compare_cpu_gpu(
            np.arange(np.prod(shape)).reshape(shape).astype(np.float32))

  @test_util.run_v1_only("b/120545219")
  def testTransposeShapes(self):
    self.assertEqual(
        [],
        array_ops.identity(array_ops.transpose(array_ops.placeholder(
            dtypes.int32, shape=[])).get_shape().dims))
    self.assertEqual(
        [100],
        array_ops.identity(array_ops.transpose(array_ops.placeholder(
            dtypes.int32, shape=[100])).get_shape().dims))
    self.assertEqual(
        [37, 100],
        array_ops.identity(array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37])).get_shape().dims))
    self.assertEqual(
        [100, 37],
        array_ops.identity(array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37]), [0, 1]).get_shape().dims))
    self.assertEqual(
        [15, 37, 100],
        array_ops.identity(array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37, 15])).get_shape().dims))
    self.assertEqual(
        [15, 100, 37],
        array_ops.identity(array_ops.transpose(
            array_ops.placeholder(
                dtypes.int32, shape=[100, 37, 15]), [2, 0, 1]).get_shape().dims))
    self.assertEqual(
        tensor_shape.TensorShape(None),
        array_ops.identity(array_ops.transpose(array_ops.placeholder(dtypes.int32)).get_shape()))
    self.assertEqual(
        tensor_shape.TensorShape(None),
        array_ops.identity(array_ops.transpose(array_ops.placeholder(dtypes.int32),
                            [0]).get_shape()))

  @test_util.run_v1_only("b/120545219")
  def testNullTensor(self):
    with self.cached_session():
      x = constant_op.constant([], dtype=dtypes.float32, shape=[1, 4, 0])
      xt = array_ops.identity(array_ops.transpose(x, [0, 2, 1])).eval()
      self.assertAllEqual(xt.shape, (1, 0, 4))

  @test_util.run_v1_only("b/120545219")
  def testScalar(self):
    with self.cached_session():
      x = constant_op.constant(42, dtype=dtypes.float32, shape=[])
      xt = array_ops.identity(array_ops.transpose(x)).eval()
      self.assertAllEqual(xt, x)

  @test_util.run_deprecated_v1
  def _testError(self, x, p, err):
    with self.cached_session():
      with self.assertRaisesOpError(err):
        array_ops.identity(array_ops.transpose(x, p)).eval()

  @test_util.run_v1_only("b/120545219")
  def testError(self):
    with self.assertRaises(ValueError):
      array_ops.identity(array_ops.transpose(
          np.arange(0., 30).reshape([2, 3, 5]), [[0, 1], [2, 3]]))
    with self.assertRaises(ValueError):
      array_ops.identity(array_ops.transpose(np.arange(0., 30).reshape([2, 3, 5]), [0, 1, 3]))
    self._testError(
        np.arange(0., 30).reshape([2, 3, 5]), [0, 1, 1], "2 is missing")

  @test_util.run_deprecated_v1
  def _InitValues(self, sizes):
    """Initializes values for input tensors.

    Args:
      sizes: Tensor dimensions.

    Returns:
      Tensor initialized to values.
    """
    total_size = 1
    for s in sizes:
      total_size *= s
    x = [f * 0.5 for f in range(1, total_size + 1)]
    return constant_op.constant(x, shape=sizes)

  # TODO(itex): Use better way to generate block layout, instead of
  # using convolution as the input.
  @test_util.run_deprecated_v1
  def testBlockNHWC(self):
    # TODO(itex): Try to test both CPU and GPU, when we support them coexistent
    for use_gpu in [True]:
      with self.cached_session(use_gpu=use_gpu):
        np_ans = np.array([[[[1.75, 3.75, 5.75],[7.75, 9.75, 11.75],[13.75, 15.75, 17.75]],
                            [[2.5, 5.5, 8.5], [11.5, 14.5, 17.5], [20.5, 23.5, 26.5]]]])

        perm = [0, 3, 1, 2]
        conv_input_size = [1, 3, 3, 2]
        conv_filter_size = [1, 1, 2, 2]
        conv_input = self._InitValues(conv_input_size)
        conv_filter = self._InitValues(conv_filter_size)
        conv_res = nn_ops.conv2d(conv_input, conv_filter, data_format="NHWC", padding="SAME")
        
        y = array_ops.identity(array_ops.transpose(conv_res, perm))
        tf_ans = self.evaluate(y)

        self.assertAllEqual(np.ravel(np_ans), np.ravel(tf_ans))
        self.assertShapeEqual(np_ans, y)

  # TODO(itex): Use better way to generate block layout, instead of
  # using convolution as the input.
  @test_util.run_deprecated_v1
  def testBlockNCHW(self):
    # TODO(itex): Try to test both CPU and GPU, when we support them coexistent
    for use_gpu in [True]:
      with self.cached_session(use_gpu=use_gpu):
        np_ans = np.array([[[[7.75, 10.5], [8.75, 12.], [9.75, 13.5]],
                            [[10.75, 15.], [11.75, 16.5], [12.75, 18.]],
                            [[13.75, 19.5 ], [14.75, 21.], [15.75, 22.5]]]])

        perm = [0, 2, 3, 1]
        conv_input_size = [1, 2, 3, 3]
        conv_filter_size = [1, 1, 2, 2]
        conv_input = self._InitValues(conv_input_size)
        conv_filter = self._InitValues(conv_filter_size)
        conv_res = nn_ops.conv2d(conv_input, conv_filter, data_format="NCHW", padding="SAME")
        
        y = array_ops.identity(array_ops.transpose(conv_res, perm))
        tf_ans = self.evaluate(y)
        
        self.assertAllEqual(np.ravel(np_ans), np.ravel(tf_ans))
        self.assertShapeEqual(np_ans, y)

if __name__ == "__main__":
  test.main()
