# Copyright (c) 2022 Intel Corporation
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for binary coefficient-wise operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
# _POW = lambda x, y: x**y
# _TRUEDIV = lambda x, y: x / y
# _FLOORDIV = lambda x, y: x // y
# _MOD = lambda x, y: x % y


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), x_values


def _default_tolerance(dtype):
  """Returns a sensible default tolerance for comparing results of a given type.

  Args:
    dtype: A datatype.
  """
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None  # Fail fast for unexpected types


class BinaryOpTest(test.TestCase):

  def _compareCpu(self, x, y, np_func, tf_func, also_compare_variables=False):
    np_ans = np_func(x, y)
    with test_util.force_cpu():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = array_ops.identity(tf_func(inx, iny))
      tf_cpu = self.evaluate(out)
      # Test that the op takes precedence over numpy operators.
      np_left = self.evaluate(tf_func(x, iny))
      np_right = self.evaluate(tf_func(inx, y))

      if also_compare_variables:
        var_x = variables.Variable(x)
        var_y = variables.Variable(y)
        self.evaluate(variables.global_variables_initializer())
        print(type(x), type(y), type(var_x), type(var_y))
        print(type(tf_func(x, var_y)), type(tf_func(var_x, y)))
        np_var_left = self.evaluate(tf_func(x, var_y))
        np_var_right = self.evaluate(tf_func(var_x, y))

    if np_ans.dtype != np.object:
      self.assertAllClose(np_ans, tf_cpu)
      self.assertAllClose(np_ans, np_left)
      self.assertAllClose(np_ans, np_right)
      if also_compare_variables:
        self.assertAllClose(np_ans, np_var_left)
        self.assertAllClose(np_ans, np_var_right)
    self.assertShapeEqual(np_ans, out)

  _GRAD_TOL = {
      dtypes_lib.float16: 1e-3,
      dtypes_lib.float32: 1e-3,
      dtypes_lib.complex64: 1e-2,
      dtypes_lib.float64: 1e-5,
      dtypes_lib.complex128: 1e-4
  }

  def _compareGradientX(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = array_ops.identity(tf_func(inx, iny))
      xs = list(x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, out, zs, x_init_value=x)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = array_ops.identity(tf_func(inxf, inyf))
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, xs, outf, zs, x_init_value=xf, delta=1e-3)
        jacob_n = jacob_n.astype(x.dtype)
      tol = self._GRAD_TOL[dtypes_lib.as_dtype(x.dtype)]
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGradientY(self,
                        x,
                        y,
                        np_func,
                        tf_func,
                        numeric_gradient_type=None):
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = array_ops.identity(tf_func(inx, iny))
      ys = list(np.shape(y))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, ys, out, zs, x_init_value=y)
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = array_ops.identity(tf_func(inxf, inyf))
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, ys, outf, zs, x_init_value=yf)
        jacob_n = jacob_n.astype(x.dtype)
    tol = self._GRAD_TOL[dtypes_lib.as_dtype(x.dtype)]
    self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGpu(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = array_ops.identity(tf_func(inx, iny))
      tf_gpu = self.evaluate(out)
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def _compareGpuBfloat16(self, x, y, np_func, tf_func):
    np_ans = np_func(x, y)
    with self.session(use_gpu=True):
      inx = tf.cast(ops.convert_to_tensor(x), tf.bfloat16)
      iny = tf.cast(ops.convert_to_tensor(y), tf.bfloat16)
      out = tf_func(inx, iny)
      out = tf.cast(out, tf.float32)
      tf_gpu = self.evaluate(out)
    self.assertAllClose(np_ans, tf_gpu, rtol=5e-2, atol=5e-2)
    self.assertShapeEqual(np_ans, out)

  def _compareBoth(self, x, y, np_func, tf_func, also_compare_variables=False):
    # self._compareCpu(x, y, np_func, tf_func, also_compare_variables)
    # if x.dtype in (np.float16, np.float32, np.float64, np.complex64,
    #                np.complex128):
    #   if tf_func not in (_FLOORDIV, math_ops.floordiv, math_ops.zeta,
    #                      math_ops.polygamma):
    #     self._compareGradientX(x, y, np_func, tf_func)
    #     self._compareGradientY(x, y, np_func, tf_func)
    #   if tf_func in (math_ops.zeta, math_ops.polygamma):
    #     # These methods only support gradients in the second parameter
    #     self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  @test_util.run_deprecated_v1
  def testFloatBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)
    self._compareBoth(x, y, np.add, math_ops.add, also_compare_variables=True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    x1 = np.random.randn(5, 6).astype(np.float32)
    x2 = np.random.randn(5, 6).astype(np.float32)
    # Remove tiny values--atan2 gradients are flaky near the origin.
    x1[np.abs(x1) < 0.05] = 0.05 * np.sign(x1[np.abs(x1) < 0.05])
    x2[np.abs(x2) < 0.05] = 0.05 * np.sign(x2[np.abs(x2) < 0.05])
    self._compareBoth(x1, x2, np.arctan2, math_ops.atan2)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      a_pos_small = np.linspace(0.1, 2, 15).reshape(1, 3, 5).astype(np.float32)
      x_pos_small = np.linspace(0.1, 10, 15).reshape(1, 3, 5).astype(np.float32)
      self._compareBoth(a_pos_small, x_pos_small, special.gammainc,
                        math_ops.igamma)
      self._compareBoth(a_pos_small, x_pos_small, special.gammaincc,
                        math_ops.igammac)
      # Need x > 1
      self._compareBoth(x_pos_small + 1, a_pos_small, special.zeta,
                        math_ops.zeta)
      n_small = np.arange(0, 15).reshape(1, 3, 5).astype(np.float32)
      self._compareBoth(n_small, x_pos_small, special.polygamma,
                        math_ops.polygamma)
    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

  @test_util.run_deprecated_v1
  def testDoubleBasic(self):
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.double)
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.double)
    self._compareBoth(x, y, np.add, math_ops.add, also_compare_variables=True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    x1 = np.random.randn(5, 6).astype(np.double)
    x2 = np.random.randn(5, 6).astype(np.double)
    # Remove tiny values--atan2 gradients are flaky near the origin.
    x1[np.abs(x1) < 0.05] = 0.05 * np.sign(x1[np.abs(x1) < 0.05])
    x2[np.abs(x2) < 0.05] = 0.05 * np.sign(x2[np.abs(x2) < 0.05])
    self._compareBoth(x1, x2, np.arctan2, math_ops.atan2)
    try:
      from scipy import special  # pylint: disable=g-import-not-at-top
      a_pos_small = np.linspace(0.1, 2, 15).reshape(1, 3, 5).astype(np.double)
      x_pos_small = np.linspace(0.1, 10, 15).reshape(1, 3, 5).astype(np.double)
      self._compareBoth(a_pos_small, x_pos_small, special.gammainc,
                        math_ops.igamma)
      self._compareBoth(a_pos_small, x_pos_small, special.gammaincc,
                        math_ops.igammac)
      # Need x > 1
      self._compareBoth(x_pos_small + 1, a_pos_small, special.zeta,
                        math_ops.zeta)
      n_small = np.arange(0, 15).reshape(1, 3, 5).astype(np.double)
      self._compareBoth(n_small, x_pos_small, special.polygamma,
                        math_ops.polygamma)
    except ImportError as e:
      tf_logging.warn("Cannot test special functions: %s" % str(e))

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
    y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype in (np.float16, np.float32):
      # if x.dtype == np.float16:
      #   # Compare fp16 theoretical gradients to fp32 numerical gradients,
      #   # since fp16 numerical gradients are too imprecise unless great
      #   # care is taken with choosing the inputs and the delta. This is
      #   # a weaker check (in particular, it does not test the op itself,
      #   # only its gradient), but it's much better than nothing.
      #   self._compareGradientX(x, y, np_func, tf_func, np.float)
      #   self._compareGradientY(x, y, np_func, tf_func, np.float)
      # else:
      #   self._compareGradientX(x, y, np_func, tf_func)
      #   self._compareGradientY(x, y, np_func, tf_func)
      self._compareGpu(x, y, np_func, tf_func)

  def _compareBCastBfloat16(self, xs, ys, dtype, np_func, tf_func):
    x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
    y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    self._compareGpuBfloat16(x, y, np_func, tf_func)      

  def _testBCastByFunc(self, funcs, xs, ys):
    dtypes = [
        np.float16,
        np.float32,
    ]
    for dtype in dtypes:
      for (np_func, tf_func) in funcs:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)
        self._compareBCast(ys, xs, dtype, np_func, tf_func)
    for (np_func, tf_func) in funcs:  
      self._compareBCastBfloat16(xs, ys, np.float32, np_func, tf_func);

  def _testBCastA(self, xs, ys):
    funcs = [
        (np.add, math_ops.add),
        (np.add, _ADD),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastB(self, xs, ys):
    funcs = [
        (np.subtract, math_ops.subtract),
        (np.subtract, _SUB),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastC(self, xs, ys):
    funcs = [
        (np.multiply, math_ops.multiply),
        (np.multiply, _MUL),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastD(self, xs, ys):
    # funcs = [
    #     (np.true_divide, math_ops.truediv),
    #     (np.floor_divide, math_ops.floordiv),
    #     (np.true_divide, _TRUEDIV),
    #     (np.floor_divide, _FLOORDIV),
    # ]
    # self._testBCastByFunc(funcs, xs, ys)
    return

  @test_util.run_deprecated_v1
  def testBCast_0A(self):
    self._testBCastA([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_0B(self):
    self._testBCastB([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_0C(self):
    self._testBCastC([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_0D(self):
    self._testBCastD([1, 3, 2], [1])

  @test_util.run_deprecated_v1
  def testBCast_1A(self):
    self._testBCastA([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_1B(self):
    self._testBCastB([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_1C(self):
    self._testBCastC([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_1D(self):
    self._testBCastD([1, 3, 2], [2])

  @test_util.run_deprecated_v1
  def testBCast_2A(self):
    self._testBCastA([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_2B(self):
    self._testBCastB([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_2C(self):
    self._testBCastC([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_2D(self):
    self._testBCastD([1, 3, 2], [3, 2])

  @test_util.run_deprecated_v1
  def testBCast_3A(self):
    self._testBCastA([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_3B(self):
    self._testBCastB([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_3C(self):
    self._testBCastC([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_3D(self):
    self._testBCastD([1, 3, 2], [3, 1])

  @test_util.run_deprecated_v1
  def testBCast_4A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])
  
  @test_util.run_deprecated_v1
  def testBCast_4B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_4C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_4D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_5A(self):
    self._testBCastA([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_5B(self):
    self._testBCastB([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_5C(self):
    self._testBCastC([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_5D(self):
    self._testBCastD([1, 3, 2], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_6A(self):
    self._testBCastA([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_6B(self):
    self._testBCastB([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_6C(self):
    self._testBCastC([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_6D(self):
    self._testBCastD([1, 3, 2], [2, 1, 1])

  @test_util.run_deprecated_v1
  def testBCast_7A(self):
    self._testBCastA([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_7B(self):
    self._testBCastB([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_7C(self):
    self._testBCastC([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_7D(self):
    self._testBCastD([1, 3, 2], [1, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8A(self):
    self._testBCastA([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8B(self):
    self._testBCastB([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8C(self):
    self._testBCastC([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_8D(self):
    self._testBCastD([2, 1, 5], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_9A(self):
    self._testBCastA([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_9B(self):
    self._testBCastB([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_9C(self):
    self._testBCastC([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_9D(self):
    self._testBCastD([2, 0, 5], [2, 0, 1])

  @test_util.run_deprecated_v1
  def testBCast_10A(self):
    self._testBCastA([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_10B(self):
    self._testBCastB([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_10C(self):
    self._testBCastC([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_10D(self):
    self._testBCastD([2, 3, 0], [2, 3, 1])

  @test_util.run_deprecated_v1
  def testBCast_11A(self):
    self._testBCastA([1, 3, 2], [1, 3, 2])
  
  @test_util.run_deprecated_v1
  def testBCast_11B(self):
    self._testBCastB([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_11C(self):
    self._testBCastC([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_11D(self):
    self._testBCastD([1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12A(self):
    self._testBCastA([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12B(self):
    self._testBCastB([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12C(self):
    self._testBCastC([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_12D(self):
    self._testBCastD([1, 1, 1, 1, 3, 2], [1, 3, 2])

  @test_util.run_deprecated_v1
  def testBCast_13A(self):
    self._testBCastA([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_13B(self):
    self._testBCastB([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_13C(self):
    self._testBCastC([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_13D(self):
    self._testBCastD([1, 3, 2, 1, 1], [1])

  @test_util.run_deprecated_v1
  def testBCast_14A(self):
    self._testBCastA([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_14B(self):
    self._testBCastB([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_14C(self):
    self._testBCastC([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_14D(self):
    self._testBCastD([2, 3, 1, 1, 5], [1])

  @test_util.run_deprecated_v1
  def testBCast_15A(self):
    self._testBCastA([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testBCast_15B(self):
    self._testBCastB([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testBCast_15C(self):
    self._testBCastC([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testBCast_15D(self):
    self._testBCastD([10, 3, 1, 2], [3, 1, 2])

  @test_util.run_deprecated_v1
  def testMismatchedDimensions(self):
    for func in [
        math_ops.add, math_ops.multiply, _ADD, _MUL,
        math_ops.subtract, _SUB
    ]:
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Dimensions must" in str(e)):
        func(
            ops.convert_to_tensor([10.0, 20.0, 30.0]),
            ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))


if __name__ == "__main__":
  test.main()
