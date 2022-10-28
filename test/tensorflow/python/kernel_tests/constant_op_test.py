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
"""Tests for ConstantOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat

class ZerosLikeTest(test.TestCase):

  def _compareZeros(self, dtype, fully_defined_shape, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      # Creates a tensor of non-zero values with shape 2 x 3.
      # NOTE(kearnes): The default numpy dtype associated with tf.string is
      # np.object (and can't be changed without breaking a lot things), which
      # causes a TypeError in constant_op.constant below. Here we catch the
      # special case of tf.string and set the numpy dtype appropriately.
      if dtype == dtypes_lib.string:
        numpy_dtype = np.string_
      else:
        numpy_dtype = dtype.as_numpy_dtype
      if fully_defined_shape:
        d = constant_op.constant(
            np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      else:
        d = array_ops.placeholder(dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = array_ops.zeros_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      # Test that the shape is correct
      if fully_defined_shape:
        self.assertEqual([2, 3], z_var.get_shape())

      # Test that the value is correct
      feed_dict = {}
      if not fully_defined_shape:
        feed_dict[d] = np.ones((2, 3), dtype=numpy_dtype)
      z_value = z_var.eval(feed_dict=feed_dict)
      self.assertFalse(np.any(z_value))
      self.assertEqual((2, 3), z_value.shape)

  @test_util.run_deprecated_v1
  def testZerosLikeCPU(self):
    for dtype in [
        dtypes_lib.half, dtypes_lib.float32, dtypes_lib.float64,
        dtypes_lib.int8, dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.uint16,
        dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.bool,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.string
    ]:
      self._compareZeros(dtype, fully_defined_shape=False, use_gpu=False)
      self._compareZeros(dtype, fully_defined_shape=True, use_gpu=False)

  @test_util.run_deprecated_v1
  def testZerosLikeGPU(self):
    for dtype in [
        dtypes_lib.half, dtypes_lib.float32, dtypes_lib.float64,
        dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.complex64,
        dtypes_lib.complex128, dtypes_lib.bool
    ]:
      self._compareZeros(dtype, fully_defined_shape=False, use_gpu=True)
      self._compareZeros(dtype, fully_defined_shape=True, use_gpu=True)

  @test_util.run_deprecated_v1
  def testZerosLikePartialShape(self):
    d = array_ops.placeholder(dtypes_lib.float32, shape=[None, 4, None])
    z = array_ops.zeros_like(d)
    self.assertEqual(d.get_shape().as_list(), z.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testZerosLikeDtype(self):
    # Make sure zeros_like works even for dtypes that cannot be cast between
    with self.cached_session():
      shape = (3, 5)
      dtypes = np.float32, np.complex64
      for in_type in dtypes:
        x = np.arange(15).astype(in_type).reshape(*shape)
        for out_type in dtypes:
          y = array_ops.zeros_like(x, dtype=out_type).eval()
          self.assertEqual(y.dtype, out_type)
          self.assertEqual(y.shape, shape)
          self.assertAllEqual(y, np.zeros(shape, dtype=out_type))

  @test_util.run_deprecated_v1
  def testZerosLikeVariant(self):
    # TODO(ebrevdo): Re-enable use_gpu=True once non-DMA Variant
    # copying between CPU and GPU is supported AND we register a
    # ZerosLike callback for GPU for Variant storing primitive types
    # in variant_op_registry.cc.
    with self.session(use_gpu=False):
      variant_tensor = tensor_pb2.TensorProto(
          dtype=dtypes_lib.variant.as_datatype_enum,
          tensor_shape=tensor_shape.TensorShape([]).as_proto(),
          variant_val=[
              tensor_pb2.VariantTensorDataProto(
                  # Match registration in variant_op_registry.cc
                  type_name=b"int",
                  metadata=np.array(1, dtype=np.int32).tobytes())
          ])
      const_variant = constant_op.constant(variant_tensor)
      zeros_like = array_ops.zeros_like(const_variant)
      zeros_like_op = logging_ops.Print(
          zeros_like, [const_variant, zeros_like],
          message="Variant storing an int, input and output of zeros_like:").op

      # Smoke test -- ensure this executes without trouble.
      # Right now, non-numpy-compatible objects cannot be returned from a
      # session.run call; similarly, objects that can't be converted to
      # native numpy types cannot be passed to ops.convert_to_tensor.
      # TODO(ebrevdo): Add registration mechanism for
      # ops.convert_to_tensor and for session.run output.
      zeros_like_op.run()
class FillTest(test.TestCase):

  def _compare(self, dims, val, np_ans, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = self.evaluate(tf_ans)
    self.assertAllClose(np_ans, out)
    # Fill does not set the shape.
    # self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, dims, val, np_ans):
    self._compare(dims, val, np_ans, False)
    self._compare(dims, val, np_ans, True)

  def testFillFloat(self):
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillDouble(self):
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt32(self):
    np_ans = np.array([[42] * 3] * 2).astype(np.int32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt64(self):
    np_ans = np.array([[-42] * 3] * 2).astype(np.int64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex64(self):
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex128(self):
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex128)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  @test_util.run_deprecated_v1
  def testFillString(self):
    np_ans = np.array([[b"yolo"] * 3] * 2)
    with self.session(use_gpu=False):
      tf_ans = array_ops.fill([2, 3], np_ans[0][0], name="fill").eval()
    self.assertAllEqual(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testFillNegative(self):
    with self.cached_session():
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(ValueError):
          array_ops.fill(shape, 7)

      # Using a placeholder so this won't be caught in static analysis.
      dims = array_ops.placeholder(dtypes_lib.int32)
      fill_t = array_ops.fill(dims, 3.0)
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(errors_impl.InvalidArgumentError):
          fill_t.eval({dims: shape})

  def testFillEmptyInput(self):
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex64)
    with self.cached_session():
      fill = array_ops.fill(np.random.random([0]), np_ans[0][0], name="fill")
      out = self.evaluate(fill)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # Non-vector dimensions.
    with self.assertRaises(ValueError):
      array_ops.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(ValueError):
      array_ops.fill([3, 2], [1.0, 2.0])

    # Partial dimension information.
    f = array_ops.fill(array_ops.placeholder(dtypes_lib.int32, shape=(4,)), 3.0)
    self.assertEqual([None, None, None, None], f.get_shape().as_list())

    f = array_ops.fill(
        [array_ops.placeholder(
            dtypes_lib.int32, shape=()), 17], 1.0)
    self.assertEqual([None, 17], f.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testGradient(self):
    with self.cached_session():
      in_v = constant_op.constant(5.0)
      out_shape = [3, 2]
      out_filled = array_ops.fill(out_shape, in_v)
      err = gradient_checker.compute_gradient_error(in_v, [], out_filled,
                                                    out_shape)
    self.assertLess(err, 1e-3)


if __name__ == "__main__":
  test.main()
