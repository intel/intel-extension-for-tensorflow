"""Tests for broadcast_to ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from intel_extension_for_tensorflow.python.test_func import test as test_lib

@test_util.run_all_in_graph_and_eager_modes 
class BroadcastToTest(test_util.TensorFlowTestCase):
  def testBroadcastToBasic(self):
    for dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
      with self.session(use_gpu=True):
        x = np.array([1, 2, 3], dtype=dtype)
        v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf, v_np)

  def testBroadcastToString(self):
    with self.session(use_gpu=True):
      x = np.array([b"1", b"2", b"3"])
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToBool(self):
    with self.session(use_gpu=True):
      x = np.array([True, False, True], dtype=np.bool_)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShape(self):
    for input_dim in range(1, 6):
      for output_dim in range(input_dim, 6):
        with self.cached_session(use_gpu=True):
          input_shape = [2] * input_dim
          output_shape = [2] * output_dim
          x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
          v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
          v_np = np.broadcast_to(x, output_shape)
          self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeInnerDim(self):
    input_shape = [2, 1, 3]
    output_shape = [2, 5, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeLargerDim(self):
    input_shape = [2, 1, 3, 2, 2, 2]
    output_shape = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 15, 3, 2, 2, 2]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeLargerDim2(self):
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToScalar(self):
    with self.session(use_gpu=True):
      x = np.array(1, dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastScalarToNonScalar(self):
    with self.session(use_gpu=True):
      x = np.array(1.0, dtype=np.float32)
      v_tf = array_ops.broadcast_to(constant_op.constant(1.0), [2, 3, 4,
                                                                1, 1, 1])
      v_np = np.broadcast_to(x, [2, 3, 4, 1, 1, 1])
      self.assertAllEqual(v_tf, v_np)

  def testBroadcastToShapeTypeAndInference(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session(use_gpu=True):
        x = np.array([1, 2, 3])
        v_tf = array_ops.broadcast_to(
            constant_op.constant(x),
            constant_op.constant([3, 3], dtype=dtype))
        shape = v_tf.get_shape().as_list()
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf, v_np)
        # check shape inference when shape input is constant
        self.assertAllEqual(shape, v_np.shape)

  def testBroadcastToBadOutputShape(self):
    with context.eager_mode():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Unable to broadcast tensor of shape"):
        self.evaluate(
            array_ops.broadcast_to(
                constant_op.constant([0, 1]), constant_op.constant([2, 1])))

  def testGradientForScalar(self):

    def func(x):
      v = array_ops.broadcast_to(x, [2, 4, 3])
      return 2 * v
    
    data_types = [dtypes.float32, dtypes.float64]
    max_errors = [1e-4, 1e-6]
    for data_type, max_error in zip(data_types, max_errors):
      x = constant_op.constant(1, dtype=data_type)
      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [x]))

      self.assertLess(err, max_error)

  def testGradientWithSameRank(self):

    def func(x):
      v = array_ops.broadcast_to(x, [2, 5, 3])
      return 2 * v
    
    data_types = [dtypes.float32, dtypes.float64]
    max_errors = [1e-4, 1e-6]
    for data_type, max_error in zip(data_types, max_errors):
      x = constant_op.constant(np.reshape(np.arange(6), (2, 1, 3)),
                              dtype=data_type)
      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [x], delta=1e-2))

      self.assertLess(err, max_error)

  def testGradientWithIncreasingRank(self):

    def func(x):
      v = array_ops.broadcast_to(x, [5, 2, 3])
      return 2 * v
    
    data_types = [dtypes.float32, dtypes.float64]
    max_errors = [1e-4, 1e-6]
    for data_type, max_error in zip(data_types, max_errors):
      x = constant_op.constant([[1], [2]],
                              dtype=data_type)
      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [x]))

      self.assertLess(err, max_error)

  def testGradientWithBroadcastAllDimensions(self):

    def func(x):
      v = array_ops.broadcast_to(x, [5, 2, 3])
      return 2 * v
    
    data_types = [dtypes.float32, dtypes.float64]
    max_errors = [1e-4, 1e-6]
    for data_type, max_error in zip(data_types, max_errors):
      x = constant_op.constant([1], dtype=data_type)
      with self.cached_session():
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(func, [x]))

      self.assertLess(err, max_error)

  def testGradientWithLargeDim(self):
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    x = constant_op.constant(np.array(np.random.randn(*input_shape),
                                      dtype=np.float32))
    def func(x):
      v = array_ops.broadcast_to(x, output_shape)
      return 2 * v

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(func, [x], delta=1e-2))

    self.assertLess(err, 1e-4)

  def testBroadcastToInvalidShape(self):
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "110,53,104,147,157,123,5,24,188,40,5,2"):
      output_shape = [110, 53, 104, 147, 157, 123, 5, 24, 188, 40, 5, 2]
      x = np.array([1, 2, 3], dtype=np.int32)
      v = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      self.evaluate(v)


if __name__ == "__main__":
  test_lib.main()
