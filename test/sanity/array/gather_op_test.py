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
"""Tests for tensorflow.ops.tf.gather."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables

_TEST_TYPES = (dtypes.int64, dtypes.float32)

# TODO(virimia): Add a benchmark for gather_v2, with batch_dims and axis set.


class GatherTest(test.TestCase, parameterized.TestCase):

  def _buildParams(self, data, dtype):
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testSimpleTwoD32(self):
    with self.session(use_gpu=True):
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                       [9, 10, 11], [12, 13, 14]])
      for dtype in _TEST_TYPES:
        for axis in range(data.ndim):
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          # The indices must be in bounds for any axis.
          indices = constant_op.constant([0, 1, 0, 2])
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = self.evaluate(gather_t)
          self.assertAllEqual(np.take(params_np, [0, 1, 0, 2], axis=axis),
                              gather_val)
          expected_shape = data.shape[:axis] + (4,) + data.shape[axis + 1:]
          self.assertEqual(expected_shape, gather_t.get_shape())

  @test_util.run_deprecated_v1
  def testHigherRank(self):
    # We check that scalar and empty indices shapes work as well
    shape = (2, 1, 3, 2)
    for indices_shape in (), (0,), (2, 0), (2, 3):
      for dtype in _TEST_TYPES:
        for axis in range(len(shape)):
          params = self._buildParams(np.random.randn(*shape), dtype)
          indices = np.random.randint(shape[axis], size=indices_shape)
          with self.cached_session(use_gpu=True) as sess:
            tf_params = constant_op.constant(params)
            tf_indices = constant_op.constant(indices)
            # Check that both positive and negative indices for axis work.
            tf_axis = constant_op.constant(axis)
            tf_negative_axis = constant_op.constant(-len(shape) + axis)
            gather = array_ops.gather(tf_params, tf_indices, axis=tf_axis)
            gather_negative_axis = array_ops.gather(
                tf_params, tf_indices, axis=tf_negative_axis)
            gather_value, gather_negative_axis_value = sess.run(
                [gather, gather_negative_axis])
            gather_np = np.take(params, indices, axis)
            self.assertAllEqual(gather_np, gather_value)
            self.assertAllEqual(gather_np, gather_negative_axis_value)
            expected_shape = (params.shape[:axis] + indices.shape +
                              params.shape[axis + 1:])
            self.assertEqual(expected_shape, gather.shape)
            self.assertEqual(expected_shape, gather_negative_axis.shape)

            # Test gradients
            gather_grad = np.random.randn(
                *gather.get_shape().as_list()).astype(dtype.as_numpy_dtype)
            if dtype.is_complex:
              gather_grad -= 1j * gather_grad
            params_grad, indices_grad, axis_grad = gradients_impl.gradients(
                gather, [tf_params, tf_indices, tf_axis], gather_grad)
            self.assertEqual(indices_grad, None)
            self.assertEqual(axis_grad, None)
            if dtype.is_integer:
              self.assertEqual(params_grad, None)
              continue
            # For axis 0, we are able to create an efficient IndexedSlices for
            # the gradient.
            if axis == 0:
              self.assertEqual(type(params_grad), ops.IndexedSlices)
              params_grad = ops.convert_to_tensor(params_grad)
            correct_params_grad = np.zeros(shape).astype(dtype.as_numpy_dtype)
            outer_dims = axis
            inner_dims = len(shape) - axis - 1
            gather_grad = gather_grad.reshape(
                shape[:axis] + (indices.size,) + shape[axis + 1:])
            for source_index, dest_index in enumerate(indices.flat):
              dest_slice = ((slice(None),) * outer_dims + (dest_index,) +
                            (slice(None),) * inner_dims)
              source_slice = ((slice(None),) * outer_dims + (source_index,) +
                              (slice(None),) * inner_dims)
              correct_params_grad[dest_slice] += gather_grad[source_slice]
            self.assertAllClose(
                correct_params_grad,
                self.evaluate(params_grad),
                atol=2e-6,
                rtol=2e-6)

  @parameterized.parameters([
      # batch_dims=0 (equivalent to tf.gather)
      dict(  # 2D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[2, 1], [0, 3]],
          expected=[[8, 7], [6, 9]]),
      dict(  # 3D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[9, 7], [8, 6]], [[6, 9], [8, 8]]]),
      dict(  # 4D indices
          batch_dims=0,
          params=[8, 9],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[8, 9], [9, 8]], [[8, 8], [9, 9]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),

      # batch_dims=indices.shape.ndims - 1
      # (equivalent to tf.compat.v1.batch_gather)
      dict(  # 2D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=2,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),
      dict(  # 2D indices (1 batch dim)
          batch_dims=-1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=-1,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),

      # 0 < batch_dims < indices.shape.ndims - 1
      dict(  # 3D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[13, 11], [12, 10]], [[20, 23], [22, 22]]]),
      dict(  # 4D indices (1 batch dim)
          batch_dims=1,
          params=[[6, 7], [8, 9]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[6, 7], [7, 6]], [[6, 6], [7, 7]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),
      dict(  # 4D indices (2 batch dims)
          batch_dims=2,
          params=[[[2, 3], [4, 5]], [[6, 7], [8, 9]]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[2, 3], [3, 2]], [[4, 4], [5, 5]]],
                    [[[7, 7], [6, 6]], [[8, 9], [9, 8]]]]),

      # axis > 0
      dict(  # 3D indices, batch_dims=1, axis=2
          # params.shape  = [I1, J1, J2] = [2, 2, 3]
          # indices.shape = [I1, K1, K2] = [2, 1, 5]
          # result.shape  = [I1, J1, K1, K2] = [2, 2, 1, 5]
          batch_dims=1,
          axis=2,
          params=[[[10, 11, 12], [13, 14, 15]], [[20, 21, 22], [23, 24, 25]]],
          indices=[[[0, 1, 2, 1, 0]], [[0, 1, 2, 1, 0]]],
          expected=[[[[10, 11, 12, 11, 10]], [[13, 14, 15, 14, 13]]],
                    [[[20, 21, 22, 21, 20]], [[23, 24, 25, 24, 23]]]]),
      dict(  # 3D indices, batch_dims=None, axis=1
          batch_dims=None,
          axis=1,
          params=[[10, 11, 12], [13, 14, 15]],
          indices=[1, 0],
          expected=[[11, 10], [14, 13]]),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testBatchDims(self, params, indices, batch_dims, expected=None,
                    axis=None):
    result = array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    self.assertAllEqual(expected, result)

    with compat.forward_compatibility_horizon(2019, 6, 11):
      result = array_ops.gather(
          params, indices, axis=axis, batch_dims=batch_dims)

    self.assertAllEqual(expected, result)

  @test_util.run_v1_only("RefVariable is not supported in v2")
  def testGatherRefVariable(self):
    with self.cached_session():
      v = variables.RefVariable(constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather(v, [0, 2])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("GatherV2", gather.op.name)
      self.assertAllEqual([[1, 2], [5, 6]], gather)

  @test_util.run_in_graph_and_eager_modes
  def testGatherResourceVariable(self):
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant([[1, 2], [3, 4], [5, 6]]))
      self.evaluate(variables.global_variables_initializer())
      gather = array_ops.gather(v, [0, 2])
      if not context.executing_eagerly():  # .op doesn't make sense in Eager
        self.assertEqual("ResourceGather", gather.op.inputs[0].op.type)
      self.assertAllEqual([[1, 2], [5, 6]], gather)

if __name__ == "__main__":
  test.main()
