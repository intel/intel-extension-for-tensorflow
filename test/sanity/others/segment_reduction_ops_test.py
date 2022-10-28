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
"""Functional tests for segment reduction ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import itertools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

class SegmentReductionHelper(test.TestCase):

  def _input(self, input_shape, dtype=dtypes_lib.int32):
    num_elem = 1
    for x in input_shape:
      num_elem *= x
    values = np.arange(1, num_elem + 1)
    np_values = values.reshape(input_shape).astype(dtype.as_numpy_dtype)
    # Add a non-zero imaginary component to complex types.
    if dtype.is_complex:
      np_values -= 1j * np_values
    return constant_op.constant(
        np_values, shape=input_shape, dtype=dtype), np_values

  def _segmentReduce(self, indices, x, op1, op2=None, num_segments=None,
                     initial_value=0):
    if not x.size:
      return np.array([])
    indices = np.asarray(indices)
    if num_segments is None:
      num_segments = indices[-1] + 1
    output = [None] * num_segments
    slice_shape = x.shape[indices.ndim:]
    x_flat = x.reshape((indices.size,) + slice_shape)
    for i, index in enumerate(indices.ravel()):
      if (output[index] is not None) and op1 == np.max:
        for j in range(0, output[index].shape[0]):
          output[index][j] = op1([output[index][j], x_flat[i][j]])
      elif output[index] is not None:
        output[index] = op1(output[index], x_flat[i])
      else:
        output[index] = x_flat[i]
    # zero initialize values that are still uncalcuated.
    initial_value_slice = np.ones(slice_shape) * initial_value
    output = [o if o is not None else initial_value_slice for o in output]
    if op2 is not None:
      output = [op2(o) for o in output]
    output = [o.reshape(slice_shape) for o in output]
    return np.array(output)

  def _mean_cum_op(self, x, y):
    return (x[0] + y, x[1] + 1) if isinstance(x, tuple) else (x + y, 2)

  def _mean_reduce_op(self, x):
    return x[0] / x[1] if isinstance(x, tuple) else x

  def _sqrt_n_reduce_op(self, x):
    return x[0] / np.sqrt(x[1]) if isinstance(x, tuple) else x


class UnsortedSegmentTest(SegmentReductionHelper):

  def __init__(self, methodName='runTest'):
    # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.unsorted_segment_sum, lambda t: 0),
                     (self._mean_cum_op, self._mean_reduce_op,
                      math_ops.unsorted_segment_mean, lambda t: 0),
                     (self._mean_cum_op, self._sqrt_n_reduce_op,
                      math_ops.unsorted_segment_sqrt_n, lambda t: 0),
                     (np.ndarray.__mul__, None,
                      math_ops.unsorted_segment_prod, lambda t: 1),
                     (np.minimum, None,
                      math_ops.unsorted_segment_min, lambda t: t.max),
                     (np.maximum, None,
                      math_ops.unsorted_segment_max, lambda t: t.min)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.unsorted_segment_sum, lambda t: 0),
                             (np.ndarray.__mul__, None,
                              math_ops.unsorted_segment_prod, lambda t: 1)]
    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32,
                                  dtypes_lib.float64]
    self.all_dtypes = (self.differentiable_dtypes +
                       [dtypes_lib.bfloat16,
                        dtypes_lib.int64, dtypes_lib.int32,
                        dtypes_lib.complex64, dtypes_lib.complex128])
    super(UnsortedSegmentTest, self).__init__(methodName=methodName)

  def testValues(self):
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in self.all_dtypes:
        ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
        tf_x, np_x = self._input(shape, dtype=dtype)
        for use_gpu in [True, False]:
          with self.cached_session(use_gpu=True):
            for np_op1, np_op2, tf_op, init_op in ops_list:
              # sqrt_n doesn't support integers
              if (np_op2 == self._sqrt_n_reduce_op and dtype.is_integer):
                continue
              # todo(philjd): enable this test once real_div supports bfloat16
              if (np_op2 in [self._sqrt_n_reduce_op, self._mean_reduce_op] and
                  dtype == dtypes_lib.bfloat16):
                continue
              np_ans = self._segmentReduce(
                  indices, np_x, np_op1, np_op2, num_segments=num_segments,
                  initial_value=init_op(dtype))
              s = tf_op(tf_x, segment_ids=indices, num_segments=num_segments)
              tf_ans = self.evaluate(s)
              if dtype is dtypes_lib.bfloat16:
                tf_ans = tf_ans.astype(np.float32)
              self.assertAllCloseAccordingToType(np_ans, tf_ans)
              self.assertShapeEqual(np_ans, s)

if __name__ == "__main__":
  test.main()
