# Copyright (c) 2022 Intel Corporation
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for 3d convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import math

import numpy as np
import tensorflow as tf

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gradients
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.eager import context

tf.compat.v1.disable_eager_execution()
def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  # FIXME(itex): CPU doesn't support NCDHW Conv3DGrad now.
  # return [("NDHWC", True), ("NCDHW", True)]
  return [("NCDHW", test.is_gpu_available())]


@test_util.run_all_without_tensor_float_32(
    "Tests Conv3d, which in some cases is implemented with a matmul. With "
    "TensorFloat-32, tests fail in some of those cases (and as of August 13 "
    "2020, only those cases)")
class Conv3DTest(test.TestCase):

  def _DtypesToTest(self, use_gpu, forward=True):
    if forward:
      return [dtypes.float32]
    else:
      return [dtypes.float32]

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, stride,
                            padding, data_format, dtype, use_gpu):
    total_size_tensor = np.prod(tensor_in_sizes)
    total_size_filter = np.prod(filter_in_sizes)

    # Initializes the input tensor with array containing numbers from 0 to 1.
    # We keep the input tensor values fairly small to avoid overflowing float16
    # during the conv3d.
    x1 = [f * 1.0 / total_size_tensor for f in range(1, total_size_tensor + 1)]
    x2 = [f * 1.0 / total_size_filter for f in range(1, total_size_filter + 1)]
    t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
    t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)

    if isinstance(stride, collections_abc.Iterable):
        strides = [1] + list(stride) + [1]
    else:
        strides = [1, stride, stride, stride, 1]
    
    # Transpose + Conv3D[NDHWC] + Transpose
    transpose_before = array_ops.transpose(t1, [0, 2, 3, 4, 1])  
    conv = nn_ops.conv3d(transpose_before, t2, strides, padding=padding,
                        data_format="NDHWC")
    transpose_after = array_ops.transpose(conv, [0, 4, 1, 2, 3])
    output = array_ops.identity(transpose_after)
    
    #Conv3D[NCDHW]
    conv_2 = nn_ops.conv3d(t1, t2, strides, padding=padding,
                        data_format="NCDHW")


    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    with self.cached_session(use_gpu=use_gpu) as sess:
        ret_gpu = sess.run(output, options=run_options, run_metadata=metadata)
        graph = metadata.partition_graphs[0]
        found_transpose_op = False
        for node in graph.node:
            if "Transpose" in node.op:
                found_transpose_op = True
                break

        self.assertTrue(~found_transpose_op, "this pattern has fusion issue!!")        
    with self.cached_session(use_gpu=use_gpu) as sess_2:
        ret_gpu_2 = sess_2.run(conv_2, options=run_options, run_metadata=metadata)   
    self.assertAllClose(ret_gpu_2, ret_gpu)

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    for data_format, use_gpu in GetTestConfigs():
      for dtype in self._DtypesToTest(use_gpu, forward=True):
        self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            stride,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)

  def _ComputeReferenceDilatedConv(self, tensor_in_sizes, filter_in_sizes,
                                   stride, dilation, padding, data_format,
                                   use_gpu):
    total_size_tensor = np.prod(tensor_in_sizes)
    total_size_filter = np.prod(filter_in_sizes)

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_tensor + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_filter + 1)]
    with self.cached_session(use_gpu=use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      if isinstance(stride, collections_abc.Iterable):
        strides = list(stride)
      else:
        strides = [stride, stride, stride]
      if data_format == "NCDHW":
        t1 = test_util.NHWCToNCHW(t1)
        full_strides = [1, 1] + strides
        full_dilation = [1, 1] + dilation
      else:
        full_strides = [1] + strides + [1]
        full_dilation = [1] + dilation + [1]
      expected = nn_ops.convolution(
          t1,
          t2,
          padding=padding,
          strides=strides,
          dilation_rate=dilation,
          data_format=data_format)
      computed = nn_ops.conv3d(
          t1,
          t2,
          strides=full_strides,
          dilations=full_dilation,
          padding=padding,
          data_format=data_format)
      computed = array_ops.identity(computed)
      if data_format == "NCDHW":
        expected = test_util.NCHWToNHWC(expected)
        computed = test_util.NCHWToNHWC(computed)
    return expected, computed

  def _VerifyDilatedConvValues(self, tensor_in_sizes, filter_in_sizes, stride,
                               padding, dilations):
    expected_results = []
    computed_results = []
    default_dilations = (
        dilations[0] == 1 and dilations[1] == 1 and dilations[2] == 1)
    for data_format, use_gpu in GetTestConfigs():
      # If any dilation rate is larger than 1, only do test on the GPU
      # because we currently do not have a CPU implementation for arbitrary
      # dilation rates.
      if default_dilations or use_gpu:
        expected, computed = self._ComputeReferenceDilatedConv(
            tensor_in_sizes, filter_in_sizes, stride, dilations, padding,
            data_format, use_gpu)
        expected_results.append(expected)
        computed_results.append(computed)
        tolerance = 1e-2 if use_gpu else 1e-5
        with self.cached_session() as sess:
          expected_values = self.evaluate(expected_results)
          computed_values = self.evaluate(computed_results)
          for e_value, c_value in zip(expected_values, computed_values):
            print("expected = ", e_value)
            print("actual = ", c_value)
            self.assertAllClose(
                e_value.flatten(), c_value.flatten(), atol=tolerance, rtol=1e-6)

  def _CreateNumpyTensor(self, sizes):
    return np.asarray([f * 1.0 for f in range(1,
                                              np.prod(sizes) + 1)],
                      dtype=np.float32).reshape(sizes)

  def testConv3D1x1x1Filter(self):
    expected_output = [
        0.18518519, 0.22222222, 0.25925926, 0.40740741, 0.5, 0.59259259,
        0.62962963, 0.77777778, 0.92592593, 0.85185185, 1.05555556, 1.25925926,
        1.07407407, 1.33333333, 1.59259259, 1.2962963, 1.61111111, 1.92592593
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 1, 2, 3, 3],
        filter_in_sizes=[2, 1, 1, 1, 1],
        stride=1,
        padding="VALID",
        expected=expected_output)


if __name__ == "__main__":
  test.main()
