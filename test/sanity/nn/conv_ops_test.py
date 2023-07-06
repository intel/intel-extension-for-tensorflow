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

"""Functional tests for convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='1'
# TODO(ITEX): remove this line when tf 2.11 is published.
import time

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.compat import collections_abc



def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NHWC", False), ("NHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCHW" format is only supported on CUDA.
    test_configs += [("NCHW", True)]
  return test_configs

@test_util.run_all_in_native_and_block_format
class Conv2DTest(test.TestCase):

  def _DtypesToTest(self, use_gpu, forward=True):
    if forward:
      # return [dtypes.float32, dtypes.float16]
      # TODO(itex): Turn on fp16 forward test, once we upgrade the
      # OneDnn which fix accuracy issue
      return [dtypes.float32]
    else:
      return [dtypes.float32]

  def _CreateNumpyTensor(self, shape):
    total_size = 1
    for s in shape:
      total_size *= s
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, dilations,
                            strides, padding, data_format, dtype, use_gpu):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      dilations: Dilated rate: [col_dilation, row_dilation]
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU
    Returns:
      Symbolic tensor value that can be used to execute the computation
    """
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)

    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW(padding)
      conv = nn_ops.conv2d(
          t1,
          t2,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format)
      self.assertEqual(conv.dtype, dtype)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)

      return array_ops.identity(conv)

  def _CompareFwdValues(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that CPU and GPU produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)

    def _SetupVal(data_format, use_gpu):
      with test_util.device(use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t1 = test_util.NHWCToNCHW(t1)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d(
            t1, t2, strides=strides, padding=padding, data_format=data_format)
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        return array_ops.identity(conv)

    tensors = []
    for (data_format, use_gpu) in GetTestConfigs():
      tensors.append(_SetupVal(data_format, use_gpu))
    values = self.evaluate(tensors)
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-3, atol=1e-3)

  def _ComputeReferenceDilatedConv(self, tensor_in_sizes, filter_in_sizes,
                                   stride, dilation, padding, data_format,
                                   use_gpu):
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)
    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      if isinstance(stride, collections_abc.Iterable):
        strides = list(stride)
      else:
        strides = [stride, stride]
      if data_format == "NCHW":
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
      computed = nn_ops.conv2d(
          t1,
          t2,
          strides=full_strides,
          dilations=full_dilation,
          padding=padding,
          data_format=data_format)
      if data_format == "NCHW":
        expected = test_util.NCHWToNHWC(expected)
        computed = test_util.NCHWToNHWC(computed)
    return array_ops.identity(expected), array_ops.identity(computed)

  def _VerifyDilatedConvValues(self, tensor_in_sizes, filter_in_sizes, strides,
                               padding, dilations, rtol=1e-4):
    expected_results = []
    computed_results = []
    for data_format, use_gpu in GetTestConfigs():
      expected, computed = self._ComputeReferenceDilatedConv(
          tensor_in_sizes, filter_in_sizes, strides, dilations, padding,
          data_format, use_gpu)
      expected_results.append(expected)
      computed_results.append(computed)
    tolerance = 1e-2 if use_gpu else 1e-5
    expected_values = self.evaluate(expected_results)
    computed_values = self.evaluate(computed_results)
    for e_value, c_value in zip(expected_values, computed_values):
      tf_logging.debug("expected = %s", e_value)
      tf_logging.debug("actual = %s", c_value)
      self.assertAllClose(
          e_value.flatten(), c_value.flatten(), atol=tolerance, rtol=rtol)

  def _VerifyValues(self,
                    tensor_in_sizes,
                    filter_in_sizes,
                    strides,
                    padding,
                    expected,
                    dilations=(1, 1),
                    gpu_only=False,
                    test_grappler_layout_optimizer=False,
                    tol=1e-5,
                    fp16_tol=1e-3):
    if gpu_only and not test.is_gpu_available(cuda_only=True):
      return
    tensors = []
    dilations = list(dilations)
    for (data_format, use_gpu) in GetTestConfigs():
      if gpu_only and not use_gpu:
        continue
      dtypes_to_test = self._DtypesToTest(use_gpu, forward=True)
      if not test_grappler_layout_optimizer and data_format == "NHWC":
        dtypes_to_test.append(dtypes.int32)
      for dtype in dtypes_to_test:
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            dilations,
            strides,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu)
        if test_grappler_layout_optimizer and data_format == "NHWC" and use_gpu:
          # Grappler's layout optimizer will not optimize a fetch node, so
          # this identity allows Grappler to optimize the Conv2D node.
          result = array_ops.identity(result)
        tensors.append(result)
      values = self.evaluate(tensors)
      for i in range(len(tensors)):
        conv = tensors[i]
        value = values[i]
        tf_logging.debug("expected = %s", expected)
        tf_logging.debug("actual = %s", value)
        tol_to_use = fp16_tol if value.dtype == np.float16 else tol
        if np.issubdtype(value.dtype, np.integer):
          self.assertAllEqual(np.rint(expected), np.ravel(value))
        else:
          self.assertAllClose(expected, np.ravel(value), atol=tol_to_use,
                              rtol=tol_to_use)
        self.assertShapeEqual(value, conv)
        self.assertEqual(value.dtype, conv.dtype.as_numpy_dtype)


  def _VerifyExplicitPaddings(self,
                              tensor_in_sizes,
                              filter_in_sizes,
                              strides,
                              padding,
                              dilations=(1, 1),
                              test_grappler_layout_optimizer=False,
                              tol=1e-5,
                              fp16_tol=1e-3):
    """Verifies Conv2D with explicit padding generates correct values.

    It does this by comparing with Conv2D without explicit padding. This
    function assumes Conv2D without explicit padding works correctly.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      strides: [row_stride, col_stride] for the convolution;
      padding: Explicit padding amounts.
      dilations: Dilation values
      test_grappler_layout_optimizer: If True, allow the Grappler layout
        optimizer to run, which turns NHWC Conv2Ds on the GPU to NCHW Conv2Ds.
      tol: The absolute and relative tolerance for non-fp16 dtypes.
      fp16_tol: The absolute and relative tolerance for fp16.
    """
    input_tensor = self._CreateNumpyTensor(tensor_in_sizes)
    filter_tensor = self._CreateNumpyTensor(filter_in_sizes)
    input_tensor = array_ops.pad(input_tensor, [(0, 0)] + padding + [(0, 0)])
    dilations = list(dilations)
    conv2d_result = nn_ops.conv2d(
        input_tensor,
        filter_tensor, [1] + list(strides) + [1],
        "VALID",
        dilations=[1] + dilations + [1])
    expected = list(self.evaluate(array_ops.reshape(conv2d_result, [-1])))
    self._VerifyValues(
        tensor_in_sizes,
        filter_in_sizes,
        strides,
        padding,
        expected,
        dilations,
        test_grappler_layout_optimizer=test_grappler_layout_optimizer,
        tol=tol,
        fp16_tol=fp16_tol)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x1Filter(self):
    expected_output = [
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DExpandedBatch(self):
    tensor_in_sizes_batch = [10, 2, 3, 3]
    tensor_in_sizes_expanded_batch = [2, 5, 2, 3, 3]
    filter_in_sizes = [1, 1, 3, 3]
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    x1 = self._CreateNumpyTensor(tensor_in_sizes_batch)
    x2 = x1.reshape(tensor_in_sizes_expanded_batch)
    conv1 = array_ops.identity(nn_ops.conv2d(
        x1,
        filter_in,
        strides=[1, 1],
        padding="VALID"))
    conv2 = array_ops.identity(nn_ops.conv2d(
        x2,
        filter_in,
        strides=[1, 1],
        padding="VALID"))
    self.assertEqual(conv1.shape, tensor_in_sizes_batch)
    self.assertEqual(conv2.shape, tensor_in_sizes_expanded_batch)
    self.assertAllEqual(
        conv1,
        self.evaluate(conv2).reshape(conv1.shape))

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmpty(self):
    expected_output = []
    self._VerifyValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmptyDilation(self):
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID")

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterDilation(self):
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID")

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride2(self):
    expected_output = [2271.0, 2367.0, 2463.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="VALID",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride2Same(self):
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)
  
  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2GroupedFilterStride2Same(self):
    expected_output = [217., 271., 333., 119., 152., 189.]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 1, 3],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D2x2Padding(self):
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 2],
        filter_in_sizes=[2, 1, 2, 1],
        strides=[1, 1],
        padding=[[2, 2], [2, 2]])

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 2],
        filter_in_sizes=[1, 1, 2, 1],
        strides=[2, 1],
        padding=[[2, 2], [2, 2]])

  def testConv2DExplicitPaddingWithLayoutOptimizer(self):
    # Test with Grappler's layout optimizer, to ensure the layout optimizer
    # handles explicit padding correctly.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 3, 2, 1],
        filter_in_sizes=[1, 2, 1, 2],
        strides=[1, 1],
        padding=[[1, 0], [0, 1]],
        dilations=[2, 1],
        test_grappler_layout_optimizer=True)

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[3, 2, 2, 1],
        strides=[1, 1],
        padding=[[2, 1], [1, 2]],
        dilations=[2, 3],
        test_grappler_layout_optimizer=True)

  # Testing for backprops
  def _RunAndVerifyBackpropInput(self,
                                 input_sizes,
                                 filter_sizes,
                                 output_sizes,
                                 strides,
                                 padding,
                                 expected,
                                 data_format,
                                 use_gpu,
                                 err,
                                 dilations=(1, 1),
                                 dtype=dtypes.float32):
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    x1 = self._CreateNumpyTensor(filter_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    with test_util.device(use_gpu):
      if len(input_sizes) == 4:
        if data_format == "NCHW":
          input_sizes = test_util.NHWCToNCHW(input_sizes)
      t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
      t1 = constant_op.constant(x1, shape=filter_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t2 = test_util.NHWCToNCHW(t2)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW((padding))
      conv = nn_ops.conv2d_backprop_input(
          t0,
          t1,
          t2,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)
      # "values" consists of two tensors for two backprops
      value = self.evaluate(conv)
      self.assertShapeEqual(value, conv)
    tf_logging.debug("expected = %s", expected)
    tf_logging.debug("actual = %s", value)
    self.assertAllCloseAccordingToType(expected, value.flatten(), atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Depth1ValidBackpropInput(self):
    expected_output = [1.0, 4.0, 4.0, 3.0, 10.0, 8.0]
    for (data_format, use_gpu) in GetTestConfigs():
      for data_type in (dtypes.float32, dtypes.float16, dtypes.bfloat16):
        self._RunAndVerifyBackpropInput(
            input_sizes=[1, 2, 3, 1],
            filter_sizes=[2, 2, 1, 1],
            output_sizes=[1, 1, 2, 1],
            strides=[1, 1],
            padding="VALID",
            expected=expected_output,
            data_format=data_format,
            use_gpu=use_gpu,
            err=1e-5,
            dtype=data_type)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmptyBackpropInput(self):
    expected_output = []
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[0, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[0, 1, 2, 1],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5)

  # Testing for backprops
  def _RunAndVerifyBackpropFilter(self,
                                  input_sizes,
                                  filter_sizes,
                                  output_sizes,
                                  strides,
                                  padding,
                                  expected,
                                  data_format,
                                  use_gpu,
                                  dilations=(1, 1),
                                  err=1e-5):
    x0 = self._CreateNumpyTensor(input_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    explicit_strides = [1] + strides + [1]
    new_padding = padding
    new_dilations = [1] + dilations + [1]
    if isinstance(new_padding, (list, tuple)):
      new_padding = [(0, 0)] + new_padding + [(0, 0)]
    if data_format == "NCHW":
      explicit_strides = test_util.NHWCToNCHW(explicit_strides)
      new_dilations = test_util.NHWCToNCHW(new_dilations)
      if isinstance(padding, (list, tuple)):
        new_padding = test_util.NHWCToNCHW(new_padding)
    for dtype in self._DtypesToTest(use_gpu=use_gpu, forward=False):
      with test_util.device(use_gpu):
        t0 = constant_op.constant(x0, shape=input_sizes, dtype=dtype)
        t1 = constant_op.constant(filter_sizes, shape=[len(filter_sizes)])
        t2 = constant_op.constant(x2, shape=output_sizes, dtype=dtype)
        if data_format == "NCHW":
          t0 = test_util.NHWCToNCHW(t0)
          t2 = test_util.NHWCToNCHW(t2)
        conv = nn_ops.conv2d_backprop_filter(
            t0,
            t1,
            t2,
            strides=explicit_strides,
            padding=new_padding,
            dilations=new_dilations,
            data_format=data_format)
        value = self.evaluate(conv)
        self.assertShapeEqual(value, conv)
      tf_logging.debug("expected = %s", expected)
      tf_logging.debug("actual = %s", value)
      self.assertArrayNear(expected, value.flatten(), err)

  # Testing for backprops
  def _RunAndVerifyBackpropFilterDilation(self, input_sizes, filter_sizes,
                                          output_sizes, strides, dilations,
                                          padding, data_format, use_gpu, err):
    x1 = self._CreateNumpyTensor(input_sizes)
    x2 = self._CreateNumpyTensor(filter_sizes)
    default_dilations = (dilations[0] == 1 and dilations[1] == 1)
    if default_dilations or use_gpu:
      with self.cached_session(use_gpu=use_gpu) as sess:
        if data_format == "NCHW":
          input_sizes = test_util.NHWCToNCHW(input_sizes)
        t1 = constant_op.constant(x1, shape=input_sizes)
        t2 = constant_op.constant(x2, shape=filter_sizes)
        full_strides = [1] + strides + [1]
        full_dilations = [1] + dilations + [1]
        if data_format == "NCHW":
          full_strides = test_util.NHWCToNCHW(full_strides)
          full_dilations = test_util.NHWCToNCHW(full_dilations)
        conv_forward = nn_ops.conv2d(
            t1,
            t2,
            strides=full_strides,
            dilations=full_dilations,
            padding=padding,
            data_format=data_format)
        conv_forward_2 = nn_ops.convolution(
            t1,
            t2,
            padding=padding,
            strides=strides,
            dilation_rate=dilations,
            data_format=data_format)
        if data_format == "NCHW":
          conv_forward = test_util.NCHWToNHWC(conv_forward)
          conv_forward_2 = test_util.NCHWToNHWC(conv_forward_2)
        conv = gradients_impl.gradients(conv_forward, t2)[0]
        conv_2 = gradients_impl.gradients(conv_forward, t2)[0]
        value = self.evaluate(conv)
        value_2 = self.evaluate(conv_2)
        self.assertShapeEqual(value, conv)
        self.assertShapeEqual(value_2, conv_2)
      tf_logging.debug("expected = %s", value_2)
      tf_logging.debug("actual = %s", value)
      self.assertArrayNear(value_2.flatten(), value.flatten(), err)

  @test_util.deprecated_graph_mode_only
  def testConv2D2x2Depth3ValidBackpropFilterDilation2x2(self):
    if test.is_gpu_available(cuda_only=True) or test_util.IsMklEnabled():
      for (data_format, use_gpu) in GetTestConfigs():
        self._RunAndVerifyBackpropFilterDilation(
            input_sizes=[1, 3, 4, 3],
            filter_sizes=[2, 2, 3, 3],
            output_sizes=[1, 1, 2, 3],
            strides=[1, 1],
            dilations=[2, 2],
            padding="VALID",
            data_format=data_format,
            use_gpu=use_gpu,
            err=1e-5)

  def _RunAndVerifyBackpropInputExplicitPadding(self,
                                                input_sizes,
                                                filter_sizes,
                                                output_sizes,
                                                strides,
                                                padding,
                                                data_format,
                                                use_gpu,
                                                dilations=(1, 1),
                                                err=2e-5):
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    if not use_gpu and dilations != (1, 1):
      return  # Non-default dilations is currently not supported on the CPU.

    x1 = self._CreateNumpyTensor(filter_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    padded_input_sizes = input_sizes[:]
    padded_input_sizes[1] += padding[0][0] + padding[0][1]
    padded_input_sizes[2] += padding[1][0] + padding[1][1]
    c = nn_ops.conv2d_backprop_input(
        padded_input_sizes,
        x1,
        x2,
        strides=[1] + strides + [1],
        padding="VALID",
        dilations=[1] + dilations + [1])
    c = c[:, padding[0][0]:(c.shape[1] - padding[0][1]), padding[1][0]:(
        c.shape[2] - padding[1][1]), :]
    expected = list(self.evaluate(array_ops.reshape(c, [-1])))
    self._RunAndVerifyBackpropInput(
        input_sizes,
        filter_sizes,
        output_sizes,
        strides,
        padding,
        expected,
        data_format,
        use_gpu=use_gpu,
        err=err,
        dilations=dilations)

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D2x2Depth1Padding_1_8_4_1_BackpropInput(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInputExplicitPadding(
          input_sizes=[1, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[1, 10, 8, 1],
          strides=[1, 1],
          padding=[[1, 8], [4, 2]],
          data_format=data_format,
          use_gpu=use_gpu,
          err=5e-5)

      self._RunAndVerifyBackpropInputExplicitPadding(
          input_sizes=[1, 5, 3, 1],
          filter_sizes=[3, 2, 1, 1],
          output_sizes=[1, 4, 8, 1],
          strides=[3, 1],
          padding=[[1, 8], [4, 2]],
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D2x2Depth1Padding_5_0_2_2_BackpropInput(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropInputExplicitPadding(
          input_sizes=[1, 3, 3, 1],
          filter_sizes=[2, 1, 1, 1],
          output_sizes=[1, 7, 7, 1],
          strides=[1, 1],
          padding=[[5, 0], [2, 2]],
          data_format=data_format,
          err=5e-5,
          use_gpu=use_gpu)

      self._RunAndVerifyBackpropInputExplicitPadding(
          input_sizes=[1, 4, 2, 1],
          filter_sizes=[3, 3, 1, 1],
          output_sizes=[1, 5, 2, 1],
          strides=[1, 2],
          padding=[[5, 0], [2, 2]],
          data_format=data_format,
          dilations=[2, 1],
          use_gpu=use_gpu)

  def _RunAndVerifyBackpropFilterExplicitPadding(self,
                                                 input_sizes,
                                                 filter_sizes,
                                                 output_sizes,
                                                 strides,
                                                 padding,
                                                 data_format,
                                                 use_gpu,
                                                 dilations=(1, 1),
                                                 err=1e-5):
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    if not use_gpu and dilations != (1, 1):
      return  # Non-default dilations is currently not supported on the CPU.

    x0 = self._CreateNumpyTensor(input_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)

    x0 = np.pad(x0, [(0, 0)] + padding + [(0, 0)], "constant")
    c = nn_ops.conv2d_backprop_filter(
        x0,
        filter_sizes,
        x2,
        strides=[1] + strides + [1],
        padding="VALID",
        dilations=[1] + dilations + [1])
    expected = list(self.evaluate(array_ops.reshape(c, [-1])))
    self._RunAndVerifyBackpropFilter(
        input_sizes,
        filter_sizes,
        output_sizes,
        strides,
        padding,
        expected,
        data_format,
        use_gpu=use_gpu,
        dilations=dilations,
        err=err)

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D2x2Depth1Padding0x0BackpropFilter(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self._RunAndVerifyBackpropFilterExplicitPadding(
          input_sizes=[1, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[1, 1, 2, 1],
          strides=[1, 1],
          padding=[[0, 0], [0, 0]],
          data_format=data_format, use_gpu=use_gpu)

      self._RunAndVerifyBackpropFilterExplicitPadding(
          input_sizes=[1, 3, 4, 2],
          filter_sizes=[2, 2, 2, 3],
          output_sizes=[1, 1, 2, 3],
          strides=[2, 2],
          padding=[[0, 0], [0, 0]],
          data_format=data_format, use_gpu=use_gpu)

  # Gradient checkers
  def ConstructAndTestGradient(self,
                               batch,
                               input_rows,
                               input_cols,
                               filter_rows,
                               filter_cols,
                               in_depth,
                               out_depth,
                               stride_rows,
                               stride_cols,
                               padding,
                               test_input,
                               data_format,
                               use_gpu,
                               num_groups=1,
                               max_err=0.003):
    assert in_depth % num_groups == 0 and out_depth % num_groups == 0
    input_shape = [batch, input_rows, input_cols, in_depth]
    filter_shape = [filter_rows, filter_cols, in_depth // num_groups, out_depth]
    # TODO(yangke): re-factor the computation of output shape.
    if padding == "VALID":
      output_rows = (input_rows - filter_rows + stride_rows) // stride_rows
      output_cols = (input_cols - filter_cols + stride_cols) // stride_cols
    elif padding == "SAME":
      output_rows = (input_rows + stride_rows - 1) // stride_rows
      output_cols = (input_cols + stride_cols - 1) // stride_cols
    else:
      self.assertIsInstance(padding, (list, tuple))
      output_rows = (input_rows + padding[1][0] + padding[1][1] - filter_rows +
                     stride_rows) // stride_rows
      output_cols = (input_cols + padding[2][0] + padding[2][1] - filter_cols +
                     stride_cols) // stride_cols
    output_shape = [batch, output_rows, output_cols, out_depth]
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    # Conv2DGrad functions are not compiled for double due to
    # a problem in the way Eigen's Conv2DGrad works for double.
    # So we disable the DOUBLE path.  We should re-enable this
    # when double support returns for CPU and/or GPU.
    for dtype in self._DtypesToTest(use_gpu=use_gpu, forward=False):
      with self.cached_session(use_gpu=use_gpu):
        input_tensor = constant_op.constant(
            input_data, shape=input_shape, dtype=dtype, name="input")
        filter_tensor = constant_op.constant(
            filter_data, shape=filter_shape, dtype=dtype, name="filter")
        strides = [1, stride_rows, stride_cols, 1]
        new_padding = padding
        if data_format == "NCHW":
          new_input_tensor = test_util.NHWCToNCHW(input_tensor)
          strides = test_util.NHWCToNCHW(strides)
          if isinstance(padding, (list, tuple)):
            new_padding = test_util.NHWCToNCHW(padding)
        else:
          new_input_tensor = input_tensor
        conv = array_ops.identity(nn_ops.conv2d(
            new_input_tensor,
            filter_tensor,
            strides,
            new_padding,
            data_format=data_format,
            name="conv"))
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        self.assertEqual(output_shape, conv.get_shape())
        if test_input:
          jacob_t, jacob_n = gradient_checker.compute_gradient(input_tensor,
                                                               input_shape,
                                                               conv,
                                                               output_shape)
        else:
          jacob_t, jacob_n = gradient_checker.compute_gradient(filter_tensor,
                                                               filter_shape,
                                                               conv,
                                                               output_shape)
        if dtype == dtypes.float32:
          reference_jacob_t = jacob_t
          err = np.fabs(jacob_t - jacob_n).max()
        else:
          # Compare fp16 theoretical gradients to fp32 theoretical gradients,
          # since fp16 numerical gradients are too imprecise.
          err = np.fabs(jacob_t - reference_jacob_t).max()

        tf_logging.debug("conv_2d gradient error = %s", err)
        self.assertLess(err, max_err)

  @test_util.deprecated_graph_mode_only
  def testInputGradientValidPaddingStrideOne(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=5,
          input_cols=4,
          filter_rows=3,
          filter_cols=3,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="VALID",
          test_input=True,
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.deprecated_graph_mode_only
  def testFilterGradientSamePaddingStrideOne(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=4,
          input_rows=6,
          input_cols=5,
          filter_rows=2,
          filter_cols=2,
          in_depth=2,
          out_depth=3,
          stride_rows=1,
          stride_cols=1,
          padding="SAME",
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.deprecated_graph_mode_only
  def testFilterGradient1_2_3_4PaddingStride3x2(self):
    for (data_format, use_gpu) in GetTestConfigs():
      self.ConstructAndTestGradient(
          batch=2,
          input_rows=8,
          input_cols=5,
          filter_rows=4,
          filter_cols=2,
          in_depth=3,
          out_depth=2,
          stride_rows=3,
          stride_cols=2,
          padding=[[0, 0], [1, 2], [3, 4], [0, 0]],
          test_input=False,
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.deprecated_graph_mode_only
  def testShapeFunctionEdgeCases(self):
    # All shapes unknown.
    c1 = nn_ops.conv2d(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.float32),
        strides=[1, 1, 1, 1],
        padding="SAME")
    self.assertEqual([None, None, None, None], c1.get_shape().as_list())

    # Incorrect input shape.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(
              dtypes.float32, shape=[1, 3]),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding="SAME")

    # Incorrect filter shape.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(
              dtypes.float32, shape=[1, 3]),
          strides=[1, 1, 1, 1],
          padding="SAME")

    # Depth mismatch.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(
              dtypes.float32, shape=[32, 20, 20, 3]),
          array_ops.placeholder(
              dtypes.float32, shape=[4, 4, 2, 2]),
          strides=[1, 1, 1, 1],
          padding="SAME")

    # Input depth divisible by filter depth (group convolution).
    # No exceptions should appear.
    nn_ops.conv2d(
        array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 8]),
        array_ops.placeholder(dtypes.float32, shape=[4, 4, 2, 16]),
        strides=[1, 1, 1, 1],
        padding="SAME")

    # Negative padding.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding=[[0, 0], [0, -1], [1, 2], [0, 0]])

    # Nonzero padding in nonspatial dimension.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding=[[1, 0], [0, 0], [0, 0], [0, 0]])

    # Nonzero NCHW padding in nonspatial dimension.
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding=[[0, 0], [0, 1], [0, 0], [0, 0]],
          data_format="NCHW")

    # Wrong amount of padding
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding=[[0, 0], [0, 0], [0, 0]])

    # Only specify one padding amount per dimension
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding=[[0], [0], [0], [0]])

    # Explicit padding elements are not lists
    with self.assertRaises(ValueError):
      nn_ops.conv2d(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.float32),
          strides=[1, 1, 1, 1],
          padding=[0, 0, 0, 0])
  @test_util.deprecated_graph_mode_only
  def testOpEdgeCases(self):
    # Illegal strides.
    with self.assertRaisesRegex((ValueError, errors_impl.UnimplementedError),
                                "strides in the batch and depth"):
      input_val = np.ones([2, 4, 10, 10])
      filter_val = np.ones([2, 4, 10, 10])
      self.evaluate(
          nn_ops.conv2d(
              input_val, filter_val, strides=[2, 1, 1, 1], padding="SAME"))
    with self.assertRaisesRegex((ValueError, errors_impl.UnimplementedError),
                                "strides in the batch and depth"):
      input_val = np.ones([2, 4, 10, 10])
      filter_val = np.ones([2, 4, 10, 10])
      self.evaluate(
          nn_ops.conv2d(
              input_val, filter_val, strides=[1, 1, 1, 2], padding="SAME"))

    # TODO(itex): Will enable when TF master fixed for V2 behavior
    # # Filter larger than input.
    # with self.assertRaisesRegex(ValueError, "Negative dimension size"):
    #   input_val = np.ones([32, 20, 20, 3])
    #   filter_val = np.ones([20, 21, 3, 2])
    #   self.evaluate(
    #       nn_ops.conv2d(
    #           input_val, filter_val, strides=[1, 1, 1, 1], padding="VALID"))
    # with self.assertRaisesRegex(ValueError, "Negative dimension size"):
    #   input_val = np.ones([32, 20, 20, 3])
    #   filter_val = np.ones([21, 20, 3, 2])
    #   self.evaluate(
    #       nn_ops.conv2d(
    #           input_val, filter_val, strides=[1, 1, 1, 1], padding="VALID"))
    #
    # # Filter larger than input + padding.
    # with self.assertRaisesRegex(ValueError, "Negative dimension size"):
    #   input_val = np.ones([32, 20, 20, 3])
    # filter_val = np.ones([24, 25, 3, 2])
    #   self.evaluate(
    #       nn_ops.conv2d(
    #           input_val,
    #           filter_val,
    #           strides=[1, 1, 1, 1],
    #           padding=[[0, 0], [2, 2], [2, 2], [0, 0]]))

    # Filter dimensions must be greater than 0.
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "filter must not have zero elements"
        "|has a non-positive dimension"):
      input_val = np.ones([1, 1, 1, 1])
      filter_val = np.ones([1, 0, 1, 1])
      self.evaluate(
          nn_ops.conv2d(
              input_val, filter_val, strides=[1, 1, 1, 1], padding="SAME"))

    # Negative padding during backprop.
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "All elements of explicit_paddings must be nonnegative"):
      filter_val = np.ones([18, 18, 3, 2])
      out_backprop_val = np.ones([32, 3, 2, 2])
      self.evaluate(
          nn_ops.conv2d_backprop_input([32, 20, 20, 3],
                                       filter_val,
                                       out_backprop_val,
                                       strides=[1, 1, 1, 1],
                                       padding=[[0, 0], [-1, 0], [0, 0], [0,
                                                                          0]]))
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "All elements of explicit_paddings must be nonnegative"):
      input_val = np.ones([32, 20, 20, 3])
      out_backprop_val = np.ones([32, 3, 2, 2])
      self.evaluate(
          nn_ops.conv2d_backprop_filter(
              input_val, [18, 18, 3, 2],
              out_backprop_val,
              strides=[1, 1, 1, 1],
              padding=[[0, 0], [-1, 0], [0, 0], [0, 0]]))

@test_util.run_all_in_native_and_block_format
class DepthwiseConv2DTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [filter_rows, filter_cols, input_depth, depth_multiplier].
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.cached_session() as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t1.set_shape(tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      conv = array_ops.identity(nn_impl.depthwise_conv2d(
          t1, t2, strides=[1, stride, stride, 1], padding=padding))
      value = self.evaluate(conv)
    tf_logging.debug("value = %s", value)
    self.assertArrayNear(expected, np.ravel(value), 1e-5)
    self.assertShapeEqual(value, conv)

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Filter(self):
    # The inputs look like this (it's a 3 x 2 matrix, each of depth 2):
    #
    # [ (1.0, 2.0), (3.0,  4.0), ( 5.0,  6.0) ]
    # [ (7.0, 8.0), (9.0, 10.0), (11.0, 12.0) ]
    #  We can view this as two inputs
    #
    #  input depth 0:
    #
    #  [ 1.0,  3.0,  5.0 ]
    #  [ 7.0,  9.0, 11.0 ]
    #
    #  input depth 1:
    #
    #  [ 2.0,  4.0,  6.0 ]
    #  [ 8.0, 10.0, 12.0 ]
    #
    # The filter looks like this (it has two 2 x 2 patches, each generating 2
    # depths):
    #
    #  filter #0:
    #
    #  [ (1.0,  3.0), ( 5.0,  7.0)]
    #  [ (9.0, 11.0), (13.0, 15.0)]
    #
    #  filter #1:
    #
    #  [ ( 2.0,  4.0), ( 6.0,  8.0)]
    #  [ (10.0, 12.0), (14.0, 16.0)]
    #
    # So the outputs are:
    #
    # (position 0, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  1.0 * 1.0 + 7.0 * 9.0 + 3.0 * 5.0 + 9.0 * 13.0 = 196
    # (position 0, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  1.0 * 2.0 + 7.0 * 10.0 + 3.0 * 6.0 + 9.0 * 14.0 = 216
    # (position 0, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  2.0 * 3.0 + 8.0 * 11.0 + 4.0 * 7.0 + 10.0 * 15.0 = 272
    # (position 0, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  2.0 * 4.0 + 8.0 * 12.0 + 4.0 * 8.0 + 10.0 * 16.0 = 296
    #
    # (position 1, 0: in_depth 0, output_depth 0 -- using filter #0)
    #  3.0 * 1.0 + 9.0 * 9.0 + 5.0 * 5.0 + 11.0 * 13.0 = 252
    # (position 1, 0: in_depth 0, output_depth 1 -- using filter #1)
    #  3.0 * 2.0 + 9.0 * 10.0 + 5.0 * 6.0 + 11.0 * 14.0 = 280
    # (position 1, 0: in_depth 1, output_depth 2 -- using filter #0)
    #  4.0 * 3.0 + 10.0 * 11.0 + 6.0 * 7.0 + 12.0 * 15.0 = 344
    # (position 1, 0: in_depth 1, output_depth 3 -- using filter #1)
    #  4.0 * 4.0 + 10.0 * 12.0 + 6.0 * 8.0 + 12.0 * 16.0 = 376
    expected_output = [196, 216, 272, 296, 252, 280, 344, 376]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        stride=1,
        padding="VALID",
        expected=expected_output)

@test_util.run_all_in_native_and_block_format
class SeparableConv2DTest(test.TestCase):

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

  def _VerifyValues(self,
                    tensor_in_sizes,
                    depthwise_filter_in_sizes,
                    pointwise_filter_in_sizes,
                    stride,
                    padding,
                    expected,
                    data_format="NHWC"):
    """Verifies the output values of the separable convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions.
      depthwise_filter_in_sizes: Depthwise filter tensor dimensions.
      pointwise_filter_in_sizes: Pointwise filter tensor dimensions.
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      data_format: string data format for input tensor.
    """
    with self.cached_session(use_gpu=True) as sess:
      t1 = self._InitValues(tensor_in_sizes)
      f1 = self._InitValues(depthwise_filter_in_sizes)
      f1.set_shape(depthwise_filter_in_sizes)
      f2 = self._InitValues(pointwise_filter_in_sizes)

      real_t1 = t1
      strides = [1, stride, stride, 1]
      if data_format == "NCHW":
        real_t1 = array_ops.transpose(t1, [0, 3, 1, 2])
        strides = [1, 1, stride, stride]
        if isinstance(padding, list):
          padding = [padding[0], padding[3], padding[1], padding[2]]

      conv = nn_impl.separable_conv2d(
          real_t1,
          f1,
          f2,
          strides=strides,
          padding=padding,
          data_format=data_format)

      if data_format == "NCHW":
        conv = array_ops.transpose(conv, [0, 2, 3, 1])

      value = self.evaluate(array_ops.identity(conv))
    tf_logging.debug("value = %s", value)
    self.assertArrayNear(expected, np.ravel(value), 2e-3)
    self.assertShapeEqual(value, conv)

  def _testSeparableConv2D(self, data_format):
    # The output is the result of two convolutions:
    # First with tensor_in[1, 4, 4, 2] * filter1[2, 2, 2, 3].
    # Second with intermediate_out[1, 4, 4, 6] * filter2[1, 1, 6, 7].
    # Complexity is O(2*3*2*2 + 6*7*1*1) as opposed to O(2*7*2*2).
    expected_output = [
        6644.5, 6971.5, 7298.5, 7625.5, 7952.5, 8279.5, 8606.5, 8154.5, 8556.5,
        8958.5, 9360.5, 9762.5, 10164.5, 10566.5, 9664.5, 10141.5, 10618.5,
        11095.5, 11572.5, 12049.5, 12526.5, 4145.5, 4346.5, 4547.5, 4748.5,
        4949.5, 5150.5, 5351.5, 12684.5, 13311.5, 13938.5, 14565.5, 15192.5,
        15819.5, 16446.5, 14194.5, 14896.5, 15598.5, 16300.5, 17002.5, 17704.5,
        18406.5, 15704.5, 16481.5, 17258.5, 18035.5, 18812.5, 19589.5, 20366.5,
        6499.5, 6814.5, 7129.5, 7444.5, 7759.5, 8074.5, 8389.5, 18724.5,
        19651.5, 20578.5, 21505.5, 22432.5, 23359.5, 24286.5, 20234.5, 21236.5,
        22238.5, 23240.5, 24242.5, 25244.5, 26246.5, 21744.5, 22821.5, 23898.5,
        24975.5, 26052.5, 27129.5, 28206.5, 8853.5, 9282.5, 9711.5, 10140.5,
        10569.5, 10998.5, 11427.5, 5746.75, 6010.75, 6274.75, 6538.75, 6802.75,
        7066.75, 7330.75, 6168.75, 6452.25, 6735.75, 7019.25, 7302.75, 7586.25,
        7869.75, 6590.75, 6893.75, 7196.75, 7499.75, 7802.75, 8105.75, 8408.75,
        2036.25, 2119.5, 2202.75, 2286.0, 2369.25, 2452.5, 2535.75
    ]

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 2],
        depthwise_filter_in_sizes=[2, 2, 2, 3],
        pointwise_filter_in_sizes=[1, 1, 6, 7],
        stride=1,
        padding="SAME",
        expected=expected_output,
        data_format=data_format)

  @test_util.run_in_graph_and_eager_modes
  def testSeparableConv2D(self):
    self._testSeparableConv2D("NHWC")

  def disabledtestSeparableConv2DNCHW(self):
    if not test.is_gpu_available():
      return
    self._testSeparableConv2D("NCHW")

  def _testSeparableConv2DEqualInputOutputDepth(self, data_format):
    # The output is the result of two convolutions:
    # First with tensor_in[1, 4, 4, 2] * filter1[2, 2, 3, 3].
    # Second with intermediate_out[1, 4, 4, 6] * filter2[1, 1, 6, 6].
    # Complexity is O(2*3*2*2 + 6*6*1*1) as opposed to O(2*6*2*2).
    expected_output = [
        5742.0, 6069.0, 6396.0, 6723.0, 7050.0, 7377.0, 7047.0, 7449.0, 7851.0,
        8253.0, 8655.0, 9057.0, 8352.0, 8829.0, 9306.0, 9783.0, 10260.0,
        10737.0, 3582.0, 3783.0, 3984.0, 4185.0, 4386.0, 4587.0, 10962.0,
        11589.0, 12216.0, 12843.0, 13470.0, 14097.0, 12267.0, 12969.0, 13671.0,
        14373.0, 15075.0, 15777.0, 13572.0, 14349.0, 15126.0, 15903.0, 16680.0,
        17457.0, 5616.0, 5931.0, 6246.0, 6561.0, 6876.0, 7191.0, 16182.0,
        17109.0, 18036.0, 18963.0, 19890.0, 20817.0, 17487.0, 18489.0, 19491.0,
        20493.0, 21495.0, 22497.0, 18792.0, 19869.0, 20946.0, 22023.0, 23100.0,
        24177.0, 7650.0, 8079.0, 8508.0, 8937.0, 9366.0, 9795.0, 4963.5, 5227.5,
        5491.5, 5755.5, 6019.5, 6283.5, 5328.0, 5611.5, 5895.0, 6178.5, 6462.0,
        6745.5, 5692.5, 5995.5, 6298.5, 6601.5, 6904.5, 7207.5, 1757.25, 1840.5,
        1923.75, 2007.0, 2090.25, 2173.5
    ]

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 2],
        depthwise_filter_in_sizes=[2, 2, 2, 3],
        pointwise_filter_in_sizes=[1, 1, 6, 6],
        stride=1,
        padding="SAME",
        expected=expected_output,
        data_format=data_format)

  @test_util.deprecated_graph_mode_only
  def testSeparableConv2DEqualInputOutputDepth(self):
    self._testSeparableConv2DEqualInputOutputDepth("NHWC")

#!!!
  # def testSeparableConv2DEqualInputOutputDepthNCHW(self):
  #   if not test.is_gpu_available():
  #     return
  #   self._testSeparableConv2DEqualInputOutputDepth("NCHW")

  def _testSeparableConv2dExplicitPadding(self, data_format):
    tensor_in_sizes = [1, 4, 4, 2]
    depthwise_filter_in_sizes = [2, 2, 2, 3]
    pointwise_filter_in_sizes = [1, 1, 6, 7]
    padding = [[0, 0], [1, 2], [3, 4], [0, 0]]
    with self.cached_session(use_gpu=True):
      # Compute the 'expected' values by manually padding before calling
      # separable_conv2d
      t1 = self._InitValues(tensor_in_sizes)
      t1 = array_ops.pad(t1, padding)
      f1 = self._InitValues(depthwise_filter_in_sizes)
      f1.set_shape(depthwise_filter_in_sizes)
      f2 = self._InitValues(pointwise_filter_in_sizes)
      conv = nn_impl.separable_conv2d(
          t1,
          f1,
          f2,
          strides=[1, 1, 1, 1],
          padding="VALID",
          data_format="NHWC")
      expected = self.evaluate(conv)
      expected = np.ravel(expected)
    self._VerifyValues(
        tensor_in_sizes=tensor_in_sizes,
        depthwise_filter_in_sizes=depthwise_filter_in_sizes,
        pointwise_filter_in_sizes=pointwise_filter_in_sizes,
        stride=1,
        padding=padding,
        expected=expected,
        data_format=data_format)

  @test_util.run_in_graph_and_eager_modes
  def testSeparableConv2dExplicitPadding(self):
    self._testSeparableConv2dExplicitPadding("NHWC")

#!!!
  # def testSeparableConv2dExplicitPaddingNCHW(self):
  #   if not test.is_gpu_available():
  #     return
  #   self._testSeparableConv2dExplicitPadding("NCHW")

@test_util.run_all_in_graph_and_eager_modes
@test_util.run_all_in_native_and_block_format
class DeepConv2DTest(test.TestCase):

  def _CompareFwdConv2D(self, tensor_in_sizes, filter_in_sizes, conv_strides,
                        padding):
    """Verifies that DeepConv2D and Conv2D produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)

    with self.cached_session(use_gpu=False) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      strides = [1] + conv_strides + [1]

      conv = array_ops.identity(nn_ops.conv2d(t1, t2, strides=strides, padding=padding))

      os.environ["TF_USE_DEEP_CONV2D"] = "0"
      values_expect = self.evaluate([conv])

      os.environ["TF_USE_DEEP_CONV2D"] = "1"
      values_test = self.evaluate([conv])

      self.assertAllClose(values_expect, values_test, rtol=1e-5, atol=1e-5)

  def _RunTestCases(self, conv_strides, padding):
    input_sizes = [[5, 5, 5, 1248], [3, 17, 17, 192], [2, 35, 35, 288],
                   [2, 6, 8, 517], [2, 7, 4, 81], [3, 11, 3, 77]]
    filter_sizes = [[3, 3, 1248, 128], [3, 3, 192, 192], [3, 3, 288, 384],
                    [3, 3, 517, 64], [3, 3, 81, 77], [3, 3, 77, 181]]
    for input_shape, filter_shape in zip(input_sizes, filter_sizes):
      self._CompareFwdConv2D(input_shape, filter_shape, conv_strides, padding)

  def testConv2D3x3FilterStride1x1Valid(self):
    self._RunTestCases([1, 1], "VALID")

  def testConv2D3x3FilterStride1x1Same(self):
    self._RunTestCases([1, 1], "SAME")


def GetInceptionFwdTest(input_size, filter_size, stride, padding,
                        gpu_only=False):

  def Test(self):
    if gpu_only and not test.is_gpu_available():
      tf_logging.info("Skipping InceptionFwd %s", (input_size, filter_size,
                                                   stride, padding))
      return
    tf_logging.info("Testing InceptionFwd %s", (input_size, filter_size, stride,
                                                padding))
    self._CompareFwdValues(input_size, filter_size, [stride, stride], padding)

  return Test


def GetInceptionFwdDilatedConvTest(input_size, filter_size, stride, padding):

  def Test(self):
    if stride == 1:
      tf_logging.info("Testing InceptionFwd with dilations %s",
                      (input_size, filter_size, stride, padding))
      self._VerifyDilatedConvValues(
          tensor_in_sizes=input_size,
          filter_in_sizes=filter_size,
          strides=[stride, stride],
          dilations=[2, 2],
          padding=padding,
          rtol=5e-4)

  return Test

@test_util.run_all_in_native_and_block_format
class FusedConv2DTest(test.TestCase):

  def _CreateNumpyTensor(self, shape):
    total_size = np.prod(shape)
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  def _CreateConv2D(self,
                    input_values,
                    filters,
                    strides=[1, 1],
                    padding="SAME"):
    return nn_ops.convolution(
        input_values, filters, strides=strides, padding=padding)

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add has refcount 1.
  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testAddWithRefCountOne(self):
    expected_output = [
        113377, 125570, 77305, 86738, 19433, 22226, 60681, 70722, 36291, 43718,
        7143, 9206, 9785, 12098, 4783, 6366, 779, 1134
    ]
    tensor_in_sizes = [1, 3, 3, 2]
    filter_in_sizes = [2, 2, 2, 2]
    bias_in_sizes = [2]

    x = self._CreateNumpyTensor(tensor_in_sizes)
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    bias_in = self._CreateNumpyTensor(bias_in_sizes)
    # To get different weights for filter
    offset = 1

    conv1 = self._CreateConv2D(x, filter_in)
    conv2 = self._CreateConv2D(conv1, filter_in + offset)

    conv = self._CreateConv2D(conv1, filter_in - offset)
    bias_add = nn_ops.bias_add(conv, bias_in)
    add = array_ops.identity(math_ops.add_n([bias_add, conv2]))

    self.assertAllEqual(
        np.rint(expected_output),
        self.evaluate(add).reshape(-1))

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add has a total refcount of 2, and Add is its last consumer.
  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testAddWithRefCountTwoAndRunAddLast(self):
    expected_output = [
        1.907175e+06, 2.253505e+06, 7.809210e+05, 9.537180e+05, 1.184170e+05,
        1.523070e+05, 5.367010e+05, 6.803700e+05, 1.867090e+05, 2.529460e+05,
        2.362300e+04, 3.522600e+04, 5.121700e+04, 7.168300e+04, 1.494300e+04,
        2.347400e+04, 1.558000e+03, 2.903000e+03
    ]
    tensor_in_sizes = [1, 3, 3, 2]
    filter_in_sizes = [2, 2, 2, 2]
    bias_in_sizes = [2]

    x = self._CreateNumpyTensor(tensor_in_sizes)
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    bias_in = self._CreateNumpyTensor(bias_in_sizes)
    # To get different weights for filter
    offset = 1

    conv1 = self._CreateConv2D(x, filter_in)
    conv2 = self._CreateConv2D(conv1, filter_in + offset)

    conv = self._CreateConv2D(conv2, filter_in - offset)
    bias_add = nn_ops.bias_add(conv, bias_in)
    add = array_ops.identity(math_ops.add_n([bias_add, conv1]))

    self.assertAllEqual(
        np.rint(expected_output),
        self.evaluate(add).reshape(-1))

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add has refcount 2 and Add (in the fused Conv2D op) is its first consumer.
  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testAddWithRefCountTwoAndRunAddFirst(self):
    expected_output = [
        176161, 194450, 120673, 134822, 30545, 34734, 96041, 111102, 58149,
        69289, 11745, 14839, 15833, 19302, 7965, 10339, 1345, 1877
    ]
    tensor_in_sizes = [1, 3, 3, 2]
    filter_in_sizes = [2, 2, 2, 2]
    bias_in_sizes = [2]

    x = self._CreateNumpyTensor(tensor_in_sizes)
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    bias_in = self._CreateNumpyTensor(bias_in_sizes)
    # To get different weights for filter
    offset = 1

    conv1 = self._CreateConv2D(x, filter_in)
    conv2 = self._CreateConv2D(conv1, filter_in + offset)

    conv = self._CreateConv2D(conv1, filter_in - offset)
    bias_add = nn_ops.bias_add(conv, bias_in)
    add = math_ops.add_n([bias_add, conv2])

    relu = nn_ops.relu(add)
    output = array_ops.identity(math_ops.add_n([relu, conv2]))

    self.assertAllEqual(
        np.rint(expected_output),
        self.evaluate(output).reshape(-1))

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add has refcount 2, and there is no dependency between its two consumers.
  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testAddWithRefCountTwoAndNoDependence(self):
    expected_output = [
        176161, 194450, 120673, 134822, 30545, 34734, 96041, 111102, 58149,
        69289, 11745, 14839, 15833, 19302, 7965, 10339, 1345, 1877
    ]
    tensor_in_sizes = [1, 3, 3, 2]
    filter_in_sizes = [2, 2, 2, 2]
    bias_in_sizes = [2]

    x = self._CreateNumpyTensor(tensor_in_sizes)
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    bias_in = self._CreateNumpyTensor(bias_in_sizes)
    # To get different weights for filter
    offset = 1

    conv1 = self._CreateConv2D(x, filter_in)
    conv2 = self._CreateConv2D(conv1, filter_in + offset)

    conv = self._CreateConv2D(conv1, filter_in - offset)
    bias_add = nn_ops.bias_add(conv, bias_in)
    add = math_ops.add_n([bias_add, conv2])

    relu1 = nn_ops.relu(add)
    relu2 = nn_ops.relu(conv2)
    output = array_ops.identity(math_ops.add_n([relu1, relu2]))

    self.assertAllEqual(
        np.rint(expected_output),
        self.evaluate(output).reshape(-1))

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add is the same as the input to the fused Conv2D op and needs a tensor
  # buffer.
  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testAddWithSameSrcAndAddTensorBuffer(self):
    expected_output = [
        57157, 63298, 39249, 44026, 9971, 11402, 31193, 36306, 19126, 22948,
        3970, 5060, 5135, 6350, 2666, 3524, 461, 674
    ]
    tensor_in_sizes = [1, 3, 3, 2]
    filter_in_sizes = [2, 2, 2, 2]
    bias_in_sizes = [2]

    x = self._CreateNumpyTensor(tensor_in_sizes)
    filter_in = self._CreateNumpyTensor(filter_in_sizes)
    bias_in = self._CreateNumpyTensor(bias_in_sizes)

    conv1 = self._CreateConv2D(x, filter_in)

    conv = self._CreateConv2D(conv1, filter_in)
    bias_add = nn_ops.bias_add(conv, bias_in)
    add = array_ops.identity(math_ops.add_n([bias_add, conv1]))

    self.assertAllEqual(
        np.rint(expected_output),
        self.evaluate(add).reshape(-1))


if __name__ == "__main__":
  test.main()
