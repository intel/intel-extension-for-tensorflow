# Copyright (c) 2022 Intel Corporation 
#
#Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
#you may not use this file except in compliance with the License.
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
"""Functional tests for pooling operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import os
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

class AvgPoolGradTest(test.TestCase):
  def _ConstructAndTestGradient(self,
                                input_sizes,
                                window_rows,
                                window_cols,
                                row_stride,
                                col_stride,
                                padding,
                                data_format,
                                use_gpu,
                                ):
    """Verifies the gradients of the max or avg pooling function.

    Args:
      input_sizes: Input tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
      data_format: Data format.
      use_gpu: whether we are running on GPU
    """

    with self.cached_session(use_gpu=use_gpu):
      total_size = 1
      for s in input_sizes:
        total_size *= s
      # Initializes the input tensor with array containing incrementing
      # numbers from 1.
      x = [f * 1.0 for f in range(1, total_size + 1)]
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_cols]
        strides = [1, 1, row_stride, col_stride]
        if isinstance(padding, list):
          padding = test_util.NHWCToNCHW(padding)
        input_tensor = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_cols, 1]
        strides = [1, row_stride, col_stride, 1]
      output = nn_ops.avg_pool(input_tensor, ksize, strides, padding, data_format)
      output_size = tf.shape(output).eval()
      total_size = 1
      for s in output_size:
        total_size *= s
      # Initializes the grad tensor with array containing incrementing
      # numbers from 1.
      y = [f * 1.0 for f in range(1, total_size + 1)]
      grad = constant_op.constant(y, shape=output_size)

    os.environ["ITEX_LAYOUT_OPT"]="0"
    with self.cached_session(use_gpu=use_gpu):
      result_plain = nn_ops.avg_pool_grad(
        input_sizes,
        grad,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format)
      result_plain = array_ops.identity(result_plain)
      result_plain = self.evaluate(result_plain)

    os.environ["ITEX_LAYOUT_OPT"]="1"
    with self.cached_session(use_gpu=use_gpu):
      result_block = nn_ops.avg_pool_grad(
        input_sizes,
        grad,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format)
      result_block = array_ops.identity(result_block)
      result_block = self.evaluate(result_block)

    # precision of plain AvgPoolGrad is tested in 
    # itex/test/tensorflow/python/kernel_tests/pooling_ops_test.py
    # so this ut just verify whether block layout is aligned with plain layout
    self.assertAllClose(result_plain, result_block)

  @test_util.run_deprecated_v1
  def testAvgPoolGrad(self):
    self._testAvgPoolGradValidPadding1_1("NHWC", True)
    self._testAvgPoolGradValidPadding1_2("NHWC", True)
    self._testAvgPoolGradValidPadding2_1("NHWC", True)
    self._testAvgPoolGradValidPadding2_2("NHWC", True)
    self._testAvgPoolGradSamePadding1_1("NHWC", True)
    self._testAvgPoolGradSamePadding1_2("NHWC", True)
    self._testAvgPoolGradSamePadding2_1("NHWC", True)
    self._testAvgPoolGradSamePadding2_2("NHWC", True)
    self._testAvgPoolGradSamePadding3_1("NHWC", True)

  def _testAvgPoolGradValidPadding1_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 3, 3, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding1_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 3, 3, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 3, 3, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 2, 2, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 2, 4, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 2, 4, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 2, 4, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_2(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[2, 2, 4, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding3_1(self, data_format, use_gpu):
    self._ConstructAndTestGradient(
        input_sizes=[1, 7, 7, 1],
        window_rows=3,
        window_cols=3,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

if __name__ == "__main__":
  test.main()
