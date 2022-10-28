# Copyright (c) 2022 Intel Corporation
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
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging

_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y


class BlockBinaryOpTest(test.TestCase):

  def _SetupConv2DBinary(self, inx, inw, data_format):
    conv = nn_ops.conv2d(
        inx,
        inw,
        dilations=(1, 1),
        strides=[1, 1],
        padding="SAME",
        data_format=data_format)
    return conv

  def _compareConvBinary(self, xs, ys, dtype, ws, data_format, np_func, tf_func, use_gpu=True, is_x_first=True, is_bf16=False):
    x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
    y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    w = (1 + np.linspace(0, 5, np.prod(ws))).astype(dtype).reshape(ws)

    np_x = x
    with self.session(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      inw = ops.convert_to_tensor(w)
      if is_bf16:
        inx = tf.cast(inx, tf.bfloat16)
        inw = tf.cast(inw, tf.bfloat16)
      x_conv = self._SetupConv2DBinary(inx, inw, data_format=data_format)
      out = array_ops.identity(x_conv)
      if is_bf16:
        out = tf.cast(out, tf.bfloat16)
      out = self.evaluate(out)
      np_x = out
    
    if is_x_first:
      np_ans = np_func(np_x, y)
    else:
      np_ans = np_func(y, np_x)

    with self.session(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      inw = ops.convert_to_tensor(w)
      iny = ops.convert_to_tensor(y)
      if is_bf16:
        inx = tf.cast(inx, tf.bfloat16)
        inw = tf.cast(inw, tf.bfloat16)
        iny = tf.cast(iny, tf.bfloat16)
      x_conv = self._SetupConv2DBinary(inx, inw, data_format=data_format)
      if is_x_first:
        out = tf_func(x_conv, iny)
      else:
        out = tf_func(iny, x_conv)
      out = array_ops.identity(out)
      if is_bf16:
        out = tf.cast(out, tf.bfloat16)
      tf_gpu = self.evaluate(out)
    if is_bf16:
      self.assertAllClose(np_ans, tf_gpu, rtol=5e-2, atol=5e-2)
    elif dtype == np.float16:
      self.assertAllClose(np_ans, tf_gpu, rtol=5e-3, atol=5e-3)
    else:
      self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)

  @test_util.run_deprecated_v1
  def testConvBinaryBCast(self):
    dtypes = [
      np.float16,
      np.float32,
    ]
    funcs = [
      (np.add, math_ops.add),
      (np.add, _ADD),
      (np.subtract, math_ops.subtract),
      (np.subtract, _SUB),
      (np.multiply, math_ops.multiply),
      (np.multiply, _MUL),
    ]
    for (np_func, tf_func) in funcs:
      self._compareConvBinary([1, 1, 1, 12], [1, 112, 112, 48], np.float32, [1, 1, 12, 48], "NHWC", np_func, tf_func, use_gpu=True, is_bf16=True, is_x_first=True)
      for dtype in dtypes:
        self._compareConvBinary([1, 1, 1, 12], [1, 112, 112, 48], dtype, [1, 1, 12, 48], "NHWC", np_func, tf_func, use_gpu=True, is_bf16=False, is_x_first=True)


if __name__ == "__main__":
  test.main()
