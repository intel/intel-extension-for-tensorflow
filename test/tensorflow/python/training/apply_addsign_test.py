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
"""Tests for tensorflow.learning.training_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util, test

import itertools

from tensorflow.python.compat import v2_compat
v2_compat.disable_v2_behavior()

import numpy as np

from intel_extension_for_tensorflow.python.test_func.test_util import TensorFlowTestCase
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import training_ops

class ApplyAddSignTest(TensorFlowTestCase):

  def _toType(self, dtype):
    if dtype == np.float16:
      return dtypes.float16
    elif dtype == np.float32:
      return dtypes.float32
    elif dtype == np.float64:
      return dtypes.float64
    elif dtype == np.int32:
      return dtypes.int32
    elif dtype == np.int64:
      return dtypes.int64
    else:
      assert False, (dtype)

  def _testTypesForApplyAddSign(self, x, y, lr, grad, alpha, sign_decay, beta, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variables.VariableV1(x)
      m = variables.VariableV1(y)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_add_sign = training_ops.apply_add_sign(var=var, m=m, lr=lr, alpha=alpha, sign_decay=sign_decay, 
                                                        beta=beta, grad=grad)
      out = self.evaluate(apply_add_sign)
      self.assertShapeEqual(out, apply_add_sign)
      
      y = y * beta + grad * (1 - beta)
      sign_gm = np.sign(grad) * np.sign(y)
      x = x - lr * (alpha + sign_decay * sign_gm) * grad

      self.assertAllCloseAccordingToType(x, out)
      self.assertAllCloseAccordingToType(y, self.evaluate(m))

  def _testTypesForResourceApplyAddSign(self, x, y, lr, grad, alpha, sign_decay, beta, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = resource_variable_ops.ResourceVariable(x)
      m = resource_variable_ops.ResourceVariable(y)

      y = y * beta + grad * (1 - beta)
      sign_gm = np.sign(grad) * np.sign(y)
      x = x - lr * (alpha + sign_decay * sign_gm) * grad

      lr = constant_op.constant(lr, self._toType(lr.dtype), [])
      alpha = constant_op.constant(alpha, self._toType(lr.dtype), [])
      sign_decay = constant_op.constant(sign_decay, self._toType(lr.dtype), [])
      beta = constant_op.constant(beta, self._toType(lr.dtype), [])
      self.evaluate(variables.global_variables_initializer())

      resource_apply_add_sign = training_ops.resource_apply_add_sign(var=var.handle, m=m.handle, lr=lr, alpha=alpha, sign_decay=sign_decay, 
                                                        beta=beta, grad=grad)
      out = self.evaluate(resource_apply_add_sign)

      self.assertAllCloseAccordingToType(self.evaluate(var), x)
      self.assertAllCloseAccordingToType(self.evaluate(m), y)  

  @test_util.run_v1_only("ApplyAddSign op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyAddSign(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      alpha = np.array(0.9).astype(dtype)
      sign_decay = np.array(0.999).astype(dtype)
      beta = np.array(1e-7).astype(dtype)
      self._testTypesForApplyAddSign(x, y, lr, grad, alpha, sign_decay, beta, use_gpu)

  @test_util.run_v1_only("ResourceApplyAddSign op returns a ref, so it is not "
                         "supported in eager mode.")
  def testResourceApplyAddSign(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(2.0, dtype=dtype)
      grad = np.arange(100, dtype=dtype)
      alpha = np.array(0.9, dtype=dtype)
      sign_decay = np.array(0.999, dtype=dtype)
      beta = np.array(1e-7, dtype=dtype)
      self._testTypesForResourceApplyAddSign(x, y, lr, grad, alpha, sign_decay, beta, use_gpu)

if __name__ == '__main__':
  googletest.main()
