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
from tensorflow.python.ops import variables
try:
  from tensorflow.python.ops.variables import VariableV1
except ImportError:
  from tensorflow.python.ops.variable_v1 import VariableV1
from tensorflow.python.platform import googletest
from tensorflow.python.ops import gen_training_ops as training_ops

class ApplyProximalAdagradTest(TensorFlowTestCase):

  def _testTypesForApplyProximalAdagrad(self, x, y, lr, l1, l2, grad, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = VariableV1(x)
      accum = VariableV1(y)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_proximal_adagrad = training_ops.apply_proximal_adagrad(var=var, accum=accum, lr=lr, 
                                                                   l1=l1, l2=l2, grad=grad)

      y += np.square(grad)
      learning_rate = lr * (1 / np.sqrt(y))
      prox_var = x
      prox_var -= grad * learning_rate
      if(l1 > 0 ):
        x = np.sign(prox_var) * (abs(prox_var) - learning_rate * l1) / (1 + l2 * learning_rate)
      else:
        x = prox_var / (1 + l2 * learning_rate)

      out = self.evaluate(apply_proximal_adagrad)
      self.assertShapeEqual(out, apply_proximal_adagrad)
      self.assertAllCloseAccordingToType(x, out)
      self.assertAllCloseAccordingToType(y, self.evaluate(accum))

  @test_util.run_v1_only("ApplyProximalAdagrad op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyProximalAdagrad(self):
    if not test.is_gpu_available():
      self.skipTest('CPU do not support')
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(0.0001).astype(dtype)
      l1 = np.array(0.001).astype(dtype)
      l2 = np.array(0.01).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForApplyProximalAdagrad(x, y, lr, l1, l2, grad, use_gpu)

if __name__ == '__main__':
  googletest.main()
