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
from tensorflow.python.training import training_ops

class ApplyMomentumTest(TensorFlowTestCase):

  def _testTypesForApplyMomentum(self, x, y, lr, grad, momentum, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = VariableV1(x)
      accum = VariableV1(y)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_momentum = training_ops.apply_momentum(var, accum, lr, grad, momentum)
      out = self.evaluate(apply_momentum)
      self.assertShapeEqual(out, apply_momentum)
      self.assertAllCloseAccordingToType(x - lr * (y * momentum + grad), out)
      self.assertAllCloseAccordingToType(y * momentum + grad, self.evaluate(accum))

  @test_util.run_v1_only("ApplyMomentum op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyMomentum(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      momentum = np.array(0.1).astype(dtype)
      self._testTypesForApplyMomentum(x, y, lr, grad, momentum, use_gpu)

if __name__ == '__main__':
  googletest.main()
