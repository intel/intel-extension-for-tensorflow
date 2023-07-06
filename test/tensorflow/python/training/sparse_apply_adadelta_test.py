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

import tensorflow as tf
from tensorflow.python.compat import v2_compat
v2_compat.disable_v2_behavior()

import numpy as np

from intel_extension_for_tensorflow.python.test_func.test_util import TensorFlowTestCase
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import training_ops

class SparseApplyAdadeltaTest(TensorFlowTestCase):

  def _testTypesForSparseApplyAdadelta(self, x, y, z, lr, rho, epsilon, grad, indices, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variables.VariableV1(x)
      accum = variables.VariableV1(y)
      accum_update = variables.VariableV1(z)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_adadelta = training_ops.sparse_apply_adadelta(var=var, accum=accum, accum_update=accum_update, lr=lr, 
                                                        rho=rho, epsilon=epsilon, grad=grad, indices=indices)
      out = self.evaluate(apply_adadelta)
      self.assertShapeEqual(out, apply_adadelta)

      tmp = np.zeros(100)
      for i in range(10):
        tmp[indices[i]] = grad[i]
      for i in range(10):
        y[indices[i]] = y[indices[i]] *  rho + np.square(tmp[i]) * (1 - rho)
      rsqrt = tf.math.rsqrt((y + epsilon))
      rsqrt = self.evaluate(rsqrt)
      update = np.sqrt((z + epsilon)) * rsqrt * tmp
      x = x - update * lr
      z = z * rho + np.square(update) * (1 - rho)
      self.assertAllCloseAccordingToType(x, out)
      self.assertAllCloseAccordingToType(y, self.evaluate(accum))

  @test_util.run_v1_only("ApplyAdadelta op returns a ref, so it is not "
                         "supported in eager mode.")
  def testSparseApplyAdadelta(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      z = np.arange(2, 102).astype(dtype)
      lr = np.array(0.1).astype(dtype)
      rho = np.array(0.95).astype(dtype)
      epsilon = np.array(1e-7).astype(dtype)
      grad = np.arange(10).astype(dtype)
      indices = np.arange(10).astype(np.int32)
      self._testTypesForSparseApplyAdadelta(x, y, z, lr, rho, epsilon, grad, indices, use_gpu)

if __name__ == '__main__':
  googletest.main()
