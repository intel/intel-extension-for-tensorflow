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

import itertools
import numpy as np

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.ops import variables
from tensorflow.python.training import training_ops

class ApplyAdagradV2Test(test_util.TensorFlowTestCase):
  """test ApplyAdagradV2 op"""

  def _testTypesForAdagradV2(self, var, accum, lr, epsilon, grad, use_gpu=None):
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variables.VariableV1(var)
      accum = variables.VariableV1(accum)
      self.evaluate(variables.global_variables_initializer())
      apply_adagradv2 = training_ops.apply_adagradv2(var, accum, lr, epsilon, grad)
      out = self.evaluate(apply_adagradv2)
      self.assertShapeEqual(out, apply_adagradv2)
      self.assertAllCloseAccordingToType(var - lr * grad * (1 / np.sqrt(accum)), out)
      self.assertAllCloseAccordingToType(accum + grad * grad, self.evaluate(accum))

  @test_util.run_v1_only("ApplyAdagradV2 op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyAdagradV2(self):
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdagradV2(x, y, lr, 0.1, grad, use_gpu)

if __name__ == "__main__":
  test.main()
