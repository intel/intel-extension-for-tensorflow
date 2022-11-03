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

import numpy as np

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops

def resource_sparse_apply_proximal_adagrad_numpy(var, accum, lr, l1, l2, grad):
    accum += grad * grad 
    lr_update = lr / np.sqrt(accum)
    prox_v = var
    prox_v -= grad * lr_update
    var = np.sign(prox_v) * np.max(np.abs(prox_v) - lr_update * l1, 0) / (1 + lr_update * l2)
    return var

class ResourceSparseApplyProximalAdagradTest(test_util.TensorFlowTestCase):
  """test ResourceSparseApplyProximalAdagrad op"""

  def testResourceSparseApplyProximalAdagrad(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      var_np = np.array([2.0, 2.0], dtype=dtype.as_numpy_dtype)
      accum_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)

      var = resource_variable_ops.ResourceVariable(var_np)
      accum = resource_variable_ops.ResourceVariable(accum_np)
      lr = constant_op.constant(3.0, dtype=dtype)
      l1 = constant_op.constant(0.1, lr.dtype)
      l2 = constant_op.constant(0.1, lr.dtype)
      grad = constant_op.constant([0.0, 0.0], dtype=lr.dtype)
      indices = constant_op.constant([0, 1], dtype=dtypes.int32)
      training_ops.resource_sparse_apply_proximal_adagrad(
        var.handle, accum.handle, lr, l1, l2, grad, indices)
      var_np = resource_sparse_apply_proximal_adagrad_numpy(var_np, accum_np,
        lr, l1, l2, grad)
      self.assertAllCloseAccordingToType(var_np[1:], self.evaluate(var[1:]))

if __name__ == "__main__":
  test.main()
