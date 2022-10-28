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



from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import training_ops
import numpy as np
"""Tests for ResourceApplyKerasMomentum."""
def keras_momentum_update_numpy(var, accum, g, lr, momentum):
  accum = accum * momentum - lr * g
  var += accum
  return var, accum

class ResourceApplyKerasMomentumTest(test.TestCase):

  def testResourceApplyKerasMomentum(self):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        v_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        accum_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        grads_np = v_np * 10
        
        # Initialize variables for tensor implementation.
        var_t = resource_variable_ops.ResourceVariable(v_np, dtype = dtype)
        accum_t = resource_variable_ops.ResourceVariable(accum_np, dtype = dtype)
        grads_t = constant_op.constant(grads_np, dtype = dtype)
        
        op_out = training_ops.resource_apply_keras_momentum(var=var_t.handle, accum=accum_t.handle, lr=2.0, grad=grads_t, momentum=0.9)
        self.evaluate(op_out)
        v_np, accum_np = keras_momentum_update_numpy(v_np, accum_np, grads_np, 2.0, 9.0)
        self.assertAllCloseAccordingToType(v_np, self.evaluate(var_t))

if __name__ == "__main__":
  test.main()
