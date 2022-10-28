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
"""Tests for ResourceApplyAdamMax."""

def adamx_update_numpy(param,
                      g_t, 
                      t,
                      m, 
                      v, 
                      learning_rate=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = np.maximum(beta2 * v, np.abs(g_t));
  param_t = param - learning_rate / (1 - beta1**t) * m_t / (v_t + epsilon)
  return param_t, m_t, v_t


class AdaMaxTest(test.TestCase):

  def doTestBasic(self, use_resource=True):
    if context.executing_eagerly() and not use_resource:
      self.skipTest(
          "Skipping test with use_resource=False and executing eagerly.")
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        v_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        t = 1
        beta1 = np.array(0.9, dtype=dtype.as_numpy_dtype)
        beta2 = np.array(0.999, dtype=dtype.as_numpy_dtype)
        beta1_power = beta1**t
        lr = np.array(0.001, dtype=dtype.as_numpy_dtype)
        epsilon = np.array(1e-8, dtype=dtype.as_numpy_dtype)
        if use_resource:
          var = resource_variable_ops.ResourceVariable(var_np, dtype = dtype)
          m = resource_variable_ops.ResourceVariable(m_np, dtype = dtype)
          v = resource_variable_ops.ResourceVariable(v_np, dtype = dtype)
        else:
          var = variables.RefVariable(var_np, dtype = dtype)
          m = variables.RefVariable(m_np, dtype = dtype)
          v = variables.RefVariable(v_np, dtype = dtype)
        beta1_t = constant_op.constant(beta1, dtype = dtype)
        beta2_t = constant_op.constant(beta2, dtype = dtype)
        beta1_power_t = variables.VariableV1(beta1_power, dtype = dtype)
        lr_t = constant_op.constant(lr, dtype = dtype)
        epsilon_t = constant_op.constant(epsilon, dtype = dtype)
        grads_t = constant_op.constant(grads_np, dtype = dtype)
        self.evaluate(variables.global_variables_initializer())
        if use_resource:
          op_out = training_ops.resource_apply_ada_max(var.handle, m.handle, v.handle, beta1_power_t, lr_t, beta1_t, beta2_t, epsilon_t, grads_t)
        else:
          op_out = training_ops.apply_ada_max(var, m, v, beta1_power_t, lr_t, beta1_t, beta2_t, epsilon_t, grads_t)
        self.evaluate(op_out)
        var_np, m_np, v_np = adamx_update_numpy(var_np, grads_np, t, m_np, v_np, lr, beta1, beta2, epsilon)
        self.assertAllCloseAccordingToType(var_np, self.evaluate(var))
  
  @test_util.deprecated_graph_mode_only
  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_tfrt("b/168527439: invalid runtime fallback "
                          "resource variable reference on GPU.")
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)


if __name__ == "__main__":
  test.main()
