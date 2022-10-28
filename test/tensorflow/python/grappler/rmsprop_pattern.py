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



from absl.testing import parameterized
import itertools
import numpy as np

from intel_extension_for_tensorflow.python.test_func import test as test_lib
from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2


_DATA_TYPES = [
    dtypes.float32,
]

_TEST_PARAM_VALUES = [
    # learning_rate, rho, momentum, epsilon, centered
    [0.05, 0.9, 0.0, 1e-3, False],
]

_TESTPARAMS = [
    [data_type] + values
    for data_type, values in itertools.product(_DATA_TYPES, _TEST_PARAM_VALUES)
]

class RMSpropOptimizerTest(test_lib.TestCase, parameterized.TestCase):
  def _rmsprop_update_numpy(self, var, g, mg, rms, mom, lr, rho, momentum,
                            epsilon, centered):
    rms_t = rms * rho + (1 - rho) * g * g
    if centered:
      mg_t = mg * rho + (1 - rho) * g
      denom_t = rms_t - mg_t * mg_t
    else:
      mg_t = mg
      denom_t = rms_t
    if momentum > 0.:
      mom_t = momentum * mom + lr * g / (np.sqrt(denom_t + epsilon))
      var_t = var - mom_t
    else:
      mom_t = mom
      var_t = var - lr * g / (np.sqrt(denom_t) + epsilon)
    return var_t, mg_t, rms_t, mom_t


  @combinations.generate(combinations.combine(mode=['graph']))
  def testDense(self):
    if not test_lib.is_gpu_available():
      self.skipTest("Skip on CPU due to the pattern not supported")
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    for (dtype, learning_rate, rho, momentum, epsilon, centered) in _TESTPARAMS:
      with test_util.force_gpu():
        # Initialize variables for numpy implementation.
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.2], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.2], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np, dtype=dtype)
        var1 = variables.Variable(var1_np, dtype=dtype)
        grads0 = constant_op.constant(grads0_np, dtype=dtype)
        grads1 = constant_op.constant(grads1_np, dtype=dtype)
        opt = rmsprop.RMSprop(
          learning_rate=learning_rate,
          rho=rho,
          momentum=momentum,
          epsilon=epsilon,
          centered=centered)

        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        if centered:
          mg0 = opt.get_slot(var0, "mg")
          mg1 = opt.get_slot(var1, "mg")
        else:
          mg0 = None
          mg1 = None

        if momentum > 0.:
          mom0 = opt.get_slot(var0, "momentum")
          mom1 = opt.get_slot(var1, "momentum")
        else:
          mom0 = None
          mom1 = None

        rms0 = opt.get_slot(var0, "rms")
        self.assertIsNotNone(rms0)
        rms1 = opt.get_slot(var1, "rms")
        self.assertIsNotNone(rms1)

        mg0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mg1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        rms1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        mom1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of RMSprop
        for _ in range(1, 4):
          self.evaluate(update)

          var0_np, mg0_np, rms0_np, mom0_np = self._rmsprop_update_numpy(
              var0_np, grads0_np, mg0_np, rms0_np, mom0_np, learning_rate, rho,
              momentum, epsilon, centered)
          var1_np, mg1_np, rms1_np, mom1_np = self._rmsprop_update_numpy(
              var1_np, grads1_np, mg1_np, rms1_np, mom1_np, learning_rate, rho,
              momentum, epsilon, centered)

          # Validate updated params
          if centered:
            self.assertAllCloseAccordingToType(mg0_np, self.evaluate(mg0))
            self.assertAllCloseAccordingToType(mg1_np, self.evaluate(mg1))
          if momentum > 0.:
            self.assertAllCloseAccordingToType(mom0_np, self.evaluate(mom0))
            self.assertAllCloseAccordingToType(mom1_np, self.evaluate(mom1))
          self.assertAllCloseAccordingToType(rms0_np, self.evaluate(rms0))
          self.assertAllCloseAccordingToType(rms1_np, self.evaluate(rms1))
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @combinations.generate(combinations.combine(mode=['graph']))
  def testGraphStructure(self):
    if not test_lib.is_gpu_available():
      self.skipTest("Skip on CPU due to the pattern not supported")
    with test_util.force_gpu():
      dtype = dtypes.float32
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.2], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.2], dtype=dtype.as_numpy_dtype)

      var0 = variables.Variable(var0_np, dtype=dtype)
      var1 = variables.Variable(var1_np, dtype=dtype)
      grads0 = constant_op.constant(grads0_np, dtype=dtype)
      grads1 = constant_op.constant(grads1_np, dtype=dtype)

      learning_rate = 0.05
      rho = 0.9
      momentum = 0.0
      epsilon = 1e-3
      centered = False

      opt = rmsprop.RMSprop(
        learning_rate=learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
        centered=centered)

      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      metadata = config_pb2.RunMetadata()
      with self.session() as sess:
        sess.run(variables.global_variables_initializer())
        output_val = sess.run(update, options=run_options, run_metadata=metadata)
        graph = metadata.partition_graphs[0]

      exist_compute_rms = False
      exist_var_update = False
      for node in graph.node:
        if 'ApplyRMSPropComputeRMS' in node.op:
          exist_compute_rms = True
        if 'ApplyRMSPropVarUpdate' in node.op:
          exist_var_update = True

      self.assertTrue(exist_compute_rms)
      self.assertTrue(exist_var_update)

if __name__ == "__main__":
  test_lib.main()
