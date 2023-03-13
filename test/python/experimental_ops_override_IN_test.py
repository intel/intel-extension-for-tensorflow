# Copyright (c) 2023 Intel Corporation
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
import tensorflow as tf
import intel_extension_for_tensorflow as itex

from intel_extension_for_tensorflow.python.test_func import test as test_lib
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.python.framework import dtypes


class InstanceNormTest(test_lib.TestCase):

  def _compute_instance_norm(self, input_shape, dtype=dtypes.float32, axis=-1):
    keras.utils.set_random_seed(1)
    x_array = np.random.randn(*input_shape).astype(np.float32)
    x = tf.Variable(x_array, dtype=dtype, trainable=True)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    beta_init = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    layer = tfa.layers.InstanceNormalization(dtype=dtype, axis=axis,
                                             gamma_initializer=gamma_init,
                                             beta_initializer=beta_init)
    with tf.GradientTape(persistent=True) as tape:
        outputs = layer(x)
        loss = tf.reduce_sum(outputs * 2)
    dx = tape.gradient(loss, x)
    dweights = tape.gradient(loss, layer.trainable_variables)
    dweights = np.stack(dweights)
    return outputs, dx, dweights

  def testInstanceNorm(self):
    input_params = [
       [(2, 3, 5, 6, 4), 3],
       [(2, 8, 8, 4), 2],
       [(3, 3, 5, 6), -1],
       [(4, 6, 8), -1],
       [(2, 8), -1],
    ]
    test_types = [dtypes.float32, dtypes.bfloat16]
    ref_outputs = []
    ref_dx = []
    ref_dweights = []
    tols = []
    for dtype in test_types:
      for input_shape, axis in input_params:
        outputs, dx, dweights = self._compute_instance_norm(input_shape, dtype, axis)
        ref_outputs.append(outputs)
        ref_dx.append(dx)
        ref_dweights.append(dweights)
        tol = 3e-1 if dtype==dtypes.bfloat16 else 1e-5
        tols.append(tol)

    # Activate keras layer optimization.
    # Enable ITEX InstanceNorm call function.
    itex.experimental_ops_override()

    opt_outputs = []
    opt_dx = []
    opt_dweights = []
    for dtype in test_types:
      for input_shape, axis in input_params:
        outputs, dx, dweights = self._compute_instance_norm(input_shape, dtype, axis)
        opt_outputs.append(outputs)
        opt_dx.append(dx)
        opt_dweights.append(dweights)

    for i in range(len(opt_outputs)):
      self.assertAllClose(opt_outputs[i], ref_outputs[i], rtol=tols[i], atol=tols[i])
      self.assertAllClose(opt_dx[i], ref_dx[i], rtol=tols[i], atol=tols[i])
      self.assertAllClose(opt_dweights[i], ref_dweights[i], rtol=tols[i], atol=tols[i])


if __name__ == "__main__":
  test_lib.main()
