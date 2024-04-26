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

import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
import numpy as np
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import tensorflow as tf
import intel_extension_for_tensorflow as itex
import tf_keras as keras

SHAPE = (5,2)
np.random.seed(1)
class LayerNormalizationTest(test_util.TensorFlowTestCase):
    """test layer normalization op"""

    def _testForwardPass(self, batch_input_shape, axis, fp32_tol=1e-4,
                            fp16_tol=1e-2):
        """Tests the forward pass of layer layer_normalization.

        Args:
        batch_input_shape: The input shape that will be used to test, including
            the batch dimension.
        axis: A list of axises to normalize. Will be passed to the `axis` argument
            of Layerlayer_normalization.
        fp32_tol: The relative and absolute tolerance for float32.
        fp16_tol: The relative and absolute tolerance for float16.
        """
        param_shape = [batch_input_shape[i] for i in axis]
        param_elems = 1
        for dim in param_shape:
            param_elems *= dim
        beta = np.arange(param_elems, dtype='float32').reshape(param_shape)
        gamma = np.arange(1, param_elems + 1, dtype='float32').reshape(param_shape)
        x = np.random.normal(size=batch_input_shape)

        for epsilon in 1e-12, 1e-3:
            for dtype in 'float32', 'float16':
                '''wenjie: LayerNorm op does not register float16 on CPU
                   test case for float16 needs to be enabled when registration is done
                '''
                if not test.is_gpu_available() and dtype == 'float16':
                    continue
                tf_layer = keras.layers.LayerNormalization(
                    axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
                    epsilon=epsilon, beta_initializer=tf.constant_initializer(beta),
                    gamma_initializer=tf.constant_initializer(gamma))
                tf_result = tf_layer(tf.cast(x, dtype))
                itex_layer = itex.ops.LayerNormalization(
                    axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
                    epsilon=epsilon, beta_initializer=tf.constant_initializer(beta),
                    gamma_initializer=tf.constant_initializer(gamma))
                itex_result = itex_layer(tf.cast(x, dtype))

                if dtype == 'float32':
                    tol = fp32_tol
                else:
                    assert dtype == 'float16'
                    tol = fp16_tol

                # We use absolute tolerances in addition to relative tolerances, because
                # some of the values are very close to zero.
                self.assertAllClose(tf_result, itex_result, rtol=tol, atol=tol)

    def testForward(self):
        # For numeric stability, we ensure the axis's dimension(s) have at least 4
        # elements.
        self._testForwardPass((4, 3), (0,))
        self._testForwardPass((3, 4), (1,))
        self._testForwardPass((4, 3, 2), (0,))
        self._testForwardPass((2, 4, 2), (1,))
        self._testForwardPass((4, 5, 6), (2,))
        self._testForwardPass((2, 3, 2), (0, 2))
        self._testForwardPass((2, 2, 2, 2), (1, 3))
        self._testForwardPass((2, 2, 2, 2), (2, 3))
        self._testForwardPass((2, 3, 4, 5), (3,))

if __name__ == "__main__":
    test.main()
