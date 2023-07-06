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
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import tensorflow as tf
import intel_extension_for_tensorflow as itex

SHAPE = (5,2)
class LayerNormalizationTest(test_util.TensorFlowTestCase):
    """test layer normalization op"""

    def _testForwardPass(self, batch_input_shape, axis, fp32_tol=2e-4,
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
        result = []
        param_shape = [batch_input_shape[i] for i in axis]
        param_elems = 1
        for dim in param_shape:
            param_elems *= dim
        beta = np.arange(param_elems, dtype='float32').reshape(param_shape)
        gamma = np.arange(1, param_elems + 1, dtype='float32').reshape(param_shape)
        np.random.seed(5)
        x = np.random.normal(size=batch_input_shape)

        for epsilon in 1e-12, 1e-3:
            for dtype in 'float32', 'float16':
                '''wenjie: LayerNorm op does not register float16 on CPU
                   test case for float16 needs to be enabled when registration is done
                '''
                if not test.is_gpu_available() and dtype == 'float16':
                    continue
                tf_layer = tf.keras.layers.LayerNormalization(
                    axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
                    epsilon=epsilon, beta_initializer=tf.constant_initializer(beta),
                    gamma_initializer=tf.constant_initializer(gamma))
                tf_result = tf_layer(tf.cast(x, dtype))
                result.append((tf_result, fp32_tol if dtype == 'float32' else fp16_tol))
        return result

    def testForward(self):
        # For numeric stability, we ensure the axis's dimension(s) have at least 4
        # elements.
        res_tf = []
        res_itex = []
        res_tf += self._testForwardPass((4, 3), (0,))
        res_tf += self._testForwardPass((3, 4), (1,))
        res_tf += self._testForwardPass((4, 3, 2), (0,))
        res_tf += self._testForwardPass((2, 4, 2), (1,))
        res_tf += self._testForwardPass((4, 5, 6), (2,))
        res_tf += self._testForwardPass((2, 3, 2), (0, 2))
        res_tf += self._testForwardPass((2, 2, 2, 2), (1, 3))
        res_tf += self._testForwardPass((2, 2, 2, 2), (2, 3))
        res_tf += self._testForwardPass((2, 3, 4, 5), (3,))
        itex.experimental_ops_override()
        res_itex += self._testForwardPass((4, 3), (0,))
        res_itex += self._testForwardPass((3, 4), (1,))
        res_itex += self._testForwardPass((4, 3, 2), (0,))
        res_itex += self._testForwardPass((2, 4, 2), (1,))
        res_itex += self._testForwardPass((4, 5, 6), (2,))
        res_itex += self._testForwardPass((2, 3, 2), (0, 2))
        res_itex += self._testForwardPass((2, 2, 2, 2), (1, 3))
        res_itex += self._testForwardPass((2, 2, 2, 2), (2, 3))
        res_itex += self._testForwardPass((2, 3, 4, 5), (3,))
        for i in range(len(res_tf)):
            tol = res_tf[i][1]
            self.assertAllClose(res_tf[i][0], res_itex[i][0], rtol=tol, atol=tol)
if __name__ == "__main__":
    test.main()
