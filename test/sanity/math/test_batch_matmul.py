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
import tensorflow as tf
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

tf.compat.v1.disable_eager_execution()
def GetRandomNormalInput(shape, dtype=np.float64):
    scale = 0.1
    loc = 0.1
    vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    return vals.reshape(shape)

@test_util.run_all_in_native_and_block_format
class BatchMatMulTest(test_util.TensorFlowTestCase):
    """test BatchMatMul op."""

    # Uses numpy to compute batch_matmul(x, y, adjoint_a, adjoint_b).
    def _npBatchMatmul(self, x, y, adjoint_a, adjoint_b):
        # output's shape depends on adj[0] and adj[1]
        if adjoint_a:
            x = np.conjugate(np.swapaxes(x, -1, -2))
        if adjoint_b:
            y = np.conjugate(np.swapaxes(y, -1, -2))
        return np.matmul(x, y)

    # Compares TensorFlow BatchMatmul with NumPy's matmul.
    def _compare(self, dtype, x_in, y_in, adjoint_a, adjoint_b, static_shape):
        x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
        y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
        x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
        y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
        x2 = x + 1.
        with self.cached_session(use_gpu=True):
            dtype = dtype
            x1 = tf.cast(x, dtype=dtype)
            y1 = tf.cast(y, dtype=dtype)
            x2 = tf.cast(x2, dtype=dtype)
            x3 = math_ops.add_n([x1,x2])
            z0 = array_ops.identity(math_ops.matmul(x3, y1, adjoint_a=adjoint_a, adjoint_b=adjoint_b))
            z0_val = self.evaluate(z0)
            z1 = self._npBatchMatmul(2 * x + 1., y, adjoint_a, adjoint_b)
            self.assertAllClose(z0_val, z1, rtol=1e-2, atol=1e-2)

    def _testNonEmpty(self, dtype, adjoint_a, adjoint_b):

        def CompareNonEmpty(self, a_shape, b_shape):
            self._compare(
                dtype,
                GetRandomNormalInput(a_shape),
                GetRandomNormalInput(b_shape),
                adjoint_a,
                adjoint_b,
                static_shape=True)

        CompareNonEmpty(self, [2, 3], [3, 5])
        CompareNonEmpty(self, [2, 2, 3], [1, 3, 5])
        CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
        CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
        CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
        CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
        CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
        CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
        CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])
        CompareNonEmpty(self, [3, 2, 2, 3], [3, 1, 3, 5])
        CompareNonEmpty(self, [5, 7, 2, 3], [5, 7, 3, 5])
        CompareNonEmpty(self, [5, 7, 2, 3], [5, 1, 3, 5])

    def testBf16(self):
        for adjoint_a_ in False, True:
            for adjoint_b_ in False, True:
                self._testNonEmpty(tf.bfloat16, adjoint_a_, adjoint_b_)
 
    def testFp64(self):
        for adjoint_a_ in False, True:
            for adjoint_b_ in False, True:
                self._testNonEmpty(tf.double, adjoint_a_, adjoint_b_)


if __name__ == "__main__":
    test.main()
