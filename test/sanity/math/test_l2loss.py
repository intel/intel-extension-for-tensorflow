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

import tensorflow as tf
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops


class L2lossTest(test_util.TensorFlowTestCase):
    """Test GPU l2loss op."""
    def _testHelper(self, feature, use_gpu, dtype):
        with self.session(use_gpu=use_gpu):
            x = tf.constant(feature, dtype=dtype)
            y = nn_ops.l2_loss(x)
            return self.evaluate(y)

    def testL2lossFp32(self):
        for size in [1, 4096, 1 << 18, 1 << 18 + 3]:
            feature = np.random.normal(size=[size])
            np_res = 0.5 * np.sum(np.square(feature))
            gpu_res = self._testHelper(feature, True, tf.float32)
            self.assertAllClose(np_res, gpu_res)
            
    def testShapeInBert(self):
        for size in [3072 * 768, 4096 * 768, 768 * 768]:
            feature = np.random.normal(size=[size])
            np_res = 0.5 * np.sum(np.square(feature))
            gpu_res = self._testHelper(feature, True, tf.float32)
            self.assertAllClose(np_res, gpu_res)            
            
            
    def testL2lossFp64(self):
        for size in [1, 4096, 1<<18, 1<<18+3]:
            feature = np.random.normal(size=[size]).astype(np.float64)
            np_res = 0.5 * np.sum(np.square(feature))
            gpu_res = self._testHelper(feature, True, tf.float64)
            self.assertAllClose(np_res, gpu_res)     

    def testL2lossHalfPrecision(self):
        for size in [1, 4096]:
            for dtype in [tf.float16, tf.bfloat16]:
                feature = np.random.normal(size=[size])
                np_res = 0.5 * np.sum(np.square(feature))
                gpu_res = self._testHelper(feature, True, dtype)
                self.assertAllClose(np_res, gpu_res, rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
    test.main()
