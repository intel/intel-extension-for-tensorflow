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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import itertools

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class BiasAddTest(test_util.TensorFlowTestCase):
    """test BiasAdd op"""

    def _testHelper(self, use_gpu, dtype):
        # set both the global and operation-level seed to ensure this result is reproducible
        with self.session(use_gpu=use_gpu):
            tf.random.set_seed(5)
            x1 = tf.random.normal(shape=[1, 2, 2, 3], seed=1, dtype=dtype)
            x2 = tf.random.normal(shape=[3], seed=1, dtype=dtype)
            return self.evaluate(nn_ops.bias_add(x1, x2))

    def testBiasAddFp32(self):
        ans_cpu = self._testHelper(False, tf.dtypes.float32)
        ans_gpu = self._testHelper(True, tf.dtypes.float32)
        self.assertAllClose(ans_cpu, ans_gpu)

    def testBiasAddFp16(self):
        ans_cpu = self._testHelper(False, tf.dtypes.float16)
        ans_gpu = self._testHelper(True, tf.dtypes.float16)
        self.assertAllClose(ans_cpu, ans_gpu)

    def testBiasAddBf16(self):
        ans_cpu = self._testHelper(False, tf.dtypes.bfloat16)
        ans_gpu = self._testHelper(True, tf.dtypes.bfloat16)
        self.assertAllClose(ans_cpu, ans_gpu)
    
    def testBiasAddDouble(self):
        ans_cpu = self._testHelper(False, tf.dtypes.double)
        ans_gpu = self._testHelper(True, tf.dtypes.double)
        self.assertAllClose(ans_cpu, ans_gpu)


if __name__ == '__main__':
    test.main()
