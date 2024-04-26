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

import os
os.environ['TF_USE_LEGACY_KERAS']='1'

import intel_extension_for_tensorflow as itex
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

class AdamWTest(test_util.TensorFlowTestCase):
    """test AdamW op"""
    def testAdamW(self):        
        # Initialize variables for numpy implementation and Create Tensorflow variables.
        size = [2,4,3]
        dtype = dtypes.float32
        var0 = np.random.normal(size=size)
        var1 = np.random.normal(size=size)
        grads0 = np.random.normal(size=size)
        grads1 = np.random.normal(size=size)
        # tf
        tf_var0 = tf.Variable(var0, dtype=dtype)
        tf_var1 = tf.Variable(var1, dtype=dtype)
        tf_grads0 = constant_op.constant(grads0, dtype=dtype)
        tf_grads1 = constant_op.constant(grads1, dtype=dtype)
        tf_adamw = tf.keras.optimizers.AdamW(weight_decay=0.04, learning_rate=0.01)  
        for _ in range(3):  # Run 3 steps of the optimizer
            tf_adamw.apply_gradients(
                zip([tf_grads0, tf_grads1], [tf_var0, tf_var1])
            )
        # itex
        itex.experimental_ops_override()
        itex_var0 = tf.Variable(var0, dtype=dtype)
        itex_var1 = tf.Variable(var1, dtype=dtype)
        itex_grads0 = constant_op.constant(grads0, dtype=dtype)
        itex_grads1 = constant_op.constant(grads1, dtype=dtype)
        itex_adamw = tf.keras.optimizers.AdamW(weight_decay=0.04, learning_rate=0.01)
        for _ in range(3):  # Run 3 steps of the optimizer
            itex_adamw.apply_gradients(
                zip([itex_grads0, itex_grads1], [itex_var0, itex_var1])
            )
        # Validate updated parameters
        self.assertAllClose(tf_var0, itex_var0)
        self.assertAllClose(tf_var1, itex_var1)

if __name__ == "__main__":
  test.main()
