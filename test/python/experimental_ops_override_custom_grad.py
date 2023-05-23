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
import intel_extension_for_tensorflow as itex
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
import time

# https://github.com/keras-team/keras/issues/18139
class CustomGradTest(test_util.TensorFlowTestCase):
    """test fused matmul"""
    def testMatMulReshapeBiasAddRelu(self):
        itex.experimental_ops_override()
        policy=tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        inputs = tf.keras.Input(shape=(1,3,))
        x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=array_ops.identity(x))
        itex_result = model(np.array([[[1.,2.,3.]],[[4.,5.,6.]]]).astype(np.float32))

if __name__ == '__main__':
    test.main()
