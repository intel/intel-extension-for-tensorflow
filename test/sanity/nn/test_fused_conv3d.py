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
import os
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

tf.compat.v1.disable_eager_execution()

@test_util.run_all_in_native_and_block_format
class FusedConv3DTest(test_util.TensorFlowTestCase):
    """test _FusedITEXConv3D"""

    # _FusedITEXConv3D[T=DT_FLOAT, _XlaHasReferenceVars=false, data_format="NDHWC", dilations=[1, 1, 1, 1, 1],
    # fused_ops=["BiasAdd"], num_args=1, padding="SAME", strides=[1, 1, 1, 1, 1],
    # _device="/job:localhost/replica:0/task:0/device:XPU:0"]
    def testFuseBiasAdd(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 8, 7, 1))
        w = tf.compat.v1.placeholder(tf.float32, shape=(1, 2, 3, 1, 1))
        b = np.random.rand(1).astype(np.float32)

        x_arr = np.random.rand(1, 5, 8, 7, 1)
        w_arr = np.random.rand(1, 2, 3, 1, 1)
        with self.session(use_gpu=False) as sess:
            fused = tf.nn.bias_add(
                nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NDHWC'), b)
            ret_cpu = sess.run(array_ops.identity(fused), feed_dict={x: x_arr, w: w_arr})
        with self.session(use_gpu=True) as sess:
            fused = tf.nn.bias_add(
                nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NDHWC'), b)
            ret_gpu = sess.run(array_ops.identity(fused), feed_dict={x: x_arr, w: w_arr})
        self.assertAllClose(ret_cpu, ret_gpu)
        
    # _FusedITEXConv3D[T=DT_FLOAT, _XlaHasReferenceVars=false, data_format="NDHWC", dilations=[1, 1, 1, 1, 1],
    # fused_ops=["BiasAdd", "Relu"], num_args=1, padding="SAME", strides=[1, 1, 1, 1, 1],
    # _device="/job:localhost/replica:0/task:0/device:XPU:0"]
    def testFuseBiasAddAndRelu(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 8, 7, 1))
        w = tf.compat.v1.placeholder(tf.float32, shape=(1, 2, 3, 1, 1))
        b = np.random.rand(1).astype(np.float32)

        x_arr = np.random.rand(1, 5, 8, 7, 1)
        w_arr = np.random.rand(1, 2, 3, 1, 1)
        with self.session(use_gpu=False) as sess:
            conv3d = nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NDHWC')
            conv_bias = tf.nn.bias_add(conv3d, b)
            fused = nn_ops.relu(conv_bias)
            ret_cpu = sess.run(array_ops.identity(fused), feed_dict={x: x_arr, w: w_arr})
        with self.session(use_gpu=True) as sess:
            conv3d = nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NDHWC')
            conv_bias = tf.nn.bias_add(conv3d, b)
            fused = nn_ops.relu(conv_bias)
            ret_gpu = sess.run(array_ops.identity(fused), feed_dict={x: x_arr, w: w_arr})
        self.assertAllClose(ret_cpu, ret_gpu)

    def testFuseBiasAddAndAdd(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 8, 7, 1))
        w = tf.compat.v1.placeholder(tf.float32, shape=(1, 2, 3, 1, 1))
        b = np.random.rand(1).astype(np.float32)

        x_arr = np.random.rand(1, 5, 8, 7, 1)
        w_arr = np.random.rand(1, 2, 3, 1, 1)

        conv1 = nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        conv2 = nn_ops.Conv3D(input=conv1, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        conv = nn_ops.Conv3D(input=conv2, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        bias_add = nn_ops.bias_add(conv, b)
        output_1 = array_ops.identity(tf.math.add_n([bias_add, conv2]))
        output_2 = tf.math.add_n([bias_add, conv2])

        with self.session(use_gpu=True) as sess:
            ret_gpu = sess.run(output_1, feed_dict={x: x_arr, w: w_arr})
        with self.session(use_gpu=False) as sess:
            ret_cpu = sess.run(output_2, feed_dict={x: x_arr, w: w_arr})

        self.assertAllClose(ret_cpu, ret_gpu)

if __name__ == '__main__':
    test.main()
