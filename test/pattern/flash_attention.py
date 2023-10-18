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
from keras import backend as K
from tensorflow import keras
from tensorflow.python.framework import test_util
from tensorflow.python.framework import config
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import time
import os
import subprocess
import sys

tf.compat.v1.disable_eager_execution()

class AttentionInKerasStableDiffusionModel():
    def __init__(self, num_heads, head_size, use_itex=False):
        super().__init__()
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.use_itex = use_itex  

    def call(self, inputs):
        q,k,v = inputs

        if self.use_itex:
            q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
            k = tf.transpose(k, (0, 2, 1, 3))  # (bs, num_heads, head_size, time)
            v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

            from intel_extension_for_tensorflow.python.ops.multi_head_attention import scaled_dot_product_attention
            attn = scaled_dot_product_attention(q, k, v, use_fast_attention=True)
            out = tf.reshape(
                attn, (-1, q.shape[2], self.num_heads * self.head_size)
            )
            return out

        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = self.td_dot(q, k) * self.scale
        weights = keras.activations.softmax(
            score
        )  # (bs, num_heads, time, time)
        attn = self.td_dot(weights, v)
        attn = tf.transpose(
            attn, (0, 2, 1, 3)
        )  # (bs, time, num_heads, head_size)
        out = tf.reshape(
            attn, (-1, q.shape[2], self.num_heads * self.head_size)
        )
        return out

    def td_dot(self, a, b):
        aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
        bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
        cc = keras.backend.batch_dot(aa, bb)
        return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))

class MHAFusionForKerasStableDiffusionModelTest(test_util.TensorFlowTestCase):

    def testReshapeReshapeMatMulReshapeBiasAddRelu(self):
        tf.random.set_seed(0)
        datatypes = [tf.float32, tf.bfloat16]
        if config.list_logical_devices('XPU'):
            datatypes = [tf.float16, tf.bfloat16]

        q_in=np.random.rand(1, 1024, 8, 64)
        k_in=np.random.rand(1, 1024, 8, 64)
        v_in=np.random.rand(1, 1024, 8, 64)

        for dtype in datatypes:
            q = tf.constant(q_in, dtype=dtype)
            k = tf.constant(k_in, dtype=dtype)
            v = tf.constant(v_in, dtype=dtype)

            os.environ['ITEX_REMAPPER'] = '0'
            with self.session(use_gpu=True) as sess:
                tf.compat.v1.keras.backend.set_session(sess)
                model = AttentionInKerasStableDiffusionModel(q.shape[2], q.shape[3])
                real = model.call([q,k,v])
                real = sess.run(real)

            os.environ['ITEX_REMAPPER'] = '1'
            with self.session(use_gpu=True) as sess:
                tf.compat.v1.keras.backend.set_session(sess)
                model = AttentionInKerasStableDiffusionModel(q.shape[2], q.shape[3])
                predict = model.call([q,k,v])
                predict = sess.run(predict)

            print("real type: ", real.dtype)
            self.assertAllCloseAccordingToType(real, predict)


if __name__ == '__main__':
    test.main()
