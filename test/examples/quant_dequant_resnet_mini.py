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
from tensorflow.core.protobuf import rewriter_config_pb2
import numpy as np
config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=True,
                                  inter_op_parallelism_threads=1)
config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF


def test_quant():
    tf.compat.v1.disable_eager_execution()
    with tf.device("cpu"):
        with tf.compat.v1.Session(config=config) as sess:
            tf.random.set_seed(0)
            x = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(1, 3, 4, 1), name='input_x')
            w = tf.compat.v1.constant(
                0.04, dtype=tf.float32, shape=[3, 3, 1, 1], name="input_w")

            quant, _, _ = tf.quantization.quantize(
                x, 0, 4, tf.qint8, "SCALED", round_mode='HALF_TO_EVEN')
            x1 = tf.quantization.dequantize(
                quant, 0, 4, "SCALED", name="dequant")
            w1, _, _ = tf.quantization.quantize(w, 0, 4, tf.qint8, "SCALED", round_mode='HALF_TO_EVEN')
            w1 = tf.quantization.dequantize(w1, 0, 4, "SCALED")
            y1 = tf.nn.conv2d(
                x1, w1, (1, 1), "SAME", data_format='NHWC')
            y1 = tf.nn.relu(y1)
            y1, _, _ = tf.quantization.quantize(y1, 0, 4, tf.qint8, "SCALED", round_mode='HALF_TO_EVEN')
            y1 = tf.quantization.dequantize(
                y1, 0, 4, "SCALED", name="dequant_y")
            p = tf.identity(y1)

            sess.run(p, feed_dict={x: np.random.rand(
                1, 3, 4, 1)})

test_quant()
