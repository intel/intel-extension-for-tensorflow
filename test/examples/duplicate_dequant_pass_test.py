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


def test_duplicate_dequant_pass():
    tf.compat.v1.disable_eager_execution()
    with tf.device("cpu"):
        with tf.compat.v1.Session(config=config) as sess:
            tf.random.set_seed(0)
            x = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(1, 2, 2, 1), name='input_x')
            y = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(1, 2, 2, 1), name='input_y')
            x1, _, _ = tf.quantization.quantize(
                x, 0, 4, tf.quint8, "SCALED", name="quant_x1")
            x2 = tf.quantization.dequantize(x1, 0, 4, "SCALED")

            x3 = tf.nn.max_pool(x2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', data_format='NHWC')
            x4, _, _ = tf.quantization.quantize(
                x3, 0, 1, tf.quint8, "SCALED", name="quant_x4")
            x5 = tf.quantization.dequantize(x4, 0, 1, "SCALED")

            x6 = tf.math.add(x2, y)
            x7 = tf.math.add(x5, x6)
            x8 = tf.identity(x7)
            sess.run(x8, feed_dict={x: np.random.rand(
                1, 2, 2, 1),
                y: np.random.rand(
                1, 2, 2, 1)})


test_duplicate_dequant_pass()
