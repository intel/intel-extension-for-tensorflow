# ==============================================================================
#  Copyright 2018-2022 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""
    DNNL_GRAPH TensorFlow Conv + BatchNorm + Relu 
"""
from tensorflow.core.protobuf import rewriter_config_pb2
import json
from tensorflow.python.client import timeline
import getpass
import ctypes

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def conv2d(input, filter, strides, padding_mode):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding_mode)


def batch_norm(input):
    """
    TODO(Pruthvi) : Refactor this to make Bn for any given depth
    batchnorm of depth 1 and input should be channel of dim_size = 1
    """
    scale = [1.0]
    offset = [0.2]
    mean = [0.4]
    variance = [0.1]

    batchnorm = tf.compat.v1.nn.fused_batch_norm(
        input,
        scale,
        offset,
        mean,
        variance,
        data_format='NHWC',
        is_training=False)
    return batchnorm


def conv2dBatchNormalizationWithRelu(input, filter, conv_strides, padding_mode, bn_depth):
    conv_output = conv2d(input, filter,  conv_strides, padding_mode)
    bn_output = batch_norm(conv_output)
    return tf.nn.relu(bn_output[0])


def conv2dBatchNormalization(input, filter, conv_strides, padding_mode, bn_depth):
    conv_output = conv2d(input, filter,  conv_strides, padding_mode)
    return batch_norm(conv_output)[0]


input = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 4, 1), name='input')

filter = tf.compat.v1.placeholder(
    tf.float32, shape=(3, 3, 1, 1), name='filter')

conv1_bn = conv2dBatchNormalization(
    input, filter, [1, 1, 1, 1], 'VALID', bn_depth=1)

# Configure the sessios, we will turn off the remapper graph optimizer
graph_options = tf.compat.v1.GraphOptions(
    rewrite_options=rewriter_config_pb2.RewriterConfig(
        remapping=rewriter_config_pb2.RewriterConfig.OFF))

config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1,
                                  graph_options=graph_options)


# Create session and run
with tf.compat.v1.Session() as sess:
    print("Python: Running with Session")
    options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    conv_bn_output = sess.run(conv1_bn,
                              feed_dict={
                                  input: np.ones(((1, 3, 4, 1))),
                                  filter: np.ones((3, 3, 1, 1)),
                              },
                              options=options,
                              run_metadata=run_metadata)
    print(conv_bn_output)
