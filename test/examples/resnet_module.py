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
    Resnet block with identity shortcut, this test will helps to capture 
    and test the following patterns 
    conv + bias + relu, conv + bias + relu + add, conv + bias
"""
from tensorflow.core.protobuf import rewriter_config_pb2
import json
from tensorflow.python.client import timeline
import getpass
import ctypes

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# using weights and bias as constants instead of variables, to mimic the frozen weights
weights = {
    'wc1': tf.compat.v1.constant(0.04, shape=[7, 7, 3, 64]),
    'wc2': tf.compat.v1.constant(0.05, shape=[1, 1, 64, 256]),
    'wc3': tf.compat.v1.constant(0.03, shape=[1, 1, 64, 64]),
    'wc4': tf.compat.v1.constant(0.09, shape=[3, 3, 64, 64]),
    'wc5': tf.compat.v1.constant(0.06, shape=[1, 1, 64, 256]),

    'wc6': tf.compat.v1.constant(0.06, shape=[1, 1, 256, 64]),
    'wc7': tf.compat.v1.constant(0.06, shape=[3, 3, 64, 64]),
    'wc8': tf.compat.v1.constant(0.06, shape=[1, 1, 64, 256])
}

biases = {
    'bias_1': tf.compat.v1.constant(0.01, shape=[64]),
    'bias_2': tf.compat.v1.constant(0.01, shape=[64]),
    'bias_3': tf.compat.v1.constant(0.01, shape=[64]),
    'bias_4': tf.compat.v1.constant(0.01, shape=[256]),
    'bias_5': tf.compat.v1.constant(0.01, shape=[256]),
    'bias_6': tf.compat.v1.constant(0.01, shape=[64]),
    'bias_7': tf.compat.v1.constant(0.01, shape=[64]),
    'bias_8': tf.compat.v1.constant(0.01, shape=[256])
}


def resnet(input):
    conv1 = tf.compat.v1.nn.conv2d(input, weights['wc1'], strides=[
                                   1, 2, 2, 1], padding='VALID')
    bias_add1 = tf.compat.v1.nn.bias_add(conv1, biases['bias_1'])
    relu1 = tf.compat.v1.nn.relu(bias_add1)

    maxpool = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[
                             1, 2, 2, 1], padding='SAME')

    conv2 = tf.compat.v1.nn.conv2d(maxpool, weights['wc3'], strides=[
                                   1, 1, 1, 1], padding='SAME')
    bias_add2 = tf.compat.v1.nn.bias_add(conv2, biases['bias_2'])
    relu2 = tf.compat.v1.nn.relu(bias_add2)

    conv4 = tf.compat.v1.nn.conv2d(relu2, weights['wc4'], strides=[
                                   1, 1, 1, 1], padding='SAME')
    bias_add4 = tf.compat.v1.nn.bias_add(conv4, biases['bias_3'])
    relu4 = tf.compat.v1.nn.relu(bias_add4)

    conv5 = tf.compat.v1.nn.conv2d(relu4, weights['wc5'], strides=[
                                   1, 1, 1, 1], padding='SAME')
    bias_add5 = tf.compat.v1.nn.bias_add(conv5, biases['bias_4'])

    conv3 = tf.compat.v1.nn.conv2d(maxpool, weights['wc2'], strides=[
                                   1, 1, 1, 1], padding='SAME')
    bias_add3 = tf.compat.v1.nn.bias_add(conv3, biases['bias_5'])

    relu5 = tf.nn.relu(tf.compat.v1.add(bias_add3, bias_add5))

    conv6 = tf.compat.v1.nn.conv2d(relu5, weights['wc6'], strides=[
                                   1, 1, 1, 1], padding='SAME')
    bias_add6 = tf.compat.v1.nn.bias_add(conv6, biases['bias_6'])
    relu6 = tf.compat.v1.nn.relu(bias_add6)

    conv7 = tf.compat.v1.nn.conv2d(relu6, weights['wc7'], strides=[
                                   1, 1, 1, 1], padding='SAME')
    bias_add7 = tf.compat.v1.nn.bias_add(conv7, biases['bias_7'])
    relu7 = tf.compat.v1.nn.relu(bias_add7)

    conv8 = tf.compat.v1.nn.conv2d(relu7, weights['wc8'], strides=[
                                   1, 1, 1, 1], padding='SAME')
    bias_add8 = tf.compat.v1.nn.bias_add(conv8, biases['bias_8'])

    return tf.compat.v1.add(bias_add8, relu5)


input = tf.compat.v1.placeholder(
    tf.float32, shape=(1, 230, 230, 3), name='input')

resnet_module = resnet(input)

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
    resnet_output = sess.run(resnet_module,
                             feed_dict={
                                 input: np.ones(((1, 230, 230, 3)))
                             },
                             options=options,
                             run_metadata=run_metadata)
    print(resnet_output)
