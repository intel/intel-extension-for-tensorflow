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
"""DNNL_GRAPH TensorFlow conv->relu->conv->relu
"""
import json
from tensorflow.python.client import timeline
import getpass
import ctypes

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# conv2d wrapper for tensorflow keras conv+bias+relu API

def conv2d(input, filter):
    output = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(output)


# construct conv2d with relu layer
input = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 4, 1), name='input')

# TODO(pruthvi): replace filter and bias with variables instead of placeholders
filter = tf.compat.v1.placeholder(
    tf.float32, shape=(3, 3, 1, 1), name='filter')

filter1 = tf.compat.v1.placeholder(
    tf.float32, shape=(3, 3, 1, 1), name='filter1')

conv1 = conv2d(input, filter)
conv2 = conv2d(conv1, filter1)


# Configure the session
config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  inter_op_parallelism_threads=1)

# Create session and run
with tf.compat.v1.Session() as sess:
    print("Python: Running with Session")
    options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    conv_output = sess.run(conv2,
                           feed_dict={
                               input: np.ones(((1, 3, 4, 1))),
                               filter: np.ones((3, 3, 1, 1)),
                               filter1: np.ones((3, 3, 1, 1)),
                           },
                           options=options,
                           run_metadata=run_metadata)
    print(conv_output)
