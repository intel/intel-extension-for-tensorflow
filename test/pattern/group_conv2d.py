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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.core.protobuf import config_pb2
import time
import os
import subprocess
import sys
import math

tf.compat.v1.disable_eager_execution()

def constant_xavier_initializer(shape, group):
    """Initializer function."""
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])/group
    else:
        fan_in = 1.0
        fan_out = 1.0
    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)

    # Average number of inputs and output connections.
    n = (fan_in + fan_out) / 2.0
    # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
    limit = math.sqrt(3.0 * 1.0 / n)
    return np.random.normal(-limit, limit, shape)

class GroupConv2DTest(test_util.TensorFlowTestCase):
    def test_group_conv(self):
        group=32
        weight_shape = [3, 3, 4, 128]
        input_shape = [1, 128, 58, 58]
        data_format = "channels_first"
        with self.session(use_gpu=True) as sess:
            weight_ = tf.constant(constant_xavier_initializer(weight_shape, group=group), dtype=tf.float32)
            weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1)
            reduced_inputs_relu = tf.random.normal(input_shape)
            xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=1)
            convolved = [tf.nn.conv2d(x, weight, padding='VALID', strides=[1, 1], data_format=('NCHW' if data_format == 'channels_first' else 'NHWC')) 
                for (x, weight) in zip(xs, weight_groups)]
            expected_result = tf.concat(convolved, axis=1)
            group_conv_result = tf.nn.conv2d(input=reduced_inputs_relu, filters=weight_, strides=[1, 1], padding='VALID', data_format=('NCHW' if data_format == 'channels_first' else 'NHWC'))
            self.assertAllClose(expected_result, group_conv_result)

if __name__ == '__main__':
    test.main()
