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
"""Tests for Dequantize + Shape fusion"""

import numpy as np
import os
import tensorflow as tf

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2

@test_util.run_all_in_graph_and_eager_modes
class DequantizeWithShapeTest(test.TestCase):
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    x_f32_np = np.random.uniform(
        low=-0.0, high=5.0, size=(1, 6, 6, 4)).astype(np.float32)

    x_f32 = array_ops.placeholder(np.float32, [None, 6, 6, 4])

    x_min = tf.math.reduce_min(x_f32)
    x_max = tf.math.reduce_max(x_f32)
    x_int8, x_min, x_max = array_ops.quantize(
        x_f32, x_min, x_max, T=dtypes.quint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True)

    y_f32_np = np.random.uniform(
        low=-2.0, high=2.0, size=(3, 3, 4, 4)).astype(np.float32)
    y_f32 = constant_op.constant(y_f32_np)

    y_min = tf.math.reduce_min(y_f32, axis=(0, 1, 2))
    y_max = tf.math.reduce_max(y_f32, axis=(0, 1, 2))
    y_int8, y_min, y_max = array_ops.quantize(
        y_f32, y_min, y_max, T=dtypes.qint8, mode="SCALED", round_mode="HALF_TO_EVEN", narrow_range=True, axis=3)

    bias_f32_np = np.random.uniform(
        low=-1, high=1.0, size=(4)).astype(np.float32)
    bias_f32 = constant_op.constant(bias_f32_np)

    conv_f32 = tf.nn.bias_add(tf.nn.conv2d(
        x_f32, y_f32, [1, 1, 1, 1], padding="SAME"), bias_f32)

    z_freezed_min = tf.math.reduce_min(conv_f32)
    z_freezed_max = tf.math.reduce_max(conv_f32)

    conv_int8_req, conv_freezed_min, conv_freezed_max = \
        load_ops_library.QuantizedConv2DWithBiasAndRequantize(input=x_int8, filter=y_int8, bias=bias_f32,
                                                              min_input=x_min, max_input=x_max, min_filter=y_min, max_filter=y_max,
                                                              min_freezed_output=z_freezed_min, max_freezed_output=z_freezed_max,
                                                              strides=[1, 1, 1, 1], padding="SAME", out_type=dtypes.qint8)

    dequantize_op = array_ops.dequantize(
        conv_int8_req, conv_freezed_min, conv_freezed_max, mode="SCALED", narrow_range=True)

    final_out = array_ops.identity(tf.shape(dequantize_op))

    with self.session(use_gpu=True) as sess:
        output_val = sess.run(final_out, feed_dict={x_f32: x_f32_np},
                              options=run_options, run_metadata=metadata)
        graph = metadata.partition_graphs[0]

    existing_pattern = False
    for node in graph.node:
        if 'Dequantize' in node.op:
            existing_pattern = True
            break
    self.assertFalse(existing_pattern)


if __name__ == "__main__":
  test.main()
