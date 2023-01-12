# Copyright (c) 2022 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for miscellaneous functionality in tensorflow.ops.nn."""
import os
import numpy as np
import tensorflow as tf

from intel_extension_for_tensorflow.python.test_func import test as test_lib
from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session


@test_util.run_all_in_native_and_block_format
@test_util.run_all_in_graph_and_eager_modes
class LayerNormAliTest(test_lib.TestCase):
  @test_util.run_deprecated_v1
  def layer_norm_pattern(self, x, offset, scale, reduction_axes,
    pattern_num=1, epsilon=1e-3):
    mean = math_ops.reduce_mean(x, reduction_axes, keepdims=True, name="mean")
    var = math_ops.reduce_mean(
        math_ops.squared_difference(x, array_ops.stop_gradient(mean)),
        reduction_axes, keepdims=True, name="variance")
    inv = math_ops.rsqrt(var + epsilon) * scale
    if pattern_num==1:
      y = x * inv + (offset - mean * inv)
    else:
      y = inv * x + (offset - mean * inv)
    return y

  @test_util.run_deprecated_v1
  def test_layer_norm_ali(self):
    if test_lib.is_gpu_available():
      self.skipTest("Skip on GPU")
    tf.compat.v1.disable_eager_execution()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    # input
    x = constant_op.constant(np.random.rand(6, 6, 6, 6),
      dtype=dtypes.float32)
    offset = constant_op.constant([0.1, 0.2, -0.1, 0.33, 0.15, 0.66])
    scale = constant_op.constant([0.13, 0.12, -0.1, 0.23, 0.19, 0.6])
    reduction_axes = (-1)
    y = self.layer_norm_pattern(x, offset, scale, reduction_axes,
      pattern_num=1)
    out = array_ops.identity(y)

    y_2 = self.layer_norm_pattern(x, offset, scale, reduction_axes,
      pattern_num=2)
    out_2 = array_ops.identity(y)


    # Compute reference value.
    os.environ['ITEX_REMAPPER'] = '0'
    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val_ref = sess.run(
          out, options=run_options, run_metadata=metadata)

    # Compute output with fusion.
    os.environ['ITEX_REMAPPER'] = '1'
    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(out, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

      # Graph should contain fused op.
      found_fused_op = False
      for node in graph.node:
        if 'LayerNorm' in node.op:
          found_fused_op = 1

      self.assertTrue(found_fused_op)
      self.assertAllClose(output_val_ref, output_val, atol=1e-5, rtol=1e-5)


    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(out_2, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

      # Graph should contain fused op.
      found_fused_op = False
      for node in graph.node:
        if 'LayerNorm' in node.op:
          found_fused_op = 1

      self.assertTrue(found_fused_op)
      # Computed output value should be close to reference value.
      self.assertAllClose(output_val_ref, output_val, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  test_lib.main()

