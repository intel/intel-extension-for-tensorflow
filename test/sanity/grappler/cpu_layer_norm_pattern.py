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

import numpy as np

from intel_extension_for_tensorflow.python.test_func import test as test_lib
from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2

@test_util.run_all_in_native_and_block_format
@test_util.run_all_in_graph_and_eager_modes
class CustomLayerNormTests(test_lib.TestCase):
  def custom_layer_norm(self, x, beta, gamma, epsilon=1e-6):
    mean = math_ops.reduce_mean(x, axis=[-1], keepdims=True)
    variance = math_ops.reduce_mean(math_ops.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * math_ops.rsqrt(variance + epsilon)
    return array_ops.identity(norm_x * gamma + beta)

  def _expected_layer_norm(self, x, beta, gamma, batch_input_shape, axis,
                           epsilon):
    """Returns the layer norm, which is computed using NumPy."""
    broadcast_shape = [batch_input_shape[i] if i in axis else 1
                       for i in range(len(batch_input_shape))]
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    expected = (x - mean) / np.sqrt(var + epsilon)
    expected *= np.reshape(gamma, broadcast_shape)
    expected += np.reshape(beta, broadcast_shape)
    return expected

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_custom(self):
    if test_lib.is_gpu_available():
      self.skipTest("Skip on GPU")
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    batch_input_shape=[2, 3, 10]

    param_shape = [10]
    param_elems = 10
    beta = np.arange(param_elems, dtype='float32').reshape(param_shape)
    gamma = np.arange(1, param_elems + 1, dtype='float32').reshape(param_shape)
    x = np.array(np.random.normal(size=batch_input_shape), dtype='float32')
    ln = self.custom_layer_norm(x, beta, gamma)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(ln, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    expected = self._expected_layer_norm(x, beta, gamma, batch_input_shape, (2, ), 1e-6)

    self.assertAllClose(output_val, expected, 1e-2)

    existing_pattern = False
    for node in graph.node:
        if 'LayerNorm' in node.op:
            existing_pattern = True
            break

    self.assertTrue(existing_pattern)

if __name__ == "__main__":
  test_lib.main()
