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

from tensorflow import keras
from tensorflow.python.tpu import bfloat16
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import combinations

class LayerNormalizationNumericsTest(keras_parameterized.TestCase):
  """Tests LayerNormalization has correct and numerically stable outputs."""

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

  def _test_forward_pass(self, batch_input_shape, axis, fp32_tol=1e-2,
                         fp16_tol=1e-2):
    """Tests the forward pass of layer layer_normalization.

    Args:
      batch_input_shape: The input shape that will be used to test, including
        the batch dimension.
      axis: A list of axises to normalize. Will be passed to the `axis` argument
        of Layerlayer_normalization.
      fp32_tol: The relative and absolute tolerance for float32.
      fp16_tol: The relative and absolute tolerance for float16.
    """
    param_shape = [batch_input_shape[i] for i in axis]
    param_elems = 1
    for dim in param_shape:
      param_elems *= dim
    beta = np.arange(param_elems, dtype='float32').reshape(param_shape)
    gamma = np.arange(1, param_elems + 1, dtype='float32').reshape(param_shape)
    x = np.random.normal(size=batch_input_shape)
    config.set_optimizer_experimental_options({'constant_folding': False})

    for epsilon in 1e-12, 1e-3:
      expected = self._expected_layer_norm(x, beta, gamma, batch_input_shape,
                                           axis, epsilon)
      for dtype in 'float32', 'float16':
        norm = keras.layers.LayerNormalization(
            axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
            epsilon=epsilon, beta_initializer=keras.initializers.constant(beta),
            gamma_initializer=keras.initializers.constant(gamma))
        y = norm(keras.backend.cast(x, dtype))
        y = array_ops.identity(y)
        actual = keras.backend.eval(y)

        if dtype == 'float32':
          tol = fp32_tol
        else:
          assert dtype == 'float16'
          tol = fp16_tol

        # We use absolute tolerances in addition to relative tolerances, because
        # some of the values are very close to zero.
        self.assertAllClose(expected, actual, rtol=tol, atol=tol)

  @combinations.generate(combinations.combine(mode=['graph']))
  def test_forward(self):
    # For numeric stability, we ensure the axis's dimension(s) have at least 4
    # elements.
    self._test_forward_pass((3, 4), (1,))
    self._test_forward_pass((4, 5, 6), (2,))

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
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    batch_input_shape=[4, 5, 10]

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

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_custom_bf16(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    batch_input_shape=[4, 5, 10]

    param_shape = [10]
    param_elems = 10
    beta = np.array(np.random.randn(param_elems), dtype='bfloat16').reshape(param_shape)
    gamma = np.array(np.random.randn(param_elems), dtype='bfloat16').reshape(param_shape)
    x = np.array(np.random.normal(size=batch_input_shape), dtype='bfloat16')
    ln = self.custom_layer_norm(x, beta, gamma)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(ln, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    expected = self._expected_layer_norm(x, beta, gamma, batch_input_shape, (2, ), 1e-6)

    # The rtol is large, due to always exist a number maybe have 25% dif.
    self.assertAllClose(output_val, expected, 0.1)

    existing_pattern = False
    for node in graph.node:
        if 'LayerNorm' in node.op:
            existing_pattern = True
            break

    self.assertTrue(existing_pattern)

  def custom_layer_norm_with_control_dependency(self, x, beta, gamma, epsilon=1e-6):
    beta = variables.Variable(beta)
    beta = beta.assign_add(np.full(beta.shape, 0.0001))
    gamma = variables.Variable(gamma)
    gamma = gamma.assign_add(np.full(beta.shape, 0.0001))

    mean = math_ops.reduce_mean(x, axis=[-1], keepdims=True)
    variance = math_ops.reduce_mean(math_ops.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * math_ops.rsqrt(variance + epsilon)
    return array_ops.identity(norm_x * gamma + beta)


  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_custom_control_dependency(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    batch_input_shape=[4, 5, 10]

    param_shape = [10]
    param_elems = 10

    beta = np.array(np.random.randn(param_elems), dtype='float32').reshape(param_shape)
    gamma = np.array(np.random.randn(param_elems), dtype='float32').reshape(param_shape)
    x = np.array(np.random.normal(size=batch_input_shape), dtype='float32')
    ln = self.custom_layer_norm_with_control_dependency(x, beta, gamma)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(ln, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    existing_pattern = False
    for node in graph.node:
        if 'LayerNorm' in node.op:
            existing_pattern = True
            break

    self.assertTrue(existing_pattern)

if __name__ == "__main__":
  test_lib.main()
