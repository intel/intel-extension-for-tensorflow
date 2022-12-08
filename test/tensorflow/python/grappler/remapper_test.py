# Copyright (c) 2022 Intel Corporation
#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Grappler Remapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np

from absl.testing import parameterized

from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import _pywrap_utils


def _input(shape):
  """Generates an input of a given shape."""
  return variables.Variable(random_ops.truncated_normal(shape, seed=0))


def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  return variables.Variable(lambda: init_ops.glorot_uniform_initializer(seed=0)
                            (shape))

def _bias(shape):
  """Generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)

def _conv2d(x, w):
  """Returns a 2d convolution layer with full stride."""
  return nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def _depthwise_conv2d(x, w):
  """Returns a 2d depthwise convolution layer with full stride."""
  return nn.depthwise_conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def _conv3d(x, w):
  """Returns a 3d convolution layer with full stride."""
  return nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def _get_config(remapping_on=False):
  """Returns a CongfigProto with remapper optimizer on/off."""
  if remapping_on:
    os.environ['ITEX_REMAPPER'] = '1'
  else:
    os.environ['ITEX_REMAPPER'] = '0'
  rewrite_config = rewriter_config_pb2.RewriterConfig()
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_config)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  return config

@test_util.run_all_in_native_and_block_format
class RemapperTest(test.TestCase, parameterized.TestCase):
  """Tests the Grappler remapper optimizer."""

  def _maybe_skip(self, mode):
    if mode == 'cuda':
      self.skipTest('This test does not pass on GPU.')
    if mode == 'mkl' and not test_util.IsMklEnabled():
      self.skipTest('MKL is not enabled.')

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_matmul_biasadd_gelu_fusion(self):
    """Test MatMul+BiasAdd+Gelu fusion."""
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    m, n, k = (3, 3, 4)  # Matrix dimensions
    for precision in ('float32', 'bfloat16'):
      for approximate in (False, True):
        # Gelu exact (approximate=False) is not supported with bfloat16
        # precision since no support for Erf with bfloat16 data type.
        # TODO(itex): Enable gelu exact with bfloat16, when Erf op is
        # supported with bfloat16.
        if precision == 'bfloat16':
          if not (approximate and is_bf16_supported):
            continue

        # Create MatMul + BiasAdd + Gelu graph
        ops.reset_default_graph()
        x = _input([m, k])
        w = _weight([k, n])
        b = _bias([n])
        if precision == 'bfloat16':
          x = math_ops.cast(x, dtypes.bfloat16)
          w = math_ops.cast(w, dtypes.bfloat16)
          b = math_ops.cast(b, dtypes.bfloat16)
        y = math_ops.matmul(x, w)
        z = nn.bias_add(y, b)
        out = nn.gelu(z, approximate=approximate)
        out = array_ops.identity(out)

        # Compute reference value.
        config = _get_config(remapping_on=False)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val_ref = sess.run(
              out, options=run_options, run_metadata=metadata)
        # Compute output with fusion.
        config = _get_config(remapping_on=True)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val = sess.run(out, options=run_options, run_metadata=metadata)
          graph = metadata.partition_graphs[0]

        # Graph should contain fused op.
        found_fused_op = False
        gelu_type = b'GeluApproximate' if approximate else b'GeluExact'
        for node in graph.node:
          if 'FusedMatMul' in node.op:
            fused_ops = node.attr['fused_ops'].list.s
            found_fused_op = len(fused_ops) == 2 and \
                fused_ops[0] == b'BiasAdd' and fused_ops[1] == gelu_type
            break
        self.assertTrue(found_fused_op)

        # Computed output value should be close to reference value.
        tol = 1e-5 if precision == 'float32' else 1e-2
        self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv2d_biassemantic_fusion(self):
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          self.skipTest('Device do not support bfloat16')

      if precision == 'float16':
        if not tf.config.list_physical_devices("XPU"):
          self.skipTest('CPU do not support float16')

      for bias_shape in ((6,), (1, 1, 6), (1, 1, 1, 6)):

        ops.reset_default_graph()
        x = _input((5, 8, 8, 1))
        f = _weight([3, 3, 1, 6])
        bias = constant_op.constant(
            [0.13, 0.12, -0.1, 0.23, 0.19, 0.6], shape=bias_shape)

        if precision == 'bfloat16':
          x = math_ops.cast(x, dtypes.bfloat16)
          f = math_ops.cast(f, dtypes.bfloat16)
          bias = math_ops.cast(bias, dtypes.bfloat16)

        if precision == 'float16':
          x = math_ops.cast(x, dtypes.float16)
          f = math_ops.cast(f, dtypes.float16)
          bias = math_ops.cast(bias, dtypes.float16)

        y =  _conv2d(x, f)
        z = math_ops.add(bias, y)
        out = array_ops.identity(z)

        # Compute reference value.
        config = _get_config(remapping_on=False)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val_ref = sess.run(
              out, options=run_options, run_metadata=metadata)
        # Compute output with fusion.
        config = _get_config(remapping_on=True)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val = sess.run(out, options=run_options, run_metadata=metadata)
          graph = metadata.partition_graphs[0]

          # Graph should contain fused op.
          found_fused_op = False
          for node in graph.node:
            if 'FusedConv2D' in node.op:
              found_fused_op = 1
          self.assertTrue(found_fused_op)

          # Computed output value should be close to reference value.
          tol = 1e-5 if precision == 'float32' else 1e-2
          self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_depth_conv2d_biassemantic_fusion(self):
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          self.skipTest('Device do not support bfloat16')

      if precision == 'float16':
        if not tf.config.list_physical_devices("XPU"):
          self.skipTest('CPU do not support float16')

      for bias_shape in ((6,), (1, 1, 6), (1, 1, 1, 6)):

        ops.reset_default_graph()
        x = _input((5, 8, 8, 1))
        f = _weight([3, 3, 1, 6])
        bias = constant_op.constant(
            [0.13, 0.12, -0.1, 0.23, 0.19, 0.6], shape=bias_shape)

        if precision == 'bfloat16':
          x = math_ops.cast(x, dtypes.bfloat16)
          f = math_ops.cast(f, dtypes.bfloat16)
          bias = math_ops.cast(bias, dtypes.bfloat16)

        if precision == 'float16':
          x = math_ops.cast(x, dtypes.float16)
          f = math_ops.cast(f, dtypes.float16)
          bias = math_ops.cast(bias, dtypes.float16)

        y =  _depthwise_conv2d(x, f)
        z = math_ops.add(bias, y)
        out = array_ops.identity(z)

        # Compute reference value.
        config = _get_config(remapping_on=False)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val_ref = sess.run(
              out, options=run_options, run_metadata=metadata)
        # Compute output with fusion.
        config = _get_config(remapping_on=True)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val = sess.run(out, options=run_options, run_metadata=metadata)
          graph = metadata.partition_graphs[0]

          # Graph should contain fused op.
          found_fused_op = False
          for node in graph.node:
            if 'FusedDepthwiseConv2dNative' in node.op:
              found_fused_op = 1
          self.assertTrue(found_fused_op)

          # Computed output value should be close to reference value.
          tol = 1e-5 if precision == 'float32' else 1e-2
          self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_depth_conv2d_bias_and_add_activation_fusion(self):
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          self.skipTest('Device do not support bfloat16')

      if precision == 'float16':
        if not tf.config.list_physical_devices("XPU"):
          self.skipTest('CPU do not support float16')

      ops.reset_default_graph()
      x = _input((5, 8, 8, 1))
      f = _weight([3, 3, 1, 6])
      bias = constant_op.constant([0.13, 0.12, -0.1, 0.23, 0.19, 0.6])
      add = constant_op.constant(
        2 * np.random.random_sample((5, 8, 8, 6)).astype('float32') - 1)

      if precision == 'bfloat16':
        x = math_ops.cast(x, dtypes.bfloat16)
        f = math_ops.cast(f, dtypes.bfloat16)
        bias = math_ops.cast(bias, dtypes.bfloat16)
        add = math_ops.cast(add, dtypes.bfloat16)

      if precision == 'float16':
        x = math_ops.cast(x, dtypes.float16)
        f = math_ops.cast(f, dtypes.float16)
        bias = math_ops.cast(bias, dtypes.float16)
        add = math_ops.cast(add, dtypes.float16)

      y = _depthwise_conv2d(x, f)
      z = nn.bias_add(y, bias)
      add_z = math_ops.add_n([z, add])
      out = array_ops.identity(tf.math.sigmoid(add_z))

      # Compute reference value.
      config = _get_config(remapping_on=False)
      with session.Session(config=config) as sess:
        sess.run(variables.global_variables_initializer())
        output_val_ref = sess.run(
            out, options=run_options, run_metadata=metadata)
      # Compute output with fusion.
      config = _get_config(remapping_on=True)
      with session.Session(config=config) as sess:
        sess.run(variables.global_variables_initializer())
        output_val = sess.run(out, options=run_options, run_metadata=metadata)
        graph = metadata.partition_graphs[0]

        # Graph should contain fused op.
        found_fused_op = False
        for node in graph.node:
          if 'FusedDepthwiseConv2dNative' in node.op:
            found_fused_op = True
            found_fused_node = node
            break
        self.assertTrue(
          found_fused_op, "this pattern has fusion issue!!")
        fused_ops = found_fused_node.attr['fused_ops'].list.s
        self.assertEqual(
          len(fused_ops), 3, "the number of fused ops is not equal to 3")
        existing_pattern = (
          fused_ops[0] == b'BiasAdd'
          and fused_ops[1] == b'Add'
          and fused_ops[2] == b'Sigmoid')
        self.assertTrue(existing_pattern, "invalid fused ops")

        # Computed output value should be close to reference value.
        tol = 1e-5 if precision == 'float32' else 1e-2
        self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv3d_biassemantic_fusion(self):
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          self.skipTest('Device do not support bfloat16')

      if precision == 'float16':
        if not tf.config.list_physical_devices("XPU"):
          self.skipTest('CPU do not support float16')

      for bias_shape in ((6,), (1, 1, 6), (1, 1, 1, 6)):

        ops.reset_default_graph()
        x = _input((5, 8, 8, 8, 1))
        f = _weight([3, 3, 3, 1, 6])
        bias = constant_op.constant(
            [0.13, 0.12, -0.1, 0.23, 0.19, 0.6], shape=bias_shape)

        if precision == 'bfloat16':
          x = math_ops.cast(x, dtypes.bfloat16)
          f = math_ops.cast(f, dtypes.bfloat16)
          bias = math_ops.cast(bias, dtypes.bfloat16)

        if precision == 'float16':
          x = math_ops.cast(x, dtypes.float16)
          f = math_ops.cast(f, dtypes.float16)
          bias = math_ops.cast(bias, dtypes.float16)

        y =  _conv3d(x, f)
        z = math_ops.add(bias, y)
        out = array_ops.identity(z)

        # Compute reference value.
        config = _get_config(remapping_on=False)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val_ref = sess.run(
              out, options=run_options, run_metadata=metadata)
        # Compute output with fusion.
        config = _get_config(remapping_on=True)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val = sess.run(out, options=run_options, run_metadata=metadata)
          graph = metadata.partition_graphs[0]

          # Graph should contain fused op.
          found_fused_op = False
          for node in graph.node:
            if 'FusedConv3D' in node.op:
              found_fused_op = 1
          self.assertTrue(found_fused_op)

          # Computed output value should be close to reference value.
          tol = 1e-5 if precision == 'float32' else 1e-2
          self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_matmul_biassemantic_fusion(self):
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          self.skipTest('Device do not support bfloat16')

      if precision == 'float16':
        if not tf.config.list_physical_devices("XPU"):
          self.skipTest('CPU do not support float16')

      for bias_shape in ((6,), (1, 6)):

        ops.reset_default_graph()
        x = _input((5, 8))
        f = _weight([8, 6])
        bias = constant_op.constant(
            [0.13, 0.12, -0.1, 0.23, 0.19, 0.6], shape=bias_shape)

        if precision == 'bfloat16':
          x = math_ops.cast(x, dtypes.bfloat16)
          f = math_ops.cast(f, dtypes.bfloat16)
          bias = math_ops.cast(bias, dtypes.bfloat16)

        if precision == 'float16':
          x = math_ops.cast(x, dtypes.float16)
          f = math_ops.cast(f, dtypes.float16)
          bias = math_ops.cast(bias, dtypes.float16)

        y =  math_ops.matmul(x, f)
        z = math_ops.add(bias, y)
        out = array_ops.identity(z)

        # Compute reference value.
        config = _get_config(remapping_on=False)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val_ref = sess.run(
              out, options=run_options, run_metadata=metadata)
        # Compute output with fusion.
        config = _get_config(remapping_on=True)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val = sess.run(out, options=run_options, run_metadata=metadata)
          graph = metadata.partition_graphs[0]

          # Graph should contain fused op.
          found_fused_op = False
          for node in graph.node:
            if 'FusedMatMul' in node.op:
              found_fused_op = 1
          self.assertTrue(found_fused_op)

          # Computed output value should be close to reference value.
          tol = 1e-5 if precision == 'float32' else 1e-2
          self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_mul_maximum_fusion(self):
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          continue

      if precision == 'float16':
        if not tf.config.list_physical_devices("XPU"):
          continue
      
      for input_shape in ((6,), (1, 6)):

        ops.reset_default_graph()
        input = _input(input_shape);
        input = math_ops.mul(input, input)
        constant = constant_op.constant(0.3)

        if precision == 'bfloat16':
          input = math_ops.cast(input, dtypes.bfloat16)
          constant = math_ops.cast(constant, dtypes.bfloat16)

        if precision == 'float16':
          input = math_ops.cast(input, dtypes.float16)
          constant = math_ops.cast(constant, dtypes.float16)
    
        mul_out = math_ops.mul(constant, input)
        max_out = math_ops.maximum(input, mul_out)
        out = array_ops.identity(max_out)

        # Compute reference value.
        config = _get_config(remapping_on=False)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val_ref = sess.run(
              out, options=run_options, run_metadata=metadata)
        # Compute output with fusion.
        config = _get_config(remapping_on=True)
        with session.Session(config=config) as sess:
          sess.run(variables.global_variables_initializer())
          output_val = sess.run(out, options=run_options, run_metadata=metadata)
          graph = metadata.partition_graphs[0]

        # Graph should contain fused op.
          found_fused_op = False
          for node in graph.node:
            if 'LeakyRelu' in node.op:
              found_fused_op = 1
          self.assertTrue(found_fused_op)
          # Computed output value should be close to reference value.
          self.assertAllCloseAccordingToType(output_val_ref, output_val)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_randomuniform_greaterequalwithcast_fusion(self):
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in (dtypes.float32, dtypes.bfloat16, dtypes.float16):
      if precision == dtypes.bfloat16:
        if not is_bf16_supported:
          self.skipTest('Device do not support bfloat16')

      if precision == dtypes.float16:
        if not tf.config.list_physical_devices("XPU"):
          self.skipTest('CPU do not support float16')

      randu = random_ops.random_uniform([5, 6], dtype=precision, seed=1)
      cmp_thr = constant_op.constant(0.5, dtype=precision)
      ge = math_ops.greater_equal(randu, cmp_thr)
      out = array_ops.identity(math_ops.cast(ge, dtype=precision))

      # Compute reference value.
      config = _get_config(remapping_on=False)
      with session.Session(config=config) as sess:
        sess.run(variables.global_variables_initializer())
        output_val_ref = sess.run(
            out, options=run_options, run_metadata=metadata)

      config = _get_config(remapping_on=True)
      with session.Session(config=config) as sess:
        sess.run(variables.global_variables_initializer())
        output_val = sess.run(out, options=run_options, run_metadata=metadata)

      graph = metadata.partition_graphs[0]
      # Graph should contain fused op.
      found_fused_op = False
      for node in graph.node:
        if '_ITEXFusedRandom' in node.op:
          found_fused_op = 1
      self.assertTrue(found_fused_op)
      # Computed output value should be close to reference value.
      self.assertAllCloseAccordingToType(output_val_ref, output_val)

if __name__ == '__main__':
  test.main()
