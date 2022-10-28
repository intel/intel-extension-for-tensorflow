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
"""Tests for Multiple session configuration on itex."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import os

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


def _get_config(remapper_on=False):
  """ Configure remapper optimizer on/off."""
  rewrite_config = rewriter_config_pb2.RewriterConfig()
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_config)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  # Configure remapper on/off on itex.
  os.environ['ITEX_REMAPPER'] = '0'
  if remapper_on:
    os.environ['ITEX_REMAPPER'] = '1'
  return config

@test_util.run_all_in_native_and_block_format
class OptimizerConfigTest(test.TestCase, parameterized.TestCase):
  """Tests the optimizer configuration."""

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_matmul_biasadd_gelu_fusion(self):
    """Test MatMul+BiasAdd+Gelu fusion."""
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    m, n, k = (3, 3, 4)  # Matrix dimensions
    for approximate in (False, True):
      # Create MatMul + BiasAdd + Gelu graph
      ops.reset_default_graph()
      x = _input([m, k])
      w = _weight([k, n])
      b = _bias([n])
      y = math_ops.matmul(x, w)
      z = nn.bias_add(y, b)
      out = nn.gelu(z, approximate=approximate)
      out = array_ops.identity(out)

      # Disable remapper.
      config = _get_config(remapper_on=False)
      with session.Session(config=config) as sess:
        sess.run(variables.global_variables_initializer())
        output_val_ref = sess.run(
            out, options=run_options, run_metadata=metadata)
        graph_ref = metadata.partition_graphs[0]

      # Enable remapper.
      config = _get_config(remapper_on=True)
      with session.Session(config=config) as sess:
        sess.run(variables.global_variables_initializer())
        output_val = sess.run(out, options=run_options, run_metadata=metadata)
        graph = metadata.partition_graphs[0]

      found_fused_op = False
      for node in graph_ref.node:
        if node.op in ('_OneDnnFusedMatMul'):
          fused_ops = node.attr['fused_ops'].list.s
          found_fused_op = len(fused_ops) == 2 and \
              fused_ops[0] == b'BiasAdd' and fused_ops[1] == gelu_type
          break
      self.assertFalse(found_fused_op)

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
      tol = 1e-5
      self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)


if __name__ == '__main__':
  test.main()
