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
"""Tests for Grappler Remapper InstanceNorm fusion."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from absl.testing import parameterized

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

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
from tensorflow.python.util import _pywrap_utils


def _input(shape):
  """Generates an input of a given shape."""
  return variables.Variable(random_ops.truncated_normal(shape, seed=0))

def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  return variables.Variable(lambda: init_ops.glorot_uniform_initializer(seed=0)
                            (shape))

def _conv2d(x, w):
  """Returns a 2d convolution layer with full stride."""
  return nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def _conv3d(x, w):
  """Returns a 3d convolution layer with full stride."""
  return nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')

def batch_normalization(x, scale, offset, reduction_axes, epsion = 1e-3):
  """Batchnorm."""
  # because when dtype is float16, nn.moments will insert cast.
  if x.dtype == 'float16':
    mean = math_ops.reduce_mean(x, reduction_axes, keepdims=True, name="mean")
    var = math_ops.reduce_mean(
        math_ops.squared_difference(x, array_ops.stop_gradient(mean)),
        reduction_axes,
        keepdims=True,
        name="variance")
  else:
    mean, var = nn.moments(x, reduction_axes, keepdims=True)
  return nn.batch_normalization(
            x,
            mean=mean,
            variance=var,
            scale=scale,
            offset=offset,
            variance_epsilon=epsion,
        )

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
class InstanceNormTest(test.TestCase, parameterized.TestCase):
  """Tests the InstanceNorm fusion."""
  # set data format, use native format or block format.
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_instance_norm_2d_nhwc(self):
    """Test InstanceNorm NHWC format fusion."""
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          continue # Device do not support bfloat16
      if precision == 'float16':
        if not test.is_gpu_available():
          continue # CPU do not support float16

      ops.reset_default_graph()

      x = _input((5, 8, 8, 1))
      f = _weight([3, 3, 1, 6])
      in_scale = constant_op.constant([0.1, 0.2, -0.1, 0.33, 0.15, 0.66])
      in_shift = constant_op.constant([0.13, 0.12, -0.1, 0.23, 0.19, 0.6])
  
      if precision == 'bfloat16':
        x = math_ops.cast(x, dtypes.bfloat16)
        f = math_ops.cast(f, dtypes.bfloat16)
        in_scale = math_ops.cast(in_scale, dtypes.bfloat16)
        in_shift = math_ops.cast(in_shift, dtypes.bfloat16)
  
      if precision == 'float16':
        x = math_ops.cast(x, dtypes.float16)
        f = math_ops.cast(f, dtypes.float16)
        in_scale = math_ops.cast(in_scale, dtypes.float16)
        in_shift = math_ops.cast(in_shift, dtypes.float16)

      x_1 = _conv2d(x, f)
      reduction_axes = (1, 2)
  
      y = batch_normalization(x_1, in_scale, in_shift, reduction_axes)
      out = array_ops.identity(y)
  
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
          if 'InstanceNorm' in node.op:
            found_fused_op = 1
        self.assertTrue(found_fused_op)
  
        # Computed output value should be close to reference value.
        tol = 1e-5 if precision == 'float32' else 1e-2
        self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_instance_norm_3d_ndhwc(self):
    """Test InstanceNorm NDHWC format fusion."""
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        # TODO(itex) ndhwc format will cased accuray error on cpu.
        if not test.is_gpu_available():
          continue # Device do not support bfloat16
        if not is_bf16_supported:
          continue # Device do not support bfloat16

      if precision == 'float16':
        if not test.is_gpu_available():
          continue # CPU do not support float16

      ops.reset_default_graph()

      x = _input((5, 8, 8, 8, 1))
      f = _weight([3, 3, 3, 1, 6])
      in_scale = constant_op.constant([0.1, 0.2, -0.1, 0.33, 0.15, 0.66])
      in_shift = constant_op.constant([0.13, 0.12, -0.1, 0.23, 0.19, 0.6])
  
      if precision == 'bfloat16':
        x = math_ops.cast(x, dtypes.bfloat16)
        f = math_ops.cast(f, dtypes.bfloat16)
        in_scale = math_ops.cast(in_scale, dtypes.bfloat16)
        in_shift = math_ops.cast(in_shift, dtypes.bfloat16)
  
      if precision == 'float16':
        x = math_ops.cast(x, dtypes.float16)
        f = math_ops.cast(f, dtypes.float16)
        in_scale = math_ops.cast(in_scale, dtypes.float16)
        in_shift = math_ops.cast(in_shift, dtypes.float16)

      x_1 = _conv3d(x, f)
      reduction_axes = (1, 2, 3)
  
      y = batch_normalization(x_1, in_scale, in_shift, reduction_axes)
      out = array_ops.identity(y)
  
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
        if 'InstanceNorm' in node.op:
          found_fused_op = 1
      self.assertTrue(found_fused_op)

      # Computed output value should be close to reference value.
      tol = 1e-5 if precision == 'float32' else 1e-2
      self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_fused_instance_norm_2d_nhwc(self):
    """Test InstanceNorm NHWC fusion."""
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          self.skipTest('Device do not support bfloat16')

      if precision == 'float16':
        if not test.is_gpu_available():
          self.skipTest('CPU do not support float16')

      ops.reset_default_graph()

      x = _input((5, 8, 8, 1))
      f = _weight([3, 3, 1, 6])
      in_scale = constant_op.constant([0.1, 0.2, -0.1, 0.33, 0.15, 0.66])
      in_shift = constant_op.constant([0.13, 0.12, -0.1, 0.23, 0.19, 0.6])
  
      if precision == 'bfloat16':
        x = math_ops.cast(x, dtypes.bfloat16)
        f = math_ops.cast(f, dtypes.bfloat16)
        in_scale = math_ops.cast(in_scale, dtypes.bfloat16)
        in_shift = math_ops.cast(in_shift, dtypes.bfloat16)
  
      if precision == 'float16':
        x = math_ops.cast(x, dtypes.float16)
        f = math_ops.cast(f, dtypes.float16)
        in_scale = math_ops.cast(in_scale, dtypes.float16)
        in_shift = math_ops.cast(in_shift, dtypes.float16)

      x_1 = _conv2d(x, f)
      reduction_axes = (1, 2)
  
      y = batch_normalization(x_1, in_scale, in_shift, reduction_axes)

      # GPU only support Relu, CPU support both Relu and LeakyRelu.
      if test.is_gpu_available():
        y = nn.relu(y)
      else:    
        y = nn.leaky_relu(y)
      out = array_ops.identity(y)
  
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
        if 'FusedInstanceNorm' in node.op:
          fused_ops = node.attr['activation_mode'].s
          found_fused_op = 1 and \
              (fused_ops == b'LeakyRelu' or fused_ops == b'Relu')

      self.assertTrue(found_fused_op)
  
      # Computed output value should be close to reference value.
      tol = 1e-5 if precision == 'float32' else 1e-2
      self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_fused_instance_norm_3d_ndhwc(self):
    """Test InstanceNorm fusion."""
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        # TODO(itex) ndhwc format will cased accuray error on cpu.
        if not test.is_gpu_available():
          continue # Device do not support bfloat16
        if not is_bf16_supported:
          continue # Device do not support bfloat16

      if precision == 'float16':
        if not test.is_gpu_available():
          continue # CPU do not support float16

      ops.reset_default_graph()

      x = _input((5, 8, 8, 8, 1))
      f = _weight([3, 3, 3, 1, 6])
      in_scale = constant_op.constant([0.1, 0.2, -0.1, 0.33, 0.15, 0.66])
      in_shift = constant_op.constant([0.13, 0.12, -0.1, 0.23, 0.19, 0.6])
  
      if precision == 'bfloat16':
        x = math_ops.cast(x, dtypes.bfloat16)
        f = math_ops.cast(f, dtypes.bfloat16)
        in_scale = math_ops.cast(in_scale, dtypes.bfloat16)
        in_shift = math_ops.cast(in_shift, dtypes.bfloat16)
  
      if precision == 'float16':
        x = math_ops.cast(x, dtypes.float16)
        f = math_ops.cast(f, dtypes.float16)
        in_scale = math_ops.cast(in_scale, dtypes.float16)
        in_shift = math_ops.cast(in_shift, dtypes.float16)

      x_1 = _conv3d(x, f)
      reduction_axes = (1, 2, 3)
  
      y = batch_normalization(x_1, in_scale, in_shift, reduction_axes)

      # GPU only support Relu, CPU support both Relu and LeakyRelu.
      if test.is_gpu_available():
        y = nn.relu(y)
      else:    
        y = nn.leaky_relu(y)
      out = array_ops.identity(y)
  
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
        if 'FusedInstanceNorm' in node.op:
          fused_ops = node.attr['activation_mode'].s
          found_fused_op = 1 and \
              (fused_ops == b'LeakyRelu' or fused_ops == b'Relu')

      self.assertTrue(found_fused_op)
  
      # Computed output value should be close to reference value.
      tol = 1e-5 if precision == 'float32' else 1e-2
      self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_fused_instance_norm_2d_nchw(self):
    """Test InstanceNorm fusion."""
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          continue # Device do not support bfloat16

      if precision == 'float16':
        if not test.is_gpu_available():
          continue # CPU do not support float16

      ops.reset_default_graph()

      x = _input((5, 1, 8, 8))
      f = _weight([3, 3, 1, 6])
      in_scale = constant_op.constant(
          [0.1, 0.2, -0.1, 0.33, 0.15, 0.66], shape=(1, 6, 1, 1))
      in_shift = constant_op.constant(
          [0.13, 0.12, -0.1, 0.23, 0.19, 0.6], shape=(1, 6, 1, 1))
  
      if precision == 'bfloat16':
        x = math_ops.cast(x, dtypes.bfloat16)
        f = math_ops.cast(f, dtypes.bfloat16)
        in_scale = math_ops.cast(in_scale, dtypes.bfloat16)
        in_shift = math_ops.cast(in_shift, dtypes.bfloat16)
  
      if precision == 'float16':
        x = math_ops.cast(x, dtypes.float16)
        f = math_ops.cast(f, dtypes.float16)
        in_scale = math_ops.cast(in_scale, dtypes.float16)
        in_shift = math_ops.cast(in_shift, dtypes.float16)

      x_1 = nn.conv2d(x, f, strides=[1, 1, 1, 1], padding='SAME', data_format="NCHW")
      reduction_axes = (2, 3)
  
      y = batch_normalization(x_1, in_scale, in_shift, reduction_axes)

      # GPU only support Relu, CPU support both Relu and LeakyRelu.
      if test.is_gpu_available():
        y = nn.relu(y)
      else:    
        y = nn.leaky_relu(y)
      out = array_ops.identity(y)
  
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
        if 'InstanceNorm' in node.op:
          found_fused_op = 1
      self.assertTrue(found_fused_op)
  
      # Computed output value should be close to reference value.
      tol = 1e-5 if precision == 'float32' else 1e-2
      self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)


  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_fused_instance_norm_3d_ncdhw(self):
    """Test InstanceNorm fusion."""
    is_bf16_supported = _pywrap_utils.IsBF16SupportedByOneDNNOnThisCPU()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for precision in ('float32', 'bfloat16', 'float16'):
      if precision == 'bfloat16':
        if not is_bf16_supported:
          continue # Device do not support bfloat16

      if precision == 'float16':
        if not test.is_gpu_available():
          continue # CPU do not support float16

      ops.reset_default_graph()

      x = _input((5, 1, 8, 8, 8))
      f = _weight([3, 3, 3, 1, 6])
      in_scale = constant_op.constant(
          [0.1, 0.2, -0.1, 0.33, 0.15, 0.66], shape=(1, 6, 1, 1, 1))
      in_shift = constant_op.constant(
          [0.13, 0.12, -0.1, 0.23, 0.19, 0.6], shape=(1, 6, 1, 1, 1))
  
      if precision == 'bfloat16':
        x = math_ops.cast(x, dtypes.bfloat16)
        f = math_ops.cast(f, dtypes.bfloat16)
        in_scale = math_ops.cast(in_scale, dtypes.bfloat16)
        in_shift = math_ops.cast(in_shift, dtypes.bfloat16)
  
      if precision == 'float16':
        x = math_ops.cast(x, dtypes.float16)
        f = math_ops.cast(f, dtypes.float16)
        in_scale = math_ops.cast(in_scale, dtypes.float16)
        in_shift = math_ops.cast(in_shift, dtypes.float16)

      x_1 = nn.conv3d(x, f, strides=[1, 1, 1, 1, 1], padding='SAME', data_format="NCDHW")
      reduction_axes = (2, 3, 4)
  
      y = batch_normalization(x_1, in_scale, in_shift, reduction_axes)

      # GPU only support Relu, CPU support both Relu and LeakyRelu.
      if test.is_gpu_available():
        y = nn.relu(y)
      else:    
        y = nn.leaky_relu(y)
      out = array_ops.identity(y)
  
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
        if 'FusedInstanceNorm' in node.op:
          fused_ops = node.attr['activation_mode'].s
          found_fused_op = 1 and \
              (fused_ops == b'LeakyRelu' or fused_ops == b'Relu')

      self.assertTrue(found_fused_op)
  
      # Computed output value should be close to reference value.
      tol = 1e-5 if precision == 'float32' else 1e-2
      self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)


if __name__ == '__main__':
  test.main()
