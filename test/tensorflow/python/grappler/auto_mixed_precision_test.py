# Copyright (c) 2022 Intel Corporation
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Advanced AutoMixedPrecision."""

import os

from absl.testing import parameterized
import tensorflow as tf

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent

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


def _conv3d(x, w):
  """Returns a 3d convolution layer with full stride."""
  return nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
  """Downsamples a feature map by 2X."""
  return nn.max_pool(
      x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def _fused_batchnorm(x, scale, offset):
  """Batchnorm."""
  return nn_impl.fused_batch_norm(
      x, scale=scale, offset=offset, is_training=True)


def _conv_bn(x):
  """Conv followed by batchnorm."""
  i = array_ops.reshape(x, [-1, 8, 8, 1])
  f = _weight([3, 3, 1, 6])
  x = _conv2d(i, f)
  s = _weight([6])
  o = _weight([6])
  y, _, _ = _fused_batchnorm(x, s, o)
  y = array_ops.identity(y)
  return y


def _conv3d_bn(x):
  """Conv3D followed by batchnorm."""
  i = array_ops.reshape(x, [-1, 8, 8, 8, 1])
  f = _weight([3, 3, 3, 1, 6])
  x = _conv3d(i, f)
  s = _weight([6])
  o = _weight([6])
  x = array_ops.reshape(x, [-1, 8, 8, 6])
  y, _, _ = _fused_batchnorm(x, s, o)
  y = array_ops.identity(y)
  return y


def _matmul_act(x):
  """Matmul followed by activation."""
  i = array_ops.reshape(x, [8, 8])
  f = _weight([8, 8])
  x = math_ops.matmul(i, f)
  y = nn.relu(x)
  return y


def _conv_pool(x):
  """(Conv -> bias -> relu -> max_pool) x2."""
  x_image = array_ops.reshape(x, [-1, 8, 8, 1])
  w_conv1 = _weight([3, 3, 1, 6])
  b_conv1 = _bias([6])
  h_conv1 = nn.relu(nn.bias_add(_conv2d(x_image, w_conv1), b_conv1))
  h_pool1 = _max_pool_2x2(h_conv1)
  w_conv2 = _weight([3, 3, 6, 4])
  b_conv2 = _bias([4])
  h_conv2 = nn.relu(nn.bias_add(_conv2d(h_pool1, w_conv2), b_conv2))
  h_pool2 = _max_pool_2x2(h_conv2)
  return h_pool2


def _depthwise_conv2d(x, w):
  """Returns a 2d depthwise convolution layer with full stride."""
  return nn.depthwise_conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _simple_loop(x, functor):
  """Simple loop whose body is provided by the functor."""
  init = (constant_op.constant(0), x)
  c = lambda i, j: i < 4
  b = lambda i, j: (i + 1, functor(j))
  ij = control_flow_ops.while_loop(c, b, init)
  return ij


def _loop_vars_intertwined(x0, y0, functor_x, functor_y):
  """Loop whose loop variables are intertwined."""
  c = lambda i, j, x, y: j < 4
  b = lambda i, j, x, y: (j + 1, i + 1, functor_y(y), functor_x(x))
  init = (constant_op.constant(0), constant_op.constant(0), x0, y0)
  ijzw = control_flow_ops.while_loop(c, b, init)
  return ijzw


def _lstm_cell(prev_c, prev_h, x):
  """Create an LSTM cell."""
  # i: input gate
  # f: forget gate
  # o: output gate
  # c: cell state
  # x: input
  # h: embedding
  bias = _bias([4])
  w = _weight([8, 16])
  ifoc = math_ops.matmul(array_ops.concat([x, prev_h], axis=1), w)
  i, f, o, c = array_ops.split(ifoc, 4, axis=1)
  i = math_ops.sigmoid(nn.bias_add(i, bias))
  f = math_ops.sigmoid(nn.bias_add(f, bias))
  o = math_ops.sigmoid(nn.bias_add(o, bias))
  c = math_ops.tanh(nn.bias_add(c, bias))
  next_c = f * prev_c + i * c
  next_h = o * math_ops.tanh(next_c)
  return next_c, next_h


def _recurrent_lstm(c, h):
  """Dynamic single-layer LSTM with TensorArray."""

  def cond(i, c, h, ta_x):
    del c, h, ta_x
    return i < 4

  def body(i, c, h, ta_x):
    x = ta_x.read(i)
    next_c, next_h = _lstm_cell(c, h, x)
    return (i + 1, next_c, next_h, ta_x)

  ta_x = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=4)
  for i in range(0, 4):
    ta_x = ta_x.write(
        i, constant_op.constant(0.1, shape=[8, 4], dtype=dtypes.float32))
  init = (constant_op.constant(0), c, h, ta_x)
  r = control_flow_ops.while_loop(cond, body, init)
  return r


def _make_node_with_color(color, input_tensor, name=None):
  """Returns a node representative of the specified list type."""
  color = color.lower()
  if color == 'w':  # Allow node
    weights = _weight(input_tensor.get_shape().as_list())
    return math_ops.matmul(input_tensor, weights, name=name)
  if color == 'g':  # Infer node
    return math_ops.add(input_tensor, 0.1, name=name)
  if color == 'c':  # Clear node
    return nn.relu(input_tensor, name=name)
  if color == 'b':  # Deny node
    return math_ops.pow(math_ops.pow(input_tensor, 2.), 0.5, name=name)
  raise ValueError('Invalid node color: ' + str(color))


def _build_simple_loop_graph(inp_colors, body_colors, out_colors):
  """Builds a test graph with a simple loop."""
  a = _input([8, 8])
  for i, color in enumerate(inp_colors):
    a = _make_node_with_color(color, a, 'input_%i' % i)

  def body(x):
    for i, color in enumerate(body_colors):
      x = _make_node_with_color(color, x, 'body_%i' % i)
    return x

  _, a = _simple_loop(a, body)
  for i, color in enumerate(out_colors):
    a = _make_node_with_color(color, a, 'output_%i' % i)
  a = array_ops.identity(a)
  return a


def _get_config(auto_mixed_precision_mode):
  """Returns a ConfigProto with auto mixed precision enabled if appropriate."""
  rewrite_config = rewriter_config_pb2.RewriterConfig(
      # do not remove duplicated nodes
      arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
  )
  if auto_mixed_precision_mode:
    # enable auto mixed precision.
    os.environ['ITEX_AUTO_MIXED_PRECISION'] = '1'
    if auto_mixed_precision_mode == "bfloat16":
      os.environ['ITEX_AUTO_MIXED_PRECISION_DATA_TYPE'] = 'BFLOAT16'
    elif auto_mixed_precision_mode == "float16":
      os.environ['ITEX_AUTO_MIXED_PRECISION_DATA_TYPE'] = 'FLOAT16'

    # do not turn Conv2D and other nodes into _FusedConv2D
    # enable auto mixed precision.
    os.environ['ITEX_AUTO_MIXED_PRECISION'] = '1'
    # disable ramapper. do not turn Conv2D and other nodes into _FusedConv2D
    os.environ['ITEX_REMAPPER'] = '0'
  else:
    assert auto_mixed_precision_mode is None
    os.environ['ITEX_AUTO_MIXED_PRECISION'] = '0'
    os.environ['ITEX_REMAPPER'] = '0'
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(
      rewrite_options=rewrite_config, build_cost_model=1)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  config.graph_options.optimizer_options.opt_level = -1
  return config

def _get_device():
  """Returns the device to run on. If mode is cpu, run on CPU"""
  if test.is_gpu_available():
    return ''
  return '/cpu:0'

def _is_cast_to_fp16(node_name):
  return node_name.endswith('-CastToFp16-AutoMixedPrecision')


def _is_cast_to_bf16(node_name):
  return node_name.endswith('-CastToBf16-AutoMixedPrecision')


def _is_cast_to_fp32(node_name):
  return node_name.endswith('-CastToFp32-AutoMixedPrecision')


def _count_casts(mode, nodes):
  """Counts the number of casts to f16 and fp32."""
  num_to_fp16 = 0
  num_to_bf16 = 0
  num_to_fp32 = 0
  for node in nodes:
    if _is_cast_to_fp16(node.name):
      num_to_fp16 += 1
    if _is_cast_to_bf16(node.name):
      num_to_bf16 += 1
    elif _is_cast_to_fp32(node.name):
      num_to_fp32 += 1
  if mode == 'float16': # pylint: disable=no-else-return
    assert num_to_bf16 == 0
    return num_to_fp16, num_to_fp32
  else:
    assert mode == 'bfloat16'
    assert num_to_fp16 == 0
    return num_to_bf16, num_to_fp32


def _build_node_map(nodes):
  node_map = {}
  for node in nodes:
    node_map[node.name] = node
  return node_map


def _example_noninlined_funcdef_shape(op):
  return [op.inputs[0].shape]


@function.Defun(
    shape_func=_example_noninlined_funcdef_shape,
    func_name='example_noninlined_funcdef_grad',
    noinline=True)
def _example_noninlined_funcdef_grad(features, grad):
  """Gradient of Swish function defined below."""
  sigmoid_features = math_ops.sigmoid(features)
  activation_grad = (
      sigmoid_features * (1.0 + features * (1.0 - sigmoid_features)))
  return grad * activation_grad


@function.Defun(
    grad_func=_example_noninlined_funcdef_grad,
    shape_func=_example_noninlined_funcdef_shape,
    func_name='example_noninlined_funcdef',
    noinline=True)
def _example_noninlined_funcdef(features):
  """Computes the Swish activation function: `x * sigmoid(x)`."""
  return features * math_ops.sigmoid(features)


class AutoMixedPrecisionTest(test.TestCase, parameterized.TestCase):
  """Tests the Grappler auto mixed precision optimizer."""
  IGNORE_PERF_VAR = 'TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'

  # TODO(benbarsdell): Add tests for eager mode with a tf.function.

  def setUp(self):
    super(AutoMixedPrecisionTest, self).setUp()
    # Enable the CUDA tests to be run on pre-Volta GPUs by telling the grappler
    # pass to ignore performance and always transform the graph.
    self._original_ignore_perf_value = os.getenv(self.IGNORE_PERF_VAR)
    os.environ[self.IGNORE_PERF_VAR] = '1'

  def tearDown(self):
    if self._original_ignore_perf_value is not None:
      os.environ[self.IGNORE_PERF_VAR] = self._original_ignore_perf_value
    else:
      del os.environ[self.IGNORE_PERF_VAR]
    super(AutoMixedPrecisionTest, self).tearDown()

  def _lower_precision_dtype(self, mode):
    return dtypes.float16 if mode == 'float16' else dtypes.bfloat16

  def _assert_output_f16(self, mode, node_map, node_name, output_port=0):
    self.assertEqual(node_map[node_name].output_info[output_port].dtype,
                     self._lower_precision_dtype(mode).as_datatype_enum)

  def _run(self, mode, fetches):
    """Runs the graph and returns the evaluation of the fetches."""
    with session.Session(config=_get_config(None)) as sess:
      sess.run(variables.global_variables_initializer())
      output_val_ref = self.evaluate(fetches)

    with session.Session(config=_get_config(mode)) as sess:
      sess.run(variables.global_variables_initializer())
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(fetches, run_metadata=metadata)

    return output_val_ref, output_val, metadata.cost_graph

  def _maybe_skip(self, mode):
    if mode == 'float16' and not test.is_gpu_available():
      self.skipTest('CPU do not support float16')

  def _run_simple_loop_test(self, mode, inp, body, out):
    """Runs a test of a simple loop.

    The loop has different node colors in different sections of the graph. The
    arguments must be strings where each character represents the color of a
    node in that section of the graph: w = allow, g = infer, c = clear,
    b = deny. CAPITALIZED characters indicate that the node is expected to be
    changed to DT_HALF during graph optimization.

    inp -> loop [ body ] -> out.

    Args:
      mode: Either 'float16' or 'bfloat16'.
      inp: A string of letters indicating the colors and expected dtypes of the
        input nodes.
      body: A string of letters indicating the colors and expected dtypes of the
        body nodes.
      out: A string of letters indicating the colors and expected dtypes of the
        output nodes.
    """
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      expected_types = []
      for section in [inp, body, out]:
        section_expected_types = []
        for color in section:
          if color.isupper():
            expected_type = self._lower_precision_dtype(mode).as_datatype_enum
          else:
            expected_type = types_pb2.DT_FLOAT
          section_expected_types.append(expected_type)
        expected_types.append(section_expected_types)
      a = _build_simple_loop_graph(inp, body, out)
    output_val_ref, output_val, cost_graph = self._run(mode, a)
    node_map = _build_node_map(cost_graph.node)

    section_names = ['input', 'while/body', 'output']
    all_types_correct = True
    for section_name, expected_types in zip(section_names, expected_types):
      for i, expected_type in enumerate(expected_types):
        node_name = section_name + '_%i' % i
        output_port = 0
        optimized_type = node_map[node_name].output_info[output_port].dtype
        if optimized_type != expected_type:
          print('Expected node %s to have type %s but got type %s' %
                (node_name, expected_type, optimized_type))
          all_types_correct = False
    self.assertTrue(all_types_correct)
    if mode == 'bfloat16':
      self.assertAllClose(output_val_ref, output_val, atol=2e-2, rtol=2e-2)
    else:
      self.assertAllClose(output_val_ref, output_val, atol=2e-3, rtol=1e-3)

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_bn(self, mode):
    """Test graph with convolution followed by batch norm."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      x = _conv_bn(x)
      output = _conv_bn(x)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    num_to_f16, num_to_fp32 = _count_casts(mode, cost_graph.node)

    self._assert_output_f16(mode, node_map, 'Conv2D')
    self._assert_output_f16(mode, node_map, 'FusedBatchNormV3')
    self._assert_output_f16(mode, node_map, 'Conv2D_1')
    self.assertEqual(num_to_f16, 3)  # Before Conv2D:0, Conv2D:1, Conv2D_1:1
    self.assertEqual(num_to_fp32, 1)  # After FusedBatchNormV3:0
    if mode == 'bfloat16':
      tol = 1e-2
    else:
      tol = 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv3d_bn(self, mode):
    """Test graph with convolution followed by batch norm."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 8, 1])
      x = _conv3d_bn(x)
      output = _conv3d_bn(x)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    num_to_fp16, num_to_fp32 = _count_casts(mode, cost_graph.node)

    self._assert_output_f16(mode, node_map, 'Conv3D')
    self._assert_output_f16(mode, node_map, 'FusedBatchNormV3')
    self._assert_output_f16(mode, node_map, 'Conv3D_1')
    self.assertEqual(num_to_fp16, 3)  # Before Conv3D:0, Conv3D:1, Conv3D_1:1
    self.assertEqual(num_to_fp32, 1)  # After FusedBatchNormV3:0
    self.assertAllClose(output_val_ref, output_val, atol=1e-2, rtol=1e-2)

  @parameterized.parameters(['bfloat16'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv3d(self, mode):
    """Test grad ops with convolution3d graph."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 8, 1])
      f = _weight([3, 3, 3, 1, 6])
      y = _conv3d(x, f)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x, f])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    self._assert_output_f16(mode, node_map, 'Conv3D')
    self._assert_output_f16(mode, node_map,
                            'gradients/Conv3D_grad/Conv3DBackpropInputV2')
    self._assert_output_f16(mode, node_map,
                            'gradients/Conv3D_grad/Conv3DBackpropFilterV2')

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    tol = 5e-2 if mode == 'bfloat16' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['bfloat16'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_bn_dropout(self, mode):
    """Test dropout precision of convolution batch norm graph."""
    self._maybe_skip(mode)
    if test.is_gpu_available():
      self.skipTest('GPU will cause error!')
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      y = _conv_bn(x)
      y = nn.dropout(y, rate=0.5)
      y = math_ops.add(y, 1, name='addition')
      y = _conv_bn(y)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    self._assert_output_f16(mode, node_map, 'Conv2D')
    self._assert_output_f16(mode, node_map, 'FusedBatchNormV3')
    # We do not assert dropout's dtype because we do not want to rely on the
    # node names of dropout's internal implementation.
    self._assert_output_f16(mode, node_map, 'addition')
    self._assert_output_f16(mode, node_map, 'Conv2D_1')

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    tol = 5e-2 if mode == 'bfloat16' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_pool(self, mode):
    """Test graph with convolution followed by pooling."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      output = _conv_pool(x)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    num_to_f16, num_to_fp32 = _count_casts(mode, cost_graph.node)

    self._assert_output_f16(mode, node_map, 'Conv2D')
    self._assert_output_f16(mode, node_map, 'Relu')
    self._assert_output_f16(mode, node_map, 'MaxPool')
    self._assert_output_f16(mode, node_map, 'Conv2D_1')
    self.assertEqual(num_to_f16, 5)
    self.assertEqual(num_to_fp32, 1)
    tol = 5e-3 if mode == 'bfloat16' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['bfloat16'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_simple_loop(self, mode):
    """Test graph with while loop."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y = _simple_loop(x, _matmul_act)[1]
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'while/MatMul')
    self._assert_output_f16(mode, node_map, 'while/Relu')
    tol = 1e-2 if mode == 'bfloat16' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['bfloat16'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_loop_with_vars_intertwined(self, mode):
    """Test graph with intertwined while loops."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      _, _, k, l = _loop_vars_intertwined(
          array_ops.ones(array_ops.shape(x)), x, _matmul_act, _matmul_act)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(k, [x])
      output = (k, l, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'while/MatMul')
    self._assert_output_f16(mode, node_map, 'while/Relu')
    self._assert_output_f16(mode, node_map, 'while/MatMul_1')
    self._assert_output_f16(mode, node_map, 'while/Relu_1')
    tol = 5e-3 if mode == 'bfloat16' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['bfloat16'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_multi_paths(self, mode):
    """Test graph with multiple paths."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y1 = _matmul_act(x)
      y2 = _matmul_act(x)
      y = y1 + y2 + x
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (g, y)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'MatMul')
    self._assert_output_f16(mode, node_map, 'Relu')
    self._assert_output_f16(mode, node_map, 'MatMul_1')
    self._assert_output_f16(mode, node_map, 'Relu_1')
    if mode == 'bfloat16':
      tol = 2e-2
    else:
      tol = 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['bfloat16'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_recurrent_lstm(self, mode):
    """Test graph with recurrent lstm."""
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      init_c = _input([8, 4])
      init_h = _input([8, 4])
      _, _, h, _ = _recurrent_lstm(init_c, init_h)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(h, [init_c, init_h])
      output = (h, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'while/concat')
    self._assert_output_f16(mode, node_map, 'while/MatMul')
    self._assert_output_f16(mode, node_map, 'while/split')
    self._assert_output_f16(mode, node_map, 'while/Sigmoid')
    self._assert_output_f16(mode, node_map, 'while/Sigmoid_1')
    self._assert_output_f16(mode, node_map, 'while/Sigmoid_2')
    self._assert_output_f16(mode, node_map, 'while/Tanh')
    self._assert_output_f16(mode, node_map, 'while/Tanh_1')
    self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_1(self, mode):
    self._run_simple_loop_test(mode, 'W', 'C', 'C')

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_2(self, mode):
    self._run_simple_loop_test(mode, 'C', 'C', 'W')

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_3(self, mode):
    self._run_simple_loop_test(mode, 'W', 'G', 'W')

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_4(self, mode):
    self._run_simple_loop_test(mode, 'W', 'gbg', 'W')

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_5(self, mode):
    self._run_simple_loop_test(mode, 'b', 'gWC', 'c')

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_6(self, mode):
    self._run_simple_loop_test(mode, 'b', 'CWCG', 'C')

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_7(self, mode):
    self._run_simple_loop_test(mode, 'C', 'GWCG', 'C')

  @parameterized.parameters(['float16', 'bfloat16'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_8(self, mode):
    self._run_simple_loop_test(mode, 'C', 'CgbgWC', 'g')

  @parameterized.parameters(['bfloat16'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_noninlined_funcdef(self, mode):
    """Test graph with non-inlined function subgraph.

    This requires the grappler pass to handle an OpDef that only appears in the
    graph's function registry instead of the global op registry.

    Args:
      mode: Either 'float16' or 'bfloat16'.
    """
    self._maybe_skip(mode)
    with ops.device(_get_device()):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y = _matmul_act(x)
      y = _example_noninlined_funcdef(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (g, y)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'MatMul')
    tol = 1e-2 if mode == 'bfloat16' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)


if __name__ == '__main__':
  test.main()
