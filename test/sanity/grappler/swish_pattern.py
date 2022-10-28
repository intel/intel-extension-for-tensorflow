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

from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes

@test_util.run_all_in_native_and_block_format
@test_util.run_all_in_graph_and_eager_modes
class SwishTest(test_lib.TestCase):
  def testValues(self):
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    z = (np_values * 0.).astype(np.float32)
    input1 = math_ops.add_n([np_values, z])
    swish1 = input1 * math_ops.sigmoid(input1)
    swish1 = array_ops.identity(swish1)

    input2 = constant_op.constant(np_values)
    swish2 = input2 * math_ops.sigmoid(input2)

    output1 = self.evaluate(swish1)
    output2 = self.evaluate(swish2)
    self.assertAllClose(output1, output2)

  def _CreateNumpyTensor(self, shape):
    total_size = 1
    for s in shape:
      total_size *= s
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  def testOneDnnSwish(self):
    input_size = [2, 5, 5, 2]
    filter_size = [3, 3, 2, 4]
    x1 = self._CreateNumpyTensor(input_size)
    x2 = self._CreateNumpyTensor(filter_size)

    conv1 = nn_ops.conv2d(x1, x2, strides=[1, 1], padding='VALID')
    z = math_ops.cast(conv1 * 0, conv1.dtype)
    conv1 = math_ops.add_n([conv1, z])
    swish1 = conv1 * math_ops.sigmoid(conv1)
    swish1 = array_ops.identity(swish1)

    conv2 = nn_ops.conv2d(x1, x2, strides=[1, 1], padding='VALID')
    z2 = math_ops.cast(conv2 * 0, conv2.dtype)
    conv2 = math_ops.add_n([conv2, z2])
    swish2 = conv2 * math_ops.sigmoid(conv2)

    output1 = self.evaluate(swish1)
    output2 = self.evaluate(swish2)
    self.assertAllClose(output1, output2)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):
    def swish(features):
      return features * math_ops.sigmoid(features)

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    np_values = np.array(
        [np.linspace(-7.0, 0.0, 5),
         np.linspace(0.0, 7.0, 5)],
        dtype=np.float32)
    z = (np_values * 0.).astype(np.float32)
    input1 = math_ops.add_n([np_values, z])
    swish1 = swish(input1)
    swish1 = array_ops.identity(swish1)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(swish1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    exsiting_swish = False
    for node in graph.node:
        if 'Swish' in node.op:
          exsiting_swish = True
          break

    self.assertTrue(exsiting_swish)

@test_util.run_all_in_native_and_block_format
@test_util.run_all_in_graph_and_eager_modes
class SwishAlphaTest(test_lib.TestCase):
  def testValues(self):
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    z = (np_values * 0.).astype(np.float32)
    input1 = math_ops.add_n([np_values, z])
    alpha1 = constant_op.constant(5., dtypes.float32)
    swish1 = input1 * math_ops.sigmoid(alpha1 * input1)
    swish1 = array_ops.identity(swish1)

    input2 = constant_op.constant(np_values)
    alpha2 = constant_op.constant(5., dtypes.float32)
    swish2 = input2 * math_ops.sigmoid(alpha2 * input2)

    output1 = self.evaluate(swish1)
    output2 = self.evaluate(swish2)
    self.assertAllClose(output1, output2)

  def _CreateNumpyTensor(self, shape):
    total_size = 1
    for s in shape:
      total_size *= s
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  def testOneDnnSwish(self):
    input_size = [2, 5, 5, 2]
    filter_size = [3, 3, 2, 4]
    x1 = self._CreateNumpyTensor(input_size)
    x2 = self._CreateNumpyTensor(filter_size)

    conv1 = nn_ops.conv2d(x1, x2, strides=[1, 1], padding='VALID')
    z = math_ops.cast(conv1 * 0, conv1.dtype)
    conv1 = math_ops.add_n([conv1, z])
    alpha1 = constant_op.constant(5., dtypes.float32)
    swish1 = conv1 * math_ops.sigmoid(alpha1 * conv1)
    swish1 = array_ops.identity(swish1)

    conv2 = nn_ops.conv2d(x1, x2, strides=[1, 1], padding='VALID')
    z2 = math_ops.cast(conv2 * 0, conv2.dtype)
    conv2 = math_ops.add_n([conv2, z2])
    alpha2 = constant_op.constant(5., dtypes.float32)
    swish2 = conv2 * math_ops.sigmoid(alpha2 * conv2)

    output1 = self.evaluate(swish1)
    output2 = self.evaluate(swish2)
    self.assertAllClose(output1, output2)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    input1 = variables.Variable(random_ops.truncated_normal((6, 1), seed=0))
    input1 = input1 * input1
    alpha1 = constant_op.constant(5., dtypes.float32)
    swish1 = input1 * math_ops.sigmoid(alpha1 * input1)
    swish1 = array_ops.identity(swish1)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(swish1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    exsiting_swish = False
    for node in graph.node:
        if 'Swish' in node.op:
          exsiting_swish = True
          break

    self.assertTrue(exsiting_swish)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testNNImplSwishApi(self):

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    input1 = variables.Variable(random_ops.truncated_normal((6, 1), seed=0))
    input1 = input1 * input1
    swish1 = nn_impl.swish(input1)
    swish1 = array_ops.identity(swish1)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(swish1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    exsiting_swish = False
    for node in graph.node:
        if 'Swish' in node.op:
          exsiting_swish = True
          break

    self.assertTrue(exsiting_swish)

if __name__ == "__main__":
  test_lib.main()
