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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import nn_ops
from tensorflow.core.protobuf import config_pb2
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import

@test_util.run_all_in_native_and_block_format
class PadWithConv3DTest(test.TestCase):

  def _model(self, fusedConv, constPaddings=True):
    input_sizes = [4, 5, 5, 3, 3]
    kernel_sizes = [3, 3, 3, 3, 3]
    stride_sizes = [1, 2, 2, 1, 1]
    if constPaddings:
      paddings = constant_op.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])
    else:
      paddings = tf.compat.v1.placeholder(tf.int32, shape=(5, 2))

    np.random.seed(1)
    val = np.random.random_sample(input_sizes)
    input = constant_op.constant(val, dtype=dtypes.float32)
    sqrt = math_ops.sqrt(input)
    pow = math_ops.pow(sqrt, 2)
    weight_val = np.random.random_sample(kernel_sizes)
    weight = constant_op.constant(weight_val, dtype=dtypes.float32)
    pad = array_ops.pad(pow, paddings)
    conv = nn_ops.conv3d(pad, weight, stride_sizes, "VALID")
    if fusedConv == True:
      conv = nn_ops.bias_add(conv, [1,2,3])
    output_sizes = [4, 3, 3, 1, 3]
    gradient_output_val = np.random.random_sample(output_sizes)
    gradient_output = constant_op.constant(gradient_output_val, dtype=dtypes.float32)
    gradient_input = nn_ops.conv3d_backprop_filter_v2(pad, kernel_sizes, gradient_output, stride_sizes, "VALID")

    return conv, gradient_input, paddings

  def _model_with_identity(self, fusedConv, constPaddings):
    conv, gradient_input, paddings = self._model(fusedConv, constPaddings)
    conv = array_ops.identity(conv)
    gradient_input = array_ops.identity(gradient_input)
    return conv, gradient_input, paddings


  @test_util.run_deprecated_v1
  def _testAccuracy(self, fusedConv, constPaddings=True):
    conv1, gradient_input1, paddings = self._model(fusedConv, constPaddings)
    conv2, gradient_input2, paddings = self._model_with_identity(fusedConv, constPaddings)

    with self.session() as sess:
      output1 = sess.run([conv1, gradient_input1])
      output2 = sess.run([conv2, gradient_input2])

    print(output1[0])
    print(output2[0])

    self.assertAllClose(output1[0], output2[0])
    self.assertAllClose(output1[1], output2[1])


  @test_util.run_deprecated_v1
  def _testGraphStructure(self, fusedConv, constPaddings=True):
    conv, gradient_input, paddings = self._model_with_identity(fusedConv, constPaddings)
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()


    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      if constPaddings:
        output_val = sess.run([conv, gradient_input], options=run_options, run_metadata=metadata)
      else:
        pad_value = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=tf.int32.as_numpy_dtype).reshape(5, 2)
        output_val = sess.run([conv, gradient_input], feed_dict={paddings: pad_value}, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]


    exist_pad_conv = False
    exist_pad_conv_backprop_filter = False
    for node in graph.node:
      if 'PadWithConv3DBackpropFilter' in node.op:
        exist_pad_conv_backprop_filter = True

      if 'PadWithConv3D' in node.op and fusedConv == False:
        exist_pad_conv = True

      if 'PadWithFusedConv3D' in node.op and fusedConv == True:
        exist_pad_conv = True
    if not constPaddings:
      self.assertTrue(exist_pad_conv)
      self.assertTrue(exist_pad_conv_backprop_filter)

  @test_util.run_deprecated_v1
  def testPadWithConv3DAccuracy(self):
    self._testAccuracy(fusedConv = False)

  def testPadWithFusedConv3DAccuracy(self):
    self._testAccuracy(fusedConv = True)

  @test_util.run_deprecated_v1
  def testPadWithConv3DGraphStructure(self):
    self._testGraphStructure(fusedConv = False, constPaddings = True)

  @test_util.run_deprecated_v1
  def testPadWithConv3DGraphStructure(self):
    self._testGraphStructure(fusedConv = False, constPaddings = False)

  @test_util.run_deprecated_v1
  def testPadWithFusedConv3DGraphStructure(self):
    self._testGraphStructure(fusedConv = True, constPaddings = True)

  @test_util.run_deprecated_v1
  def testPadWithFusedConv3DGraphStructure(self):
    self._testGraphStructure(fusedConv = True, constPaddings = False)

@test_util.run_all_in_native_and_block_format
class PadWithConv3DBackpropFilterWithBiasTest(test.TestCase):

  def _model(self):
    input_sizes = [4, 5, 5, 3, 3]
    kernel_sizes = [3, 3, 3, 3, 3]
    stride_sizes = [1, 2, 2, 1, 1]
    paddings = constant_op.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])

    np.random.seed(1)
    val = np.random.random_sample(input_sizes)
    input = constant_op.constant(val, dtype=dtypes.float32)
    weight_val = np.random.random_sample(kernel_sizes)
    weight = constant_op.constant(weight_val, dtype=dtypes.float32)
    pad = array_ops.pad(input, paddings)
    conv = nn_ops.conv3d(pad, weight, stride_sizes, "VALID")
    output_sizes = conv.get_shape().as_list()
    gradient_output_val = np.random.random_sample(output_sizes)

    pad1 = array_ops.pad(input, paddings)
    gradient_output = constant_op.constant(gradient_output_val, dtype=dtypes.float32)
    biasadd_grad = nn_ops.bias_add_grad(gradient_output)
    biasadd_grad = math_ops.pow(biasadd_grad, 2)
    gradient_input_explicit_pad = nn_ops.conv3d_backprop_filter_v2(pad1, kernel_sizes, gradient_output, stride_sizes, "VALID")
    gradient_input_explicit_pad = array_ops.identity(gradient_input_explicit_pad)

    return biasadd_grad, gradient_input_explicit_pad

  @test_util.run_deprecated_v1
  def _testAccuracy(self):
    biasadd_grad, gradient_input_explicit_pad = self._model()

    os.environ['ITEX_REMAPPER'] = '0'
    with self.session() as sess:
      output1 = sess.run([biasadd_grad, gradient_input_explicit_pad])

    os.environ['ITEX_REMAPPER'] = '1'
    with self.session() as sess:
      output2 = sess.run([biasadd_grad, gradient_input_explicit_pad])
    self.assertAllClose(output1, output2)


  @test_util.run_deprecated_v1
  def _testGraphStructure(self):
    biasadd_grad, gradient_input_explicit_pad = self._model()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run([biasadd_grad, gradient_input_explicit_pad], options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]


    exist_pad_conv_backprop_filter_bias = False
    for node in graph.node:
      if 'PadWithConv3DBackpropFilterWithBias' in node.op:
        exist_pad_conv_backprop_filter_bias = True

    self.assertTrue(exist_pad_conv_backprop_filter_bias)

  @test_util.run_deprecated_v1
  def testPadWithConv3DBackpropFilterWithBiasAccuracy(self):
    self._testAccuracy()

  @test_util.run_deprecated_v1
  def testPadWithConv3DBackpropFilterWithBiasGraphStructure(self):
    self._testGraphStructure()

if __name__ == "__main__":
  test.main()
