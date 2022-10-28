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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import nn_ops
from tensorflow.core.protobuf import config_pb2
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import


class ConvBackpropFilterTest(test.TestCase):

  def _model(self):
    input_sizes = [4, 5, 5, 3]
    kernel_sizes = [3, 3, 3, 8]
    stride_sizes = [1, 2, 2, 1]
    paddings = constant_op.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

    np.random.seed(1)
    val = np.random.random_sample(input_sizes)
    input = constant_op.constant(val, dtype=dtypes.float32)
    sqrt = math_ops.sqrt(input)
    pow = math_ops.pow(sqrt, 2)
    weight_val = np.random.random_sample(kernel_sizes)
    weight = constant_op.constant(weight_val, dtype=dtypes.float32)
    pad = array_ops.pad(pow, paddings)
    conv = nn_ops.conv2d(pad, weight, stride_sizes, "VALID")

    output_sizes = conv.get_shape().as_list()
    gradient_output_val = np.random.random_sample(output_sizes)
    gradient_output = constant_op.constant(gradient_output_val, dtype=dtypes.float32)
    gradient_input = nn_ops.conv2d_backprop_filter(pad, kernel_sizes, gradient_output, stride_sizes, "VALID")

    return conv, gradient_input

  def _model_with_identity(self):
    conv, gradient_input = self._model()
    conv = array_ops.identity(conv)
    gradient_input = array_ops.identity(gradient_input)
    return conv, gradient_input


  @test_util.run_deprecated_v1
  def testAccuracy(self):
    conv1, gradient_input1 = self._model()
    conv2, gradient_input2 = self._model_with_identity()

    with self.session() as sess:
      output1 = sess.run([conv1, gradient_input1])
      output2 = sess.run([conv2, gradient_input2])

    print(output1[0])
    print(output2[0])

    self.assertAllClose(output1[0], output2[0])
    self.assertAllClose(output1[1], output2[1])


  @test_util.run_deprecated_v1
  def testGraphStructure(self):
    conv, gradient_input = self._model_with_identity()
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run([conv, gradient_input], options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]


    exist_pad_conv = False
    exist_pad_conv_backprop_filter = False
    for node in graph.node:
      if 'PadWithConv2DBackpropFilter' in node.op:
        exist_pad_conv_backprop_filter = True

      if 'PadWithConv2D' in node.op:
        exist_pad_conv = True

    self.assertTrue(exist_pad_conv)
    self.assertTrue(exist_pad_conv_backprop_filter)

if __name__ == "__main__":
  test.main()
