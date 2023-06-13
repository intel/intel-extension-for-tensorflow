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

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.core.protobuf import config_pb2
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test as test_lib


class ContractionWithReshapeAndBiasAddGradFusionTest(test_lib.TestCase):
  """Move BiasAddGrad after reshape. So that the Contraction(MatMul
     and Conv2DBackpropFilter) can be fused with BiasAddGrad.

                  before fusion
                      root
                        |               \
                      Reshape      BiasAddGrad
            /           |
      Contraction   Contraction
                  After fusion
                      root
                        |
                      reshape
            /           |               \
      Contraction   Contraction     BiasAddGrad
  """

  @test_util.run_deprecated_v1
  def testConv1DBackwardFusion(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    input_shape = [2, 6, 2]
    filter_shape = [3, 2, 4]
    bias_shape = [4]

    val1 = np.random.uniform(size = input_shape)
    val2 = np.random.uniform(size = filter_shape)
    val3 = np.random.uniform(size = bias_shape)

    test_types = [dtypes.float32, dtypes.bfloat16]
    for dtype in test_types:
      x1_i = constant_op.constant(val1, dtype=dtype)
      x2_i = constant_op.constant(val2, dtype=dtype)
      x3_i = constant_op.constant(val3, dtype=dtype)
      filter = tf.nn.relu(x2_i)
      bias = tf.nn.relu(x3_i)
      y = tf.nn.conv1d(x1_i, filter, 1, "SAME")
      y = tf.nn.bias_add(y, bias)
      y = tf.square(y)
      grads = tf.gradients(y, [x1_i, x2_i, x3_i])

      os.environ['ITEX_REMAPPER'] = '0'
      with self.session() as sess:
        output1 = sess.run(grads)

      os.environ['ITEX_REMAPPER'] = '1'
      with self.session() as sess:
        output2 = sess.run(grads, options=run_options, run_metadata=metadata)

      graph = metadata.partition_graphs[0]
      exist_fusion_type = False
      for node in graph.node:
        if 'Conv2DBackpropFilterWithBias' in node.op:
          exist_fusion_type = True

      self.assertTrue(exist_fusion_type)
      for i, out1 in enumerate(output1):
        self.assertAllCloseAccordingToType(out1, output2[i])

  @test_util.run_deprecated_v1
  def testDenseLayerBackwardFusion(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    input_shape = [2, 3, 4]
    filter_shape = [4, 5]
    bias_shape = [5]

    val1 = np.random.uniform(size = input_shape)
    val2 = np.random.uniform(size = filter_shape)
    val3 = np.random.uniform(size = bias_shape)

    test_types = [dtypes.bfloat16]
    for dtype in test_types:
      x1_i = constant_op.constant(val1, dtype=dtype)
      x2_i = constant_op.constant(val2, dtype=dtype)
      x3_i = constant_op.constant(val3, dtype=dtype)
      filter = tf.nn.relu(x2_i)
      bias = tf.nn.relu(x3_i)
      y = tf.tensordot(x1_i, filter, [[2], [0]])
      y = tf.nn.bias_add(y, bias)
      y = tf.square(y)
      grads = tf.gradients(y, [x1_i, x2_i, x3_i])

      os.environ['ITEX_REMAPPER'] = '0'
      with self.session() as sess:
        output1 = sess.run(grads)

      os.environ['ITEX_REMAPPER'] = '1'
      with self.session() as sess:
        output2 = sess.run(grads, options=run_options, run_metadata=metadata)

      graph = metadata.partition_graphs[0]
      exist_fusion_type = False
      for node in graph.node:
        if 'FusedMatMulGrad' in node.op:
          exist_fusion_type = True

      # FusedMatMulGrad fusion is cancelled on GPU side.
      if not test_util.is_gpu_available():
        self.assertTrue(exist_fusion_type)
      for i, out1 in enumerate(output1):
        self.assertAllCloseAccordingToType(out1, output2[i])

if __name__ == "__main__":
  test_lib.main()

