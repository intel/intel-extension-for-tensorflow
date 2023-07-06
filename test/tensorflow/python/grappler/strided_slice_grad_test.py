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

from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.core.protobuf import config_pb2
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test as test_lib


class StridedSliceGradTest(test_lib.TestCase):
  """Remap StridedSliceGrad to Pad when the strides of it is 1.
  """

  @test_util.run_deprecated_v1
  def testStridedSliceGrad(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    input_shape_1 = [3, 4]
    input_shape_2 = [3, 4, 5, 6]
    input_shape_3 = [3, 4, 5, 6, 7]

    val1 = np.random.uniform(size = input_shape_1)
    val2 = np.random.uniform(size = input_shape_2)
    val3 = np.random.uniform(size = input_shape_3)

    test_types = [dtypes.float32, dtypes.bfloat16]
    for dtype in test_types:
      x1_i = constant_op.constant(val1, dtype=dtype)
      x2_i = constant_op.constant(val2, dtype=dtype)
      x3_i = constant_op.constant(val3, dtype=dtype)
      x1 = nn_ops.relu(x1_i)
      x2 = nn_ops.relu(x2_i)
      x3 = nn_ops.relu(x3_i)

      input_list = [
          (x1_i, x1[1, 1:3]), 
          (x1_i, x1[:, 1:4:1]),
          (x2_i, x2[0, :2, :, :]), 
          (x2_i, x2[:, 1:3, :3:1, :]),
          (x2_i, x2[1, :, 1:, 2:5]),
          (x3_i, x3[0, 2:, :, :5:1, 3:7]), 
          (x3_i, x3[1:, 1:, 4, :, :5]),
          (x3_i, x3[:, :, :, :, 2]),
      ]
      for x, y in input_list:
        y = tf.square(y)
        grads = tf.gradients(y, x)
        os.environ['ITEX_REMAPPER'] = '0'
        with self.session() as sess:
          output1 = sess.run(grads)

        os.environ['ITEX_REMAPPER'] = '1'
        with self.session() as sess:
          output2 = sess.run(grads, options=run_options, run_metadata=metadata)

        graph = metadata.partition_graphs[0]
        exist_pad = False
        for node in graph.node:
          if 'Pad' in node.op:
            exist_pad = True

        self.assertTrue(exist_pad)
        self.assertAllEqual(output1, output2)


if __name__ == "__main__":
  test_lib.main()

