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


import numpy as np
import os

import tensorflow.compat.v1 as tf

try:
  from intel_extension_for_tensorflow.python.test_func import test as test_lib
except ImportError:
  from tensorflow.python.platform import test as test_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.core.protobuf import config_pb2


@test_util.run_all_in_graph_and_eager_modes
class DropoutTest(test_lib.TestCase):
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):

    shape1 = (16, 16, 512, 512)
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    in_array = np.ones((np.prod(shape1)))
    in_array = in_array.astype(np.float32).reshape(shape1)
    in_x = tf.placeholder(tf.float32, shape=shape1)

    for dtype in [tf.float32, tf.half, tf.bfloat16]:
      in_x_d = tf.cast(in_x, dtype=dtype)
      x = tf.nn.dropout(in_x_d, rate=0.5, seed=1)
      x = tf.nn.softmax(x)
      x = tf.identity(x)
      x = tf.gradients(x, in_x_d)

      with self.session(use_gpu=True) as sess:
        output_val = sess.run(x, options=run_options, run_metadata=metadata,
                              feed_dict={in_x: in_array})
        graph = metadata.partition_graphs[0]

      existing_pattern = False
      for node in graph.node:
        if '_ITEXFusedRandom' in node.op:
          existing_pattern = True
          break
      if test_util.is_gpu_available() or dtype != tf.half:
        self.assertTrue(existing_pattern)


if __name__ == "__main__":
  test_lib.main()
