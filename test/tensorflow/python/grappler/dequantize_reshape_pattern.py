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
import tensorflow as tf
try:
    from intel_extension_for_tensorflow.python.test_func import test as test_lib
except ImportError:
    from tensorflow.python.platform import test as test_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2

@test_util.run_all_in_graph_and_eager_modes
class DequantizeWithReshapeTest(test_lib.TestCase):
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):
    if not test_lib.is_gpu_available():
      self.skipTest("Skip on GPU due to the pattern not supported")
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    in_array = np.random.uniform(low=-128, high=127, size=[1*3*224*224]).astype(np.int8)
    x = tf.constant(in_array, dtype=dtypes.int8)
    x = tf.math.greater_equal(x, 0)
    x = tf.where(x)
    x = tf.cast(x, dtypes.qint8)
    x_min = -5.0
    x_max = 5.0
    dequantize_op = array_ops.dequantize(x, x_min, x_max, mode="SCALED")
    final_out = tf.reshape(dequantize_op, [1, -1])
    final_out = tf.math.greater_equal(final_out, 0)
    with self.session(use_gpu=True) as sess:
        output_val = sess.run(final_out, options=run_options, run_metadata=metadata)
        graph = metadata.partition_graphs[0]

    existing_pattern = False
    for node in graph.node:
        if 'FusedDequantizeWithReshape' in node.op:
            existing_pattern = True
            break
    self.assertTrue(existing_pattern)

if __name__ == "__main__":
  test_lib.main()
