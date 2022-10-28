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

from intel_extension_for_tensorflow.python.test_func import test as test_lib
from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2

@test_util.run_all_in_native_and_block_format
@test_util.run_all_in_graph_and_eager_modes
class BatchMatMulWithMulAndAddTest(test_lib.TestCase):
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):
    if test_lib.is_gpu_available():
      self.skipTest("Skip on GPU due to the pattern not supported")
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a = constant_op.constant(np.arange(1, 25, dtype=np.float32), shape=[2, 2, 2, 3])
    a2 = constant_op.constant(np.arange(25, 49, dtype=np.float32), shape=[2, 2, 2, 3])
    b = constant_op.constant(np.arange(25, 49, dtype=np.float32), shape=[2, 2, 3, 2])
    scale_value = constant_op.constant(np.array(3, dtype=np.float32), shape=[1])
    badd_value = constant_op.constant(np.arange(1, 9, dtype=np.float32), shape=[2, 1, 2, 2])
    a = math_ops.add_n([a,a2,a])

    c1 = math_ops.matmul(a, b)
    c1 = math_ops.mul(c1, scale_value)
    d1 = math_ops.add(c1, badd_value)
    d1 = array_ops.identity(d1)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]

    existing_pattern = False
    for node in graph.node:
        if 'BatchMatMulV2' in node.op:
            fused_ops = node.attr['fused_ops'].list.s
            existing_pattern = len(fused_ops) == 2 and fused_ops[0] == b"Mul" and fused_ops[1] == b"BinaryAdd"
            break

    self.assertTrue(existing_pattern)

if __name__ == "__main__":
  test_lib.main()
