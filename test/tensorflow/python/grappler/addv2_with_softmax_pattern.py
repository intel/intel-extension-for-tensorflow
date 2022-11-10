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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

@test_util.run_all_in_graph_and_eager_modes
class AddWithSoftmaxTest(test_lib.TestCase):
  @test_util.run_deprecated_v1
  def _npSoftmax(self, features, dim=-1, log=False):
    if dim == -1:
      dim = len(features.shape) - 1
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    is_fp16 = features.dtype == np.float16
    if is_fp16:
      # Do the compute in fp32 and cast the input back to fp32.
      features = features.astype(np.float32)
    e = np.exp(features - np.reshape(
        np.amax(
            features, axis=dim), one_only_on_dim))
    softmax = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    if log:
      res = np.log(softmax)
    else:
      res = softmax
    if is_fp16:
      res = res.astype(np.float16)
    return res

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):
    if not test_lib.is_gpu_available():
      self.skipTest("Skip on CPU due to the pattern not supported")
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    
    left_shape_lst = [[2, 1, 4, 2], [2, 2, 4, 2], [2, 2, 1, 4, 2], [2, 2, 2, 4, 2]]
    right_shape_lst = [[2, 1, 4, 2], [2, 1, 4, 2], [2, 2, 2, 4, 2], [2, 2, 1, 1, 2]]
    dtypes_lst = [np.float32, np.float16]
    for left_shape in left_shape_lst:
      for right_shape in right_shape_lst:
        for dtype in dtypes_lst:
          np_features = np.random.uniform(low=1.0, high=1.0, size=left_shape).astype(dtype)
          adder_features = np.random.uniform(low=0.0, high=1.0, size=right_shape).astype(dtype)

          x_tensor = tf.constant(np_features)
          adder_tensor = tf.constant(adder_features)
          new_adder = math_ops.add_v2(x_tensor, adder_tensor)
          out = nn_ops.softmax(new_adder, axis = -1)
          final_out = array_ops.identity(out)
          np_softmax = self._npSoftmax(np.array(np_features+adder_features), dim=-1, log=False)
          with self.session(use_gpu=True) as sess:
              output_val = sess.run(final_out, options=run_options, run_metadata=metadata)
              graph = metadata.partition_graphs[0]
          
          can_fuse = (len(left_shape) == len(right_shape) 
                      and len(right_shape) == 4 
                      and left_shape[0] == right_shape[0] 
                      and left_shape[2] == right_shape[2]
                      and left_shape[3] == right_shape[3]
                      and right_shape[1] == 0)
                      
          existing_pattern = False
          for node in graph.node:
              if '_ITEXFusedAddV2WithSoftmax' in node.op:
                  existing_pattern = True
                  break
          if can_fuse:
            self.assertTrue(existing_pattern)
          self.assertAllClose(np_softmax, output_val)
    
if __name__ == "__main__":
  test_lib.main()
