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
import tensorflow as tf

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test as test_lib

from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.core.protobuf import config_pb2
import time
import os
import subprocess
import sys

tf.compat.v1.disable_eager_execution()

@test_util.run_all_in_native_and_block_format
class WeightCacheTest(test_util.TensorFlowTestCase):
    def test_conv2d(self):
        x = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
        w = np.ones([1, 2, 1, 1]).astype(np.float32)
        b = np.array([1], dtype=np.float32)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        expected_result = np.array([[[[4.], [6.], [4.]],
                                    [[10.], [12.], [7.]],
                                    [[16.], [18.], [10.]]]])
        with self.session() as sess:
            graph = tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC'), b)
            fused = array_ops.identity(graph)
            for i in range(2):
              result = sess.run(fused,options=run_options, run_metadata=metadata)
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            is_filter_const = False
            for node in graph.node:
                if 'FusedConv2D' in node.op:
                    is_filter_const = node.attr['is_filter_const'].b
                    # TODO(itex): GPU plain format does not support weight cahce for now.
                    # TODO(itex): GPU plain format does not run native layout.
                    if test_lib.is_gpu_available() and node.op == "_ITEXFusedConv2D":
                        is_filter_const = True
                    break
            self.assertTrue(is_filter_const, "weight cache is failed!")
            self.assertAllClose(expected_result, result)

if __name__ == '__main__':
    test.main()
