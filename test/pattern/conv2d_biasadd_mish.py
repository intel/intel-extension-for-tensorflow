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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.core.protobuf import config_pb2
import time
import os
import subprocess
import sys
def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

tf.compat.v1.disable_eager_execution()
class FusedConv2DTest(test_util.TensorFlowTestCase):
    """test _FusedITEXConv2D"""
    def test_conv2d_biasadd_mish(self):
        x = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
        w = np.ones([1, 2, 1, 1]).astype(np.float32)
        b = np.array([1], dtype=np.float32)
        x_arr = np.random.rand(1, 5, 8, 7, 1)
        w_arr = np.random.rand(1, 2, 3, 1, 1)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        expected_result = np.array([[[[4.], [6.], [4.]],
                                    [[10.], [12.], [7.]],
                                    [[16.], [18.], [10.]]]])
        expected_result = expected_result * (np.tanh(softplus_np(expected_result)))
        with self.session(use_gpu=True) as sess:
            fused_graph = tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC'), b)
            fused_graph = tf.math.multiply(fused_graph, tf.tanh(tf.math.softplus(fused_graph)))
            fused = array_ops.identity(fused_graph) 
            result = sess.run(fused,options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if "FusedConv2D" in node.op:
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 2 and fused_ops[0] == b'BiasAdd' and fused_ops[1] == b'_ITEXMish'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
            self.assertAllClose(expected_result, result)

if __name__ == '__main__':
    test.main()
