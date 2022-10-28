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
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
import time
import os
import subprocess
import sys

tf.compat.v1.disable_eager_execution()
class FusedMatMulTest(test_util.TensorFlowTestCase):
    """test fused matmul"""

    def testMatMulBiasAddGeluerf(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(3, 4))
        y = tf.compat.v1.placeholder(tf.float32, shape=(4, 5))
        b = np.random.rand(5).astype(np.float32)

        x_arr = np.random.rand(3, 4)
        y_arr = np.random.rand(4, 5)
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        fused = tf.nn.bias_add(tf.matmul(x, y), b)
        fused = load_ops_library.gelu(fused,approximate=False)
        fused = array_ops.identity(fused)

        with self.session(use_gpu=True) as sess:
            # fused = tf.nn.bias_add(tf.matmul(x, y), b)
            start_time = time.time()
            ret_gpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr},options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedMatMul'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 2 and fused_ops[0] == b'BiasAdd' and fused_ops[1] == b'GeluExact'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")

        with self.session(use_gpu=False) as sess:
            ret_cpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr})
        

        self.assertAllClose(ret_cpu, ret_gpu)

if __name__ == '__main__':
    test.main()
