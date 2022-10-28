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



import tensorflow as tf
import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.core.protobuf import config_pb2
import time
import os
import subprocess
import sys

tf.compat.v1.disable_eager_execution()
class FusedMatMulTest(test_util.TensorFlowTestCase):
    """test fused matmul"""
    # {{node Mul}} = _OneDnnFusedBatchMatMulV2[T=DT_FLOAT, _XlaHasReferenceVars=false, adj_x=false, adj_y=false, fused_ops=["Mul"], 
    # is_filter_const=false, num_args=1, _device="/job:localhost/replica:0/task:0/device:XPU:0"]
    # (_arg_Placeholder_0_0/_9, _arg_Placeholder_1_0_1/_11, Mul/y, Placeholder_DMT_3, Placeholder_1_DMT_4, Mul/y_DMT_5) 
    # device: /job:localhost/replica:0/task:0/device:XPU:0

    def testFuseMul(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 5))
        y = tf.compat.v1.placeholder(tf.float32, shape=(2, 5, 5))
        scale = np.array([2.0], dtype=np.float32)

        x_arr = np.random.rand(1, 5, 5)
        y_arr = np.random.rand(2, 5, 5)
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        bmm = math_ops.matmul(x, y, transpose_a=False, transpose_b=False)
        fused = tf.math.multiply(bmm, scale)
        fused = array_ops.identity(fused)

        with self.session(use_gpu=True) as sess:
            os.environ['ITEX_ENABLE_REMAPPER'] = '1'
            start_time = time.time()
            ret_gpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr},options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_FusedBatchMatMulV2'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'Mul'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
        with self.session(use_gpu=False) as sess:
            os.environ['ITEX_ENABLE_REMAPPER'] = '0'
            # CPU does not support the fusion of BatchMatMulV2 + Mul
            ret_cpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr})
        
        self.assertAllClose(ret_cpu, ret_gpu)


if __name__ == '__main__':
    test.main()
