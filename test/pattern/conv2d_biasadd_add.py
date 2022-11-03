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
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import time
import os

tf.compat.v1.disable_eager_execution()

class FusedConv2DTest(test_util.TensorFlowTestCase):

    def testConv2dBiasAdd(self):
        x = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
        w = np.ones([1, 2, 1, 1]).astype(np.float32)
        b = np.array([1], dtype=np.float32)
          
        conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
        conv2 = tf.nn.conv2d(conv1, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
        conv = tf.nn.conv2d(conv2, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
        
        bias_add = nn_ops.bias_add(conv, b)
        output= array_ops.identity(math_ops.add_n([bias_add, conv2]))
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()

        with self.session(use_gpu=True) as sess:
            start_time = time.time()
            ret_gpu = sess.run(output, options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if "FusedConv2D" in node.op:
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 2 and fused_ops[0] == b'BiasAdd' and fused_ops[1] == b'Add'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
        with self.session(use_gpu=False) as sess:
            ret_ref = sess.run(output,options=run_options, run_metadata=metadata)

        self.assertAllClose(ret_ref, ret_gpu)  
        
if __name__ == '__main__':
    test.main()
