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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.ops import math_ops
import numpy as np
import time
import os
import subprocess
import sys

tf.compat.v1.disable_eager_execution()
class FusedConv3DTest(test_util.TensorFlowTestCase):
    
    def testFuseBiasAddGelu_erf(self):
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 8, 7, 1))
        w = tf.compat.v1.placeholder(tf.float32, shape=(1, 2, 3, 1, 1))
        b = np.random.rand(1).astype(np.float32)

        x_arr = np.random.rand(1, 5, 8, 7, 1)
        w_arr = np.random.rand(1, 2, 3, 1, 1)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()

        conv1 = nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        conv2 = nn_ops.Conv3D(input=conv1, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        conv = nn_ops.Conv3D(input=conv2, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        bias_add = nn_ops.bias_add(conv, b)
        output= array_ops.identity(math_ops.add_n([bias_add, conv2]))
        conv_bias_gelu_erf = load_ops_library.itex_gelu(output,approximate=False)
        fused = array_ops.identity(conv_bias_gelu_erf)
        
        # fused pattern output value from gpu side
        with self.session(use_gpu=True) as sess:
            os.environ['ITEX_ENABLE_REMAPPER'] = '1'
            start_time = time.time()
            ret_gpu = sess.run(fused, feed_dict={x: x_arr, w: w_arr},options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))

            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if "FusedConv3D" in node.op:
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 3 and fused_ops[0] == b'BiasAdd' and fused_ops[1] == b'Add' and fused_ops[2] == b'GeluExact'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
            
        # reference value which is no fusion
        with self.session(use_gpu=False) as sess:
            os.environ['ITEX_ENABLE_REMAPPER'] = '0'
            ret_ref = sess.run(fused, feed_dict={x: x_arr, w: w_arr},options=run_options, run_metadata=metadata)

        
        
        self.assertAllClose(ret_ref, ret_gpu)  


if __name__ == '__main__':
    test.main()
