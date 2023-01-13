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


import time
import os
import numpy as np
import tensorflow as tf
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test as test_lib
from tensorflow.python.platform import test
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.core.protobuf import config_pb2


tf.compat.v1.disable_eager_execution()
@test_util.run_all_in_native_and_block_format
class FusedMatMulTest(test_util.TensorFlowTestCase):
    """test fused matmul"""
    # {{node Mul}} = _OneDnnFusedBatchMatMulV2[T=DT_FLOAT, _XlaHasReferenceVars=false, adj_x=false, adj_y=false, fused_ops=["BinaryMul"], 
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
                if 'FusedBatchMatMulV2' in node.op:
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'BinaryMul'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
        with self.session(use_gpu=False) as sess:
            os.environ['ITEX_ENABLE_REMAPPER'] = '0'
            # CPU does not support the fusion of BatchMatMulV2 + Mul
            ret_cpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr})
        
        self.assertAllClose(ret_cpu, ret_gpu)

    def testOneDnnFusedBatchMatMulV2(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(10, 10, 10, 10, 10))
        y = tf.compat.v1.placeholder(tf.float32, shape=(10, 10, 10, 10, 10))
        w = tf.compat.v1.placeholder(tf.float32, shape=(10, 10, 10, 10, 10))
        x_arr = np.random.rand(10, 10, 10, 10, 10)
        y_arr = np.random.rand(10, 10, 10, 10, 10)
        w_arr = np.random.rand(10, 10, 10, 10, 10)

        conv1 = nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        conv2 = nn_ops.Conv3D(input=y, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
        scale = np.array([2.0], dtype=np.float32)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        bmm = math_ops.matmul(conv1, conv2, transpose_a=False, transpose_b=False)
        fused = tf.math.multiply(bmm, scale)
        fused = array_ops.identity(fused)

        with self.session(use_gpu=True) as sess:
            os.environ['ITEX_ENABLE_REMAPPER'] = '1'
            start_time = time.time()
            ret_gpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr, w: w_arr}, options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if 'FusedBatchMatMulV2' in node.op:
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'BinaryMul'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")



@test_util.run_all_in_native_and_block_format
@test_util.run_all_in_graph_and_eager_modes
class MulAndBatchMatMulWithAddTest(test_lib.TestCase):
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
    add_value = constant_op.constant(np.arange(1, 9, dtype=np.float32), shape=[2, 1, 2, 2])
    a = math_ops.add_n([a,a2,a])

    mul = math_ops.mul(a, scale_value)
    bmm = math_ops.matmul(mul, b)
    out = math_ops.Add(x=bmm, y=add_value)
    out = array_ops.identity(out)


    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(out, options=run_options, run_metadata=metadata)
      print('output_val', output_val)
      graph = metadata.partition_graphs[0]

    existing_pattern = False
    for node in graph.node:
        if 'FusedBatchMatMulV2' in node.op:
            fused_ops = node.attr['fused_ops'].list.s
            existing_pattern = len(fused_ops) == 2 and fused_ops[0] == b"Mul" and fused_ops[1] == b"BinaryAdd"
            break

    self.assertTrue(existing_pattern)


if __name__ == '__main__':
    test.main()
    test_lib.main()