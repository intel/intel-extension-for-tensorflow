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

from curses import meta
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
class ComparisonWithCastTest(test_util.TensorFlowTestCase):
    """test comparison op + cast"""
    def _init_input(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 5))
        y = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 5))
        x_arr = np.random.rand(1, 5, 5)
        y_arr = np.random.rand(1, 5, 5)

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        return x, y, x_arr, y_arr, run_options, metadata

    def _test_fusion(self, fused, fused_op_name, x, y, x_arr, y_arr, run_options, metadata):
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
                if node.op in (fused_op_name):
                    found_fused_op = True
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
        with self.session(use_gpu=False) as sess:
            os.environ['ITEX_ENABLE_REMAPPER'] = '0'
            # CPU does not support the fusion of comparison op + Cast
            ret_cpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr})

        self.assertAllClose(ret_cpu, ret_gpu)

    def testGreaterEqualWithCast(self):
        x, y, x_arr, y_arr, run_options, metadata = self._init_input()
        mid = math_ops.greater_equal(x, y)
        fused = math_ops.cast(mid, tf.float32)
        fused = array_ops.identity(fused)
        self._test_fusion(fused, '_ITEXGreaterEqualWithCast', x, y, x_arr, y_arr, run_options, metadata)

    def testLessEqualWithCast(self):
        x, y, x_arr, y_arr, run_options, metadata = self._init_input()
        mid = math_ops.less_equal(x, y)
        fused = math_ops.cast(mid, tf.float32)
        fused = array_ops.identity(fused)
        self._test_fusion(fused, '_ITEXLessEqualWithCast', x, y, x_arr, y_arr, run_options, metadata)

    def testNotEqualWithCast(self):
        x, y, x_arr, y_arr, run_options, metadata = self._init_input()
        mid = math_ops.not_equal(x, y)
        fused = math_ops.cast(mid, tf.float32)
        fused = array_ops.identity(fused)
        self._test_fusion(fused, '_ITEXNotEqualWithCast', x, y, x_arr, y_arr, run_options, metadata)

if __name__ == '__main__':
    test.main()

