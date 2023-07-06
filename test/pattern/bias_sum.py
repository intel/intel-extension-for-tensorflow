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
import time
import tensorflow as tf
import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_grad 
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

tf.compat.v1.disable_eager_execution()
class BiasSumTest(test_util.TensorFlowTestCase):
    def testAddGrad(self):
        class op:
            def __init__(self):
                self.skip_input_indices = None
                self.inputs = []

        dtype = dtypes.float32
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()

        x_arr = np.array([1, 2, 3, 4, 5, 6],
                         dtype=dtype.as_numpy_dtype).reshape(3, 2)
        y_arr = np.array([1, 2, 3, 4],
                         dtype=dtype.as_numpy_dtype).reshape(2, 2)
        b_arr = np.array([1, 2], dtype=dtype.as_numpy_dtype)

        x = tf.compat.v1.placeholder(dtype, shape=(None, 2))
        y = tf.compat.v1.placeholder(dtype, shape=(2, 2))
        b = tf.compat.v1.placeholder(dtype, shape=(2))

        r_m = tf.matmul(x, y)
        r = r_m + b

        o = op()
        o.inputs.append(r_m)
        o.inputs.append(b)
        grad = r - random_ops.random_uniform([3, 2])
        gx, gy = math_grad._AddGrad(o, grad)
        r = gx + gy

        with self.session(use_gpu=True) as sess:
            start_time = time.time()
            ret = sess.run(r, feed_dict={x: x_arr, y: y_arr, b: b_arr},
                           options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            found_non_fused_op = False
            for node in graph.node:
                if node.op in ('BiasAddGrad'):
                    found_fused_op = True
                if node.op in ('Sum'):
                    found_non_fused_op = True

            is_found = found_fused_op and not found_non_fused_op
            self.assertTrue(is_found, "this pattern has fusion issue!!")

        os.environ['ITEX_REMAPPER'] = '0'
        with self.session(use_gpu=True) as sess:
            ret_expect = sess.run(r, feed_dict={x: x_arr, y: y_arr, b: b_arr},
                                  options=run_options, run_metadata=metadata)
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('BiasAddGrad'):
                    found_fused_op = True
                    break
            self.assertFalse(found_fused_op, "this pattern has fusion issue!!")

        self.assertAllClose(ret, ret_expect, atol=1e-5, rtol=1e-5)

if __name__ == '__main__':
  test.main()
