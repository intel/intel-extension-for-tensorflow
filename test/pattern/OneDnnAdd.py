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
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.core.protobuf import config_pb2
import time
import os

tf.compat.v1.disable_eager_execution()
class OneDnnAddTest(test_util.TensorFlowTestCase):

  def testOneDnnAdd(self):
    if not tf.config.list_physical_devices('XPU'):
      self.skipTest("No GPU available")
    x = tf.compat.v1.placeholder(tf.float32, shape=(10, 10, 10, 10, 10))
    y = tf.compat.v1.placeholder(tf.float32, shape=(10, 10, 10, 10, 10))
    w = tf.compat.v1.placeholder(tf.float32, shape=(10, 10, 10, 10, 10))
    x_arr = np.random.rand(10, 10, 10, 10, 10)
    y_arr = np.random.rand(10, 10, 10, 10, 10)
    w_arr = np.random.rand(10, 10, 10, 10, 10)

    conv1 = nn_ops.Conv3D(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')
    conv2 = nn_ops.Conv3D(input=y, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',data_format='NDHWC')

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    fused = array_ops.identity(math_ops.Add(x=conv1, y=conv2))

    with self.session(use_gpu=True) as sess:
      os.environ['ITEX_REMAPPER'] = '1'
      os.environ['ITEX_LAYOUT_OPT'] = '1'
      start_time = time.time()
      ret_gpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr, w: w_arr}, options=run_options, run_metadata=metadata)
      duration = time.time() - start_time
      print("end to end duration is : {}".format(duration))
      # Graph should contain fused op.
      graph = metadata.partition_graphs[0]
      found_op = False
      for node in graph.node:
        if '_OneDnnAdd' in node.op:
          found_op = True
          break
      self.assertTrue(found_op, "this pattern has rewrite issue!!")
    with self.session(use_gpu=False) as sess:
      os.environ['ITEX_REMAPPER'] = '0'
      ret_cpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr, w: w_arr}, options=run_options, run_metadata=metadata)
    self.assertAllClose(ret_cpu, ret_gpu)

if __name__ == '__main__':
  test.main()
