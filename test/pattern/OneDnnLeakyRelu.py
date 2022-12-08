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
class OneDnnLeakyReluTest(test_util.TensorFlowTestCase):

  def _test_impl(self, coef):
    x = coef * np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
    w = np.ones([1, 2, 1, 1]).astype(np.float32)
          
    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
    mid= array_ops.identity(math_ops.add_n([conv, conv]))

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    output = array_ops.identity(nn_ops.leaky_relu(mid))

    with self.session(use_gpu=True) as sess:
      os.environ['ITEX_REMAPPER'] = '1'
      os.environ['ITEX_LAYOUT_OPT'] = '1'
      start_time = time.time()
      ret_np = sess.run(mid, options=run_options, run_metadata=metadata)
      ret_gpu = sess.run(output, options=run_options, run_metadata=metadata)
      duration = time.time() - start_time
      print("end to end duration is : {}".format(duration))
      # Graph should contain output op.
      graph = metadata.partition_graphs[0]
      found_op = False
      for node in graph.node:
        if '_OneDnnLeakyRelu' in node.op:
          found_op = True
          break
      self.assertTrue(found_op, "this pattern has rewrite issue!!")
      if coef < 0:
       self.assertAllClose(0.2 * ret_np, ret_gpu)
      else:
       self.assertAllClose(ret_np, ret_gpu)

    with self.session(use_gpu=False) as sess:
      os.environ['ITEX_REMAPPER'] = '0'
      ret_cpu = sess.run(output, options=run_options, run_metadata=metadata)
    self.assertAllClose(ret_cpu, ret_gpu)

  def testOneDnnLeakyRelu(self):
    if not tf.config.list_physical_devices('XPU'):
      self.skipTest("No GPU available")
    for coef in (1, -1):
      self._test_impl(coef)

if __name__ == '__main__':
  test.main()
