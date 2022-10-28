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
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

tf.compat.v1.disable_eager_execution()

@test_util.run_all_in_native_and_block_format
class FusedMatMulTest(test_util.TensorFlowTestCase):
  """test fused matmul"""

  # set ITEX_VLOG=1, you can see log like:
  # _ITEXFusedMatMul[T=DT_FLOAT, _XlaHasReferenceVars=false, epsilon=0, fused_ops=["BiasAdd"],
  # num_args=1, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/device:XPU:0"]
  def testFuseBiasAdd(self):
    x = tf.compat.v1.placeholder(tf.float32, shape=(3, 4))
    y = tf.compat.v1.placeholder(tf.float32, shape=(4, 5))
    b = np.random.rand(5).astype(np.float32)

    x_arr = np.random.rand(3, 4)
    y_arr = np.random.rand(4, 5)
    with self.session(use_gpu=False) as sess:
      fused = array_ops.identity(tf.nn.bias_add(tf.matmul(x, y), b))
      ret_cpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr})
    with self.session(use_gpu=True) as sess:
      fused = array_ops.identity(tf.nn.bias_add(tf.matmul(x, y), b))
      ret_gpu = sess.run(fused, feed_dict={x: x_arr, y: y_arr})
    self.assertAllClose(ret_cpu, ret_gpu)

  # set ITEX_VLOG=1, you can see log like:
  # _FusedBatchMatMulV2[T=DT_FLOAT, _XlaHasReferenceVars=false, adj_x=false, adj_y=false, fused_ops=["Mul"],
  # num_args=1, _device="/job:localhost/replica:0/task:0/device:XPU:0"]
  def testFuseMul(self):
    x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 5))
    y = tf.compat.v1.placeholder(tf.float32, shape=(2, 5, 5))
    scale = np.array([2.0], dtype=np.float32)

    x_arr = np.random.rand(1, 5, 5)
    y_arr = np.random.rand(2, 5, 5)
    with self.session(use_gpu=False) as sess:
      bmm = math_ops.matmul(x, y, transpose_a=False, transpose_b=False)
      # CPU does not support the fusion of BatchMatMulV2 + Mul
      ret_cpu = sess.run(array_ops.identity(tf.math.multiply(bmm, scale)), feed_dict={x: x_arr, y: y_arr})
    with self.session(use_gpu=True) as sess:
      bmm = math_ops.matmul(x, y, transpose_a=False, transpose_b=False)
      fused = tf.math.multiply(bmm, scale)
      ret_gpu = sess.run(array_ops.identity(fused), feed_dict={x: x_arr, y: y_arr})
    self.assertAllClose(ret_cpu, ret_gpu)

  def testUnsuccessInPlace(self):
    x = tf.compat.v1.placeholder(tf.float32, shape=(3, 3))
    y = tf.compat.v1.placeholder(tf.float32, shape=(3, 3))
    b = np.random.rand(3).astype(np.float32)

    x_arr = np.random.rand(3, 3)
    y_arr = np.random.rand(3, 3)

    with self.session(use_gpu=False) as sess:
      fused = tf.identity(math_ops.add_n([x, tf.nn.bias_add(tf.matmul(x, y), b)]))
      sess.run(fused, feed_dict={x: x_arr, y: y_arr})


if __name__ == '__main__':
    test.main()
