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
import os

import tensorflow.compat.v1 as tf

try:
  from intel_extension_for_tensorflow.python.test_func import test as test_lib
except ImportError:
  from tensorflow.python.platform import test as test_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.core.protobuf import config_pb2


class FusedBinaryTest(test_lib.TestCase):

  def _testGraphStructure(self, shape1):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    in_array = np.random.uniform(size=np.prod(shape1))
    in_array = in_array.astype(np.float32).reshape(shape1)
    in_x = tf.placeholder(tf.float32, shape=shape1)
    in_z = tf.placeholder(tf.float32, shape=shape1)

    dtypes = [tf.float32, tf.bfloat16]
    if tf.config.list_physical_devices('XPU'):
      dtypes += [tf.half]
    for dtype in dtypes:
      in_x_d = tf.cast(in_x, dtype=dtype)
      in_z_d = tf.cast(in_z, dtype=dtype)
      x = in_x_d - 1
      x = tf.math.multiply(x, in_z_d)
      x = x - in_z_d
      x = tf.reshape(x, [-1])

      with self.session(use_gpu=True) as sess:
        output_val = sess.run(x, options=run_options, run_metadata=metadata,
                              feed_dict={in_x: in_array, in_z: in_array})
        graph = metadata.partition_graphs[0]

      existing_pattern = False
      for node in graph.node:
        if 'ITEXFusedBinary' in node.op:
          existing_pattern = True
          break
      self.assertTrue(existing_pattern)

      y_atol = 1e-6

      if dtype is not tf.float32:
        in_array = in_array.astype(np.float16)
        y_atol = 1e-3
        if dtype is tf.bfloat16:
          y_atol = 6e-3

      y = in_array - 1
      y = y * in_array
      y = y - in_array
      y = y.reshape((-1))
      self.assertAllClose(output_val, y, atol=y_atol)

  def _testGraphStructureSub1(self, shape1):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    in_array = np.random.uniform(size=np.prod(shape1))
    in_array = in_array.astype(np.float32).reshape(shape1)
    in_x = tf.placeholder(tf.float32, shape=shape1)
    in_z = tf.placeholder(tf.float32, shape=shape1)
    dtypes = [tf.float32, tf.bfloat16]
    if tf.config.list_physical_devices('XPU'):
      dtypes += [tf.half]
    for dtype in dtypes:
      in_x_d = tf.cast(in_x, dtype=dtype)
      in_z_d = tf.cast(in_z, dtype=dtype)
      x = 1 - in_x_d
      x = tf.math.multiply(x, in_z_d)
      x = in_z_d - x
      x = tf.reshape(x, [-1])

      with self.session(use_gpu=True) as sess:
        output_val = sess.run(x, options=run_options, run_metadata=metadata,
                              feed_dict={in_x: in_array, in_z: in_array})
        graph = metadata.partition_graphs[0]

      existing_pattern = False
      for node in graph.node:
        if 'ITEXFusedBinary' in node.op:
          existing_pattern = True
          break
      self.assertTrue(existing_pattern)

      y_atol = 1e-6

      if dtype is not tf.float32:
        in_array = in_array.astype(np.float16)
        y_atol = 1e-3
        if dtype is tf.bfloat16:
          y_atol = 1e-2

      y = 1 - in_array
      y = y * in_array
      y = in_array - y
      y = y.reshape((-1))
      self.assertAllClose(output_val, y, atol=y_atol)

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructure(self):
    self._testGraphStructure((16, 16, 512, 512))
    self._testGraphStructure((64, 64))

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testGraphStructureSub1(self):
    self._testGraphStructureSub1((16, 16, 512, 512))
    self._testGraphStructureSub1((1280, 16))

  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def testShape(self):
    shape = (1, 1)
    scalar = 2.0

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    for dtype in [float, np.float64]:
      in_x = tf.placeholder(tf.float32, shape=shape)
      in_y = tf.placeholder(tf.float32, shape=shape)
      in_array = np.random.uniform(size=np.prod(shape)).astype(dtype).reshape(shape)
      with self.session(use_gpu=True) as sess:
        x = tf.math.add(tf.math.multiply(in_x, in_y), scalar)
        x = array_ops.identity(x)
        output_val = sess.run(x, options=run_options, run_metadata=metadata,
                              feed_dict={in_x: in_array, in_y: in_array})

        graph = metadata.partition_graphs[0]
        existing_pattern = False
        if os.getenv('ITEX_REMAPPER') == '1':
          for node in graph.node:
            if 'ITEXFusedBinary' in node.op:
              existing_pattern = True
              break
          self.assertTrue(existing_pattern)

        self.assertTrue(output_val.shape == shape)


if __name__ == "__main__":
  test_lib.main()
