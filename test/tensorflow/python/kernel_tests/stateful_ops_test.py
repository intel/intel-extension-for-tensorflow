#copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for stateful_ops.py."""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variables
from tensorflow import raw_ops
from tensorflow.python.framework import ops
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.test_func import test_util

class StatefulOpsTest(test.TestCase):

  def testStatefulUniformInt(self):
    input = variables.Variable([1,2,3,4], dtype=tf.int64)
    output = raw_ops.StatefulUniformInt(resource = input.handle, algorithm = 1, shape = [2,1], minval = 0, maxval = 10)
    self.assertGreaterEqual(output.numpy().min(), 0)
    self.assertLess(output.numpy().max(), 10)

  def testStatefulUniform(self):
    input = variables.Variable([1,2,3,4], dtype=tf.int64)
    output = raw_ops.StatefulUniform(resource = input.handle, algorithm = 1, shape = [2,1])
    self.assertGreaterEqual(output.numpy().min(), 0.0)
    self.assertLess(output.numpy().min(), 1.0)

  def testStatefulUniformFullInt(self):
    input = variables.Variable([1,2,3,4], dtype=tf.int64)
    output = raw_ops.StatefulUniformFullInt(resource = input.handle, algorithm = 1, shape = [2,1], dtype=tf.dtypes.uint64)
    self.assertEqual(output.dtype, tf.dtypes.uint64)

  def testStatefulTruncatedNormal(self):
    input = variables.Variable([1,2,3,4], dtype=tf.int64)
    output = raw_ops.StatefulTruncatedNormal(resource = input.handle, algorithm = 1, shape = [10000000])
    diff = abs(output.numpy().mean())
    self.assertLess(diff, 0.1)

if __name__ == "__main__":
  test.main()

