# Copyright (c) 2023 Intel Corporation
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

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variables

import numpy as np

class DynamicGRUOpTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def testDynamicGRU(self):
    time_steps = 8
    num_units = 5
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)
    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    concat_inputs =  array_ops.placeholder(
        dtypes.float32, shape=(time_steps, batch_size, input_size))

    cell = rnn_cell_impl.GRUCell(num_units=num_units)

    outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
        cell,
        inputs=concat_inputs,
        sequence_length=sequence_length,
        time_major=True,
        dtype=dtypes.float32)

    feeds = {concat_inputs: input_values}

    with self.session() as sess:
      variables.global_variables_initializer().run(feed_dict=feeds)
      sess.run([outputs_dynamic, state_dynamic], feed_dict=feeds)

    self.assertEqual(outputs_dynamic.shape, concat_inputs.shape)

  @test_util.deprecated_graph_mode_only
  def testMklGRU(self):
    # GPU doesn't support MklGRU.
    if test.is_gpu_available():
      self.skipTest("Skip on GPU due to the pattern not supported")

    with self.session() as sess:
      x = random_ops.random_uniform([36, 36, 36])
      h_prev = constant_op.constant([0.0], shape=[36, 36])
      w_ru = random_ops.random_uniform([72, 72])
      w_c = random_ops.random_uniform([72, 36])
      b_ru = random_ops.random_uniform([72])
      b_c = random_ops.random_uniform([36])
      
      output = load_ops_library.MklGRU(x=x, h_prev=h_prev, w_ru=w_ru, w_c=w_c,
                                       b_ru=b_ru, b_c=b_c, is_filter_const=True,
                                       TimeDim=100, x_format="NTC")
      r = self.evaluate(output)
      h_out = r.h_out
      h_n = r.h_n

    self.assertEqual(x.shape, h_out.shape)
    self.assertEqual(h_prev.shape, h_n.shape)


if __name__ == "__main__":
  test.main()
