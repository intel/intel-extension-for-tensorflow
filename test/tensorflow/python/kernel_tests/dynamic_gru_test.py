
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
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


if __name__ == "__main__":
  test.main()
