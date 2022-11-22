import numpy as np
import tensorflow as tf

time_steps = 8
num_units = 3
input_size = 5
batch_size = 2

input_values = np.random.randn(time_steps, batch_size, input_size)
sequence_length = np.random.randint(0, time_steps, size=batch_size)

with tf.compat.v1.Session() as sess:
  concat_inputs =  tf.compat.v1.placeholder(
      tf.dtypes.float32, shape=(time_steps, batch_size, input_size))

  cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=num_units)

  with tf.compat.v1.variable_scope("dynamic_scope"):
    outputs_dynamic, state_dynamic = tf.compat.v1.nn.dynamic_rnn(
        cell,
        inputs=concat_inputs,
        sequence_length=sequence_length,
        time_major=True,
        dtype=tf.dtypes.float32)

  feeds = {concat_inputs: input_values}

  # Initialize
  tf.compat.v1.global_variables_initializer().run(feed_dict=feeds)

  sess.run([outputs_dynamic, state_dynamic], feed_dict=feeds)
