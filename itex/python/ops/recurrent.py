# Copyright (c) 2021-2022 Intel Corporation
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-classes-have-attributes
"""Recurrent layers for TF 2."""

#import uuid

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
#from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
#from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python import keras
from keras import activations
try:
  from keras.src import backend
except ImportError:
  from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
try:
  from keras.engine.input_spec import InputSpec
  from keras.layers import LSTMV1
except ImportError:
  from keras.src.engine.input_spec import InputSpec
  from keras.src.layers import LSTMV1

_ITEX_AVAILABLE_MSG = 'Layer %s will use ITEX kernels when running on GPU.'
_ITEX_NOT_AVAILABLE_MSG = ('Layer %s will not use ITEX kernels since it '
                           'doesn\'t meet the criteria. It will '
                           'use a generic GPU kernel as fallback when running '
                           'on GPU.')

def _canonical_to_params(weights, biases, shape, transpose_weights=False):
  """Utility function convert variable to Itex compatible parameter.

  Note that Keras weights for kernels are different from the Itex format. Eg.:

  ```
    Keras                                       Itex
    kernel: (ic, 4 * hc)          <--------->  kernel: (4, hc, ic)
    recurrent_kernel: (hc, 4 * hc)             recurrent_kernel: (4, hc, hc)
  ```

  Args:
    weights: list of weights for the individual kernels and recurrent kernels.
    biases: list of biases for individual gate.
    shape: the shape for the converted variables that will be feed to Itex.
    transpose_weights: boolean, whether to transpose the weights.

  Returns:
    The converted weights that can be feed to Itex ops as param.
  """
  def convert(w):
    return array_ops.transpose(w) if transpose_weights else w

  weights = [array_ops.reshape(convert(x), shape) for x in weights]
  biases = [array_ops.reshape(x, shape) for x in biases]
  return array_ops.concat(weights + biases, axis=0)

def calculate_sequence_by_mask(mask, time_major):
  """Calculate the sequence length tensor (1-D) based on the masking tensor.

  The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
  any timestep that should be masked, the corresponding field will be False.
  Consider the following example:
    a = [[True, True, False, False],
         [True, True, True, False]]
  It is a (2, 4) tensor, and the corresponding sequence length result should be
  1D tensor with value [2, 3]. Note that the masking tensor must be right
  padded that could be checked by, e.g., `is_sequence_right_padded()`.

  Args:
    mask: Boolean tensor with shape [batch, timestep] or [timestep, batch] if
      time_major=True.
    time_major: Boolean, which indicates whether the mask is time major or batch
      major.
  Returns:
    sequence_length: 1D int32 tensor.
  """
  timestep_index = 0 if time_major else 1
  return math_ops.reduce_sum(math_ops.cast(mask, dtypes.int32),
                             axis=timestep_index)

def is_sequence_right_padded(mask):
  """Check the mask tensor and see if it right padded.

  For CuDNN kernel, it uses the sequence length param to skip the tailing
  timestep. If the data is left padded, or not a strict right padding (has
  masked value in the middle of the sequence), then CuDNN kernel won't be work
  properly in those cases.

  Left padded data: [[False, False, True, True, True]].
  Right padded data: [[True, True, True, False, False]].
  Mixture of mask/unmasked data: [[True, False, True, False, False]].

  Note that for the mixed data example above, the actually data RNN should see
  are those 2 Trues (index 0 and 2), the index 1 False should be ignored and not
  pollute the internal states.

  Args:
    mask: the Boolean tensor with shape [batch, timestep]

  Returns:
    boolean scalar tensor, whether the mask is strictly right padded.
  """
  max_seq_length = array_ops.shape(mask)[1]
  count_of_true = math_ops.reduce_sum(math_ops.cast(mask, dtypes.int32), axis=1)
  right_padded_mask = array_ops.sequence_mask(
      count_of_true, maxlen=max_seq_length)
  return math_ops.reduce_all(math_ops.equal(mask, right_padded_mask))

def has_fully_masked_sequence(mask):
  # See https://github.com/tensorflow/tensorflow/issues/33148 for more details.
  # Cudnn kernel will error out if the input sequence contains any fully masked
  # data. We walk around this issue by rerouting the computation to standard
  # kernel, until the issue on cudnn side has been fixed.
  # For a fully masked sequence, it will contain all Falses. To make it easy to
  # check, we inverse the boolean, check if any of the sequence has all True.
  return math_ops.reduce_any(
      math_ops.reduce_all(
          math_ops.logical_not(mask),
          axis=1))

# add Itex prefix to avoid name conflicting with keras LSTM
@keras.utils.generic_utils.register_keras_serializable(package="Itex")
class ItexLSTM(LSTMV1):
  """Long Short-Term Memory layer - Hochreiter 1997.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (ITEX-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the ITEX kernel
  (see below for details), the layer will use a fast ITEX implementation.

  The requirements to use the ITEX implementation are:
  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `use_bias` is `True`
  4. Inputs, if use masking, are strictly right-padded.
  5. Eager execution is enabled in the outermost context.

  For example:
  >>> import intel_extension_for_tensorflow as itex
  >>> inputs = tf.random.normal([32, 10, 8])
  >>> lstm = itex.ops.LSTM(4)
  >>> output = lstm(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> lstm = itex.ops.LSTM(4, return_sequences=True, return_state=True)
  >>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
  >>> print(whole_seq_output.shape)
  (32, 10, 4)
  >>> print(final_memory_state.shape)
  (32, 4)
  >>> print(final_carry_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
      is applied (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs. Default: `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
      the forget gate at initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition to the
      output. Default: `False`.
    go_backwards: Boolean (default `False`). If True, process the input sequence
      backwards and return the reversed sequence.
    stateful: Boolean (default `False`). If True, the last state for each sample
      at index i in a batch will be used as initial state for the sample of
      index i in the following batch.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    unroll: Boolean (default `False`). If True, the network will be unrolled,
      else a symbolic loop will be used. Unrolling can speed-up a RNN, although
      it tends to be more memory-intensive. Unrolling is only suitable for short
      sequences.

  Call arguments:
    inputs: A 3D tensor with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[batch, timesteps]` indicating whether
      a given timestep should be masked (optional, defaults to `None`).
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               time_major=False,
               unroll=False,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.

    super(ItexLSTM, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        time_major=time_major,
        unroll=unroll,
        **kwargs)

    self.state_spec = [
        InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
    ]
    # TODO: support non-bias case in the future
    self._could_use_itex_kernel = (
        self.activation in (activations.tanh, nn.tanh) and
        self.recurrent_activation in (activations.sigmoid, nn.sigmoid) and
        use_bias)

    if config.list_logical_devices('XPU'):
      # Only show the message when there is GPU available, itex LSTM only support GPU currently
      if self._could_use_itex_kernel:
        logging.debug(_ITEX_AVAILABLE_MSG % self.name)
      else:
        logging.warning(_ITEX_NOT_AVAILABLE_MSG % self.name)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    """A dummy docstring."""
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    # TODO: support ragged_input and mask in the future
    self._could_use_itex_kernel = (self._could_use_itex_kernel and \
       (not is_ragged_input))

    # LSTM does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(
        inputs, initial_state, None)

    self._maybe_reset_cell_dropout_mask(self.cell)

    if isinstance(mask, list):
      mask = mask[0]

    gpu_lstm_kwargs = {
        'cell': self.cell,
        'inputs': inputs,
        'mask': mask,
        'training': training,
        'initial_state': initial_state,
        'sequence_lengths': row_lengths,
        'go_backwards': self.go_backwards,
        'time_major': self.time_major,
    }
    normal_lstm_kwargs = gpu_lstm_kwargs.copy()

    normal_lstm_kwargs.update({
        'unroll': self.unroll,
        'zero_output_for_mask': self.zero_output_for_mask,
    })

    can_use_gpu = ((config.list_logical_devices('XPU')) and
                   (mask is None or is_itex_supported_inputs\
                   (mask, self.time_major)))
    if self._could_use_itex_kernel and can_use_gpu:
      last_output, outputs, new_h, new_c = gpu_lstm(
          **gpu_lstm_kwargs)
    else:
      # Fall back to use the normal LSTM.
      last_output, outputs, new_h, new_c = standard_lstm(
          **normal_lstm_kwargs)

    states = [new_h, new_c]
    if self.stateful:
      #Below cast is caused by states has differnet datat type with input when set stateful in official tensorflow
      #Maybe remove this in the future
      states = [math_ops.cast(i, self.states[0].dtype) for i  in states]
      updates = [
          state_ops.assign(self_state, state)
          for self_state, state in zip(self.states, states)
      ]
      self.add_update(updates)

    if self.return_sequences:
      output = backend.maybe_convert_to_ragged(
          is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards)
    else:
      output = last_output

    if self.return_state:
      return [output] + list(states)
    return output

  def get_config(self):
    """A dummy docstring."""
    derive_config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(ItexLSTM, self).get_config()
    return dict(list(base_config.items()) + list(derive_config.items()))

def standard_lstm(cell, inputs, mask, training, initial_state, sequence_lengths,
                  go_backwards, time_major, unroll, zero_output_for_mask):
  """LSTM with standard kernel implementation.

  Args:
    cell: a LSTM cell instance.
    inputs: input tensor of LSTM layer.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    unroll: Boolean (default `False`). If True, the network will be unrolled,
      else a symbolic loop will be used. Unrolling can speed-up a RNN, although
      it tends to be more memory-intensive. Unrolling is only suitable for short
      sequences.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    state_1: the cell hidden state, which has same shape as init_c.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = backend.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  kwargs = {'training': training}

  def step(inputs, states):
    return cell(inputs, states, **kwargs)

  last_output, outputs, new_states = backend.rnn(
      step,
      inputs,
      initial_state,
      go_backwards=go_backwards,
      mask=mask,
      unroll=unroll,
      input_length=sequence_lengths if sequence_lengths \
                   is not None else timesteps,
      time_major=time_major,
      zero_output_for_mask=zero_output_for_mask)
  return last_output, outputs, new_states[0], new_states[1]


def gpu_lstm(cell, inputs, mask, training, initial_state, sequence_lengths,
             go_backwards, time_major):
  """LSTM with ITEX implementation which is only available for GPU.

  Note that currently only right padded data is supported, or the result will be
  polluted by the unmasked data which should be filtered.

  Args:
    cell: a LSTM cell instance.
    inputs: input tensor of LSTM layer.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].

  Returns:
    last_output: Output tensor for the last timestep, which has shape
      [batch, units].
    outputs: Output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: The cell output, which has same shape as init_h.
    state_1: The cell hidden state, which has same shape as init_c.
    runtime: Constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should not be used by user.
  """
  #TODO: Below cast is caused by states has differnet datat type with input when set stateful in official tensorflow
  #Maybe remove this in the future
  init_h = math_ops.cast(_read_variable_value(initial_state[0]), inputs.dtype)
  init_c = math_ops.cast(_read_variable_value(initial_state[1]), inputs.dtype)
  kernel = math_ops.cast(_read_variable_value(cell.kernel), inputs.dtype)
  recurrent_kernel = math_ops.cast(_read_variable_value(cell.recurrent_kernel),\
                                   inputs.dtype)
  bias = math_ops.cast(_read_variable_value(cell.bias), inputs.dtype)

  if not time_major:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))

  weights = array_ops.split(kernel, 4, axis=1)
  weights += array_ops.split(recurrent_kernel, 4, axis=1)

  params = _canonical_to_params(
      weights=weights,
      biases=array_ops.split(bias, 4),
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  # TODO, generate mask in c++ side
  dropout = _read_variable_value(cell.dropout)
  if 0 < dropout < 1.0:
    dp_mask = cell.get_dropout_mask_for_cell(inputs[0], training, count=4)
    dp_mask = array_ops.concat(dp_mask, axis=0)
  else:
    dp_mask = 0

  recurrent_dropout = _read_variable_value(cell.recurrent_dropout)
  if 0 < recurrent_dropout < 1.0:
    rec_dp_mask = cell.get_recurrent_dropout_mask_for_cell(
        init_h, training, count=4)
    rec_dp_mask = array_ops.concat(rec_dp_mask, axis=0)
  else:
    rec_dp_mask = 0

  if mask is not None:
    sequence_lengths = calculate_sequence_by_mask(
        mask, time_major)

  if sequence_lengths is not None:
    if go_backwards:
      # Three reversals are required. E.g.,
      # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
      # reversed_input_to_itex = [3, 2, 1, 0, 0]
      # output_from_itex = [6, 5, 4, 0, 0]
      # expected_output = [0, 0, 6, 5 ,4]
      inputs = array_ops.reverse_sequence_v2(
          inputs, sequence_lengths, seq_axis=0, batch_axis=1)
    outputs, h, c, _ = load_ops_library.itex_rnn(
        input=inputs,
        input_h=init_h,
        input_c=init_c,
        params=params,
        dropout=dropout,
        dropout_mask=dp_mask,
        recurrent_dropout=recurrent_dropout,
        recurrent_dropout_mask=rec_dp_mask,
        sequence_lengths=sequence_lengths,
        rnn_mode='lstm',
        var_seq_length=True,
        is_training=training)
    outputs = array_ops.reshape(outputs,
                                [array_ops.shape(inputs)[0], \
                                array_ops.shape(inputs)[1], \
                                array_ops.shape(init_h)[1]])
    h = array_ops.reshape(h, [array_ops.shape(init_h)[0], \
                          array_ops.shape(init_h)[1]])
    c = array_ops.reshape(c, [array_ops.shape(init_c)[0], \
                          array_ops.shape(init_c)[1]])
    if go_backwards:
      outputs = array_ops.reverse_sequence_v2(
          outputs, sequence_lengths, seq_axis=0, batch_axis=1)
      outputs = array_ops.reverse(outputs, axis=[0])
  else:
    # # Fill the array with shape [batch] with value of max timesteps.
    # sequence_length = array_ops.fill([array_ops.shape(inputs)[1]],
    #                                  array_ops.shape(inputs)[0])
    if go_backwards:
      # Reverse axis 0 since the input is already convert to time major.
      inputs = array_ops.reverse(inputs, axis=[0])
    outputs, h, c, _ = load_ops_library.itex_rnn(
        input=inputs,
        input_h=init_h,
        input_c=init_c,
        params=params,
        dropout=dropout,
        dropout_mask=dp_mask,
        recurrent_dropout=recurrent_dropout,
        recurrent_dropout_mask=rec_dp_mask,
        sequence_lengths=0,
        rnn_mode='lstm',
        is_training=training)
    outputs = array_ops.reshape(outputs,
                                [array_ops.shape(inputs)[0], \
                                array_ops.shape(inputs)[1], \
                                array_ops.shape(init_h)[1]])
    h = array_ops.reshape(h, [array_ops.shape(init_h)[0], \
                          array_ops.shape(init_h)[1]])
    c = array_ops.reshape(c, [array_ops.shape(init_c)[0], \
                          array_ops.shape(init_c)[1]])

  last_output = outputs[-1]
  if not time_major:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])

  # In the case of variable length input, the ITEX kernel will fill zeros for
  # the output, whereas the default keras behavior is to bring over the previous
  # output for t-1, so that in the return_sequence=False case, user can quickly
  # get the final effect output instead just 0s at the last timestep.
  # In order to mimic the default keras behavior, we copy the final h state as
  # the last_output, since it is numerically same as the output.
  if mask is not None:
    last_output = h
  return last_output, outputs, h, c


def is_itex_supported_inputs(mask, time_major):
  if time_major:
    mask = array_ops.transpose(mask)

  return math_ops.logical_and(
      is_sequence_right_padded(mask),
      math_ops.logical_not(has_fully_masked_sequence(mask)))


def _read_variable_value(v):
  """Read the value of a variable if it is variable."""
  if isinstance(v, variables.Variable):
    return v.read_value()
  return v
