# Copyright (c) 2023 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
"""ITEX optimization for some TensorFlow API."""
import builtins
import logging
import types
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util import compat

from intel_extension_for_tensorflow.python.device import get_backend
from intel_extension_for_tensorflow.python.ops.layer_norm import _layer_norm
from intel_extension_for_tensorflow.python.ops.activations import gelu as itex_gelu
from intel_extension_for_tensorflow.python.ops.recurrent import gpu_lstm
from intel_extension_for_tensorflow.python.ops.recurrent import is_itex_supported_inputs

format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger(__name__)

def copy_func(f, name=None):
  '''
  return a function with same code, globals, defaults, closure, and
  name (or provide a new name)
  '''
  fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
       f.__defaults__, f.__closure__)
  # in case f was given attrs (note this dict is a shallow copy):
  fn.__dict__.update(f.__dict__)
  return fn

@tf.custom_gradient
def itex_optimized_bmm(x, w, b):
  result = tf.raw_ops.BatchMatMulV2(x=x, y=w)
  result = tf.nn.bias_add(result, b)
  def grad(upstream):
    '''
    # This is the gradient using BMM
    dx = tf.raw_ops.BatchMatMulV2(x=upstream, y=w, adj_y=True)
    dy = tf.raw_ops.BatchMatMulV2(x=x, y=upstream, adj_x=True)
    dy = tf.math.reduce_sum(dy,axis=0)
    db = tf.raw_ops.BiasAddGrad(out_backprop=upstream)
    '''
    upstream_shape = tf.shape(upstream)
    # upstream rank must > 3
    batch_size = tf.math.reduce_prod(upstream_shape[:-1])
    upstream_2d = tf.reshape(upstream,[batch_size,-1])
    x_2d = tf.reshape(x,[batch_size,-1])
    dx = tf.raw_ops.BatchMatMulV2(x=upstream, y=w, adj_y=True)
    dy = tf.matmul(x_2d,upstream_2d,transpose_a=True)
    db = tf.raw_ops.BiasAddGrad(out_backprop=upstream_2d)
    return dx, dy, db
  return result, grad

def itex_dense_layer_call(self, inputs, training=None):
  r"""ITEX optimized dense layer"""
  if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype: # pylint: disable=protected-access
    inputs = tf.cast(inputs, dtype=self._compute_dtype_object) # pylint: disable=protected-access

  is_ragged = isinstance(inputs, tf.RaggedTensor)
  if is_ragged:
    # In case we encounter a RaggedTensor with a fixed last dimension
    # (last dimension not ragged), we can flatten the input and restore
    # the ragged dimensions at the end.
    if tf.compat.dimension_value(inputs.shape[-1]) is None:
      raise ValueError(
          "Dense layer only supports RaggedTensors when the "
          "innermost dimension is non-ragged. Received: "
          f"inputs.shape={inputs.shape}."
      )
    original_inputs = inputs
    if inputs.flat_values.shape.rank > 1:
      inputs = inputs.flat_values
    else:
      # Innermost partition is encoded using uniform_row_length.
      # (This is unusual, but we can handle it.)
      if inputs.shape.rank == 2:
        inputs = inputs.to_tensor()
        is_ragged = False
      else:
        for _ in range(original_inputs.ragged_rank - 1):
          inputs = inputs.values
        inputs = inputs.to_tensor()
        original_inputs = tf.RaggedTensor.from_nested_row_splits(
            inputs, original_inputs.nested_row_splits[:-1]
        )

  rank = inputs.shape.rank
  if rank == 2 or rank is None:
    # We use embedding_lookup_sparse as a more efficient matmul
    # operation for large sparse input tensors. The op will result in a
    # sparse gradient, as opposed to
    # sparse_ops.sparse_tensor_dense_matmul which results in dense
    # gradients. This can lead to sigfinicant speedups, see b/171762937.
    if isinstance(inputs, tf.SparseTensor):
      # We need to fill empty rows, as the op assumes at least one id
      # per row.
      inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
      # We need to do some munging of our input to use the embedding
      # lookup as a matrix multiply. We split our input matrix into
      # separate ids and weights tensors. The values of the ids tensor
      # should be the column indices of our input matrix and the
      # values of the weights tensor can continue to the actual matrix
      # weights.  The column arrangement of ids and weights will be
      # summed over and does not matter. See the documentation for
      # sparse_ops.sparse_tensor_dense_matmul a more detailed
      # explanation of the inputs to both ops.
      ids = tf.SparseTensor(
          indices=inputs.indices,
          values=inputs.indices[:, 1],
          dense_shape=inputs.dense_shape,
      )
      weights = inputs
      outputs = tf.nn.embedding_lookup_sparse(
          self.kernel, ids, weights, combiner="sum"
      )
    else:
      outputs = tf.matmul(a=inputs, b=self.kernel)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)
    if is_ragged:
      outputs = original_inputs.with_flat_values(outputs)
  # Broadcast kernel to inputs.
  else:
    if training == True and self.use_bias:
      outputs = itex_optimized_bmm(inputs, self.kernel, self.bias)
    else:
      outputs = tf.raw_ops.BatchMatMulV2(x=inputs, y=self.kernel)
      if self.use_bias:
        outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)

    if is_ragged:
      outputs = original_inputs.with_flat_values(outputs)

  return outputs

def _can_use_onednn_layer_norm(self, ndims):
  """Return false if Itex layernorm implementation cannot be used.

  Check if the axis is contiguous and can be collapsed into the last axis.
  The self.axis is assumed to have no duplicates.
  """
  self._data_format = "NHWC" # pylint: disable=protected-access
  self._is_one_axis_len = None # pylint: disable=protected-access
  can_use_onednn_layer_norm = True
  axis = sorted(self.axis)
  if axis[-1] != ndims - 1 or ndims < 2 or ndims > 4 or axis[-1] - axis[0] != len(axis) - 1: # pylint: disable=line-too-long
    can_use_onednn_layer_norm = False

  if can_use_onednn_layer_norm and (axis[-1] == 3 or self.axis[-1] == -1):
    self.data_format = 'NHWC'

  if len(axis) == 1:
    self._is_one_axis_len = True # pylint: disable=protected-access
  else:
    self._is_one_axis_len = False # pylint: disable=protected-access

  if self.dtype == 'float64':
    raise ValueError('Itex Layernorm only support float32, bfloat16 and float16.') # pylint: disable=line-too-long

  return can_use_onednn_layer_norm

def experimental_ops_override():
  '''
  using itex api in some tf and keras functions.
  '''
  try:
    from pkg_resources import packaging # pylint: disable=import-outside-toplevel
    version = packaging.version.parse
    if version(tf.__version__) < version("2.9.0"):
      return
    if version(tf.__version__).release >= version("2.13").release:
        # New versions of Keras require importing from `keras.src` when
        # importing internal symbols.
        from keras.src import backend # pylint: disable=import-outside-toplevel
        from keras.src.utils import tf_utils # pylint: disable=import-outside-toplevel
    else:
        from keras import backend # pylint: disable=import-outside-toplevel
        from keras.utils import tf_utils # pylint: disable=import-outside-toplevel
    tf_ln_call = copy_func(tf.keras.layers.LayerNormalization.call)
    tf_lstm_call = copy_func(tf.keras.layers.LSTM.call)
    tf_lstm_build = copy_func(tf.keras.layers.LSTM.build)

  except BaseException: # pylint: disable=broad-except
    return
  def itex_layer_norm_build(self, input_shape):
    self._param_dtype = None # pylint: disable=protected-access
    # Raise parameters of fp16 layer tch norm to fp32
    if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16: # pylint: disable=no-else-return,consider-using-in
      self._param_dtype = dtypes.float32 # pylint: disable=protected-access
    else:
      self._param_dtype =  self.dtype or dtypes.float32 # pylint: disable=protected-access
    self.axis = tf_utils.validate_axis(self.axis, input_shape)
    input_shape = tf.TensorShape(input_shape)
    param_shape = [input_shape[dim] for dim in self.axis]
    self._use_layernorm = _can_use_onednn_layer_norm(self, len(input_shape)) # pylint: disable=protected-access
    if self.scale:
      self.gamma = self.add_weight(
          name="gamma",
          shape=param_shape,
          dtype=self._param_dtype if self._use_layernorm else None, # pylint: disable=protected-access
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True,
          experimental_autocast=False,
      )
    else:
      self.gamma = None

    if self.center:
      self.beta = self.add_weight(
          name="beta",
          shape=param_shape,
          dtype=self._param_dtype if self._use_layernorm else None, # pylint: disable=protected-access
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True,
          experimental_autocast=False,
      )
    else:
      self.beta = None

    self.built = True

  def itex_layer_norm_call(self, inputs):
    if not self._use_layernorm: # pylint: disable=protected-access
      return tf_ln_call(self, inputs) # pylint: disable=not-callable
    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)

    # Broadcasting only necessary for norm when the axis is not just
    # the last dimension
    broadcast_shape = [1] * ndims
    for dim in self.axis:
      broadcast_shape[dim] = input_shape.dims[dim].value

    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and self.axis != [ndims - 1]):
        return array_ops.reshape(v, broadcast_shape)
      return v

    beta = self.beta if self.beta is not None else self._beta_const # pylint: disable=protected-access
    gamma = self.gamma if self.gamma is not None else self._gamma_const # pylint: disable=protected-access
    if self._is_one_axis_len: # pylint: disable=protected-access
      outputs, _, _ = _layer_norm(
        inputs,
        scale=gamma,
        offset=beta,
        epsilon=self.epsilon,
        is_training=True,
        data_format="NHWC")
    else:
      # Collapse dims before self.axis, and dims in self.axis
      pre_dim, in_dim = (1, 1)
      axis = sorted(self.axis)
      tensor_shape = array_ops.shape(inputs)
      for dim in range(0, ndims):
        dim_tensor = tensor_shape[dim]
        if dim < axis[0]:
          pre_dim = pre_dim * dim_tensor
        else:
          assert dim in axis
          in_dim = in_dim * dim_tensor

      squeezed_shape = [1, pre_dim, in_dim]
      inputs = array_ops.reshape(inputs, squeezed_shape)

      # self.gamma and self.beta have the wrong shape for layer_norm, so
      # we cannot pass them as the scale and offset parameters. Therefore, we
      # create two constant tensors in correct shapes for layer_norm and
      # later construct a separate calculation on the scale and offset.
      scale = array_ops.ones([in_dim], dtype=dtypes.float32)
      offset = array_ops.zeros([in_dim], dtype=dtypes.float32)

      # Compute layer normalization.
      outputs, _, _ = _layer_norm(inputs, scale, offset, self.epsilon, True, "NHWC") # pylint: disable=line-too-long
      outputs = array_ops.reshape(outputs, tensor_shape)
      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

      if scale is not None:
        outputs = outputs * math_ops.cast(scale, outputs.dtype)
      if offset is not None:
        outputs = outputs + math_ops.cast(offset, outputs.dtype)

    return outputs

  def itex_instance_norm_call(self, inputs):
    """ITEX version 'call' function for InstanceNormalization in tensorflow_addons package.

    This is implemented by itex_layer_norm.
    The InstanceNormalization takes the axis as input,
    and the size of gamma and beta is the size of the specified axis.
    """
    input_shape = array_ops.shape(inputs)
    ndims = len(input_shape)
    axis = self.axis
    if axis < 0:
      axis = ndims + axis
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis]

    if axis != 1:
      # Because itex_layer_norm computes mean and variance across the last axis,
      # But the InstanceNorm in tensorflow_addons pkg computes them on axes
      # except for the first and the specified axis, So for convenience transpose
      # the specified axis to the axis at subscript 1 position, and collapse subsequent
      # axes as the last axis.
      perm_shape = list(range(0, ndims))
      perm_shape.pop(axis)
      perm_shape.insert(1, axis)
      inputs = array_ops.transpose(inputs, perm_shape)

    # Collapse dims after 1.
    in_dim = 1
    tensor_shape = array_ops.shape(inputs)
    for dim in range(0, ndims):
      dim_tensor = tensor_shape[dim]
      if dim > 1:
        in_dim = in_dim * dim_tensor

    squeezed_shape = [tensor_shape[0], tensor_shape[1], in_dim]
    inputs = array_ops.reshape(inputs, squeezed_shape)

    # self.gamma and self.beta have the wrong shape for layer_norm, so
    # we cannot pass them as the scale and offset parameters. Therefore, we
    # create two constant tensors in correct shapes for layer_norm and
    # later construct a separate calculation on the scale and offset.
    scale = array_ops.ones([in_dim], dtype=dtypes.float32)
    offset = array_ops.zeros([in_dim], dtype=dtypes.float32)

    outputs, _, _ = _layer_norm(inputs,
                                scale=scale,
                                offset=offset,
                                epsilon=self.epsilon,
                                is_training=True)
    outputs = array_ops.reshape(outputs, tensor_shape)
    if axis != 1:
      perm_back_shape = list(range(0, ndims))
      perm_back_shape.pop(1)
      perm_back_shape.insert(axis, 1)
      outputs = array_ops.transpose(outputs, perm_back_shape)

    if self.gamma is not None:
      gamma = array_ops.reshape(self.gamma, broadcast_shape)
      outputs = outputs * math_ops.cast(gamma, outputs.dtype)
    if self.beta is not None:
      if axis == ndims - 1:
        # Use biasadd to avoid Sum in bwd process to improve perf.
        outputs = tf.nn.bias_add(outputs, math_ops.cast(self.beta, outputs.dtype))
      else:
        beta = array_ops.reshape(self.beta, broadcast_shape)
        outputs = outputs + math_ops.cast(beta, outputs.dtype)

    return outputs

  def itex_lstm_build(self, input_shape):
    tf_lstm_build(self, input_shape)
    self._could_use_itex_kernel = (
        self.activation in (tf.keras.activations.tanh, tf.nn.tanh) and
        self.recurrent_activation in (tf.keras.activations.sigmoid, tf.nn.sigmoid) and
        self.use_bias) and (config.list_logical_devices('XPU'))
    # TODO: use ITEX get_backend to check GPU.
    if config.list_logical_devices('XPU'):
      # Only show the message when there is GPU available, itex LSTM only support GPU currently
      if self._could_use_itex_kernel:
        logging.debug('Layer %s will use ITEX kernels when running on GPU.' % self.name)
      else:
        logging.warning('Layer %s will not use ITEX kernels since it '
                        'doesn\'t meet the criteria. It will '
                        'use a generic GPU kernel as fallback when running '
                        'on GPU.' % self.name)

  def itex_lstm_call(self, inputs, mask=None, training=None, initial_state=None):
    # if is ragged tensor or on CPU, fall back
    if (not self._could_use_itex_kernel):
      return tf_lstm_call(self, inputs, mask, training, initial_state)
    if (isinstance(inputs, tf.RaggedTensor)):
      return tf_lstm_call(self, inputs, mask, training, initial_state)
    # when mask is not None:
    if mask is not None:
      if isinstance(mask, list):
        mask = mask[0]
      if not is_itex_supported_inputs(mask, self.time_major):
        return tf_lstm_call(self, inputs, mask, training, initial_state)

    inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    inputs, initial_state, _ = self._process_inputs(
      inputs, initial_state, None)
    self._maybe_reset_cell_dropout_mask(self.cell)
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
    last_output, outputs, new_h, new_c = gpu_lstm(**gpu_lstm_kwargs)
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

  try:
    import tensorflow_addons as tfa # pylint: disable=import-outside-toplevel
    tfa.layers.InstanceNormalization.call = itex_instance_norm_call
  except BaseException: # pylint: disable=broad-except
    logger.warning("itex experimental ops override: tensorflow_addons is not installed.") # pylint: disable=line-too-long
  try:
    tf.keras.layers.Dense.call = itex_dense_layer_call
    tf.keras.layers.LayerNormalization.call = itex_layer_norm_call
    tf.keras.layers.LayerNormalization.build = itex_layer_norm_build

    from tensorflow.nn import gelu # pylint: disable=import-outside-toplevel
    gelu = itex_gelu
    tf.nn.gelu = itex_gelu
    tf.keras.layers.LSTM.call = itex_lstm_call
    tf.keras.layers.LSTM.build = itex_lstm_build
    logger.info("itex experimental ops override is enabled.")
  except BaseException: # pylint: disable=broad-except
    logger.error("Cannot override itex ops.")
  try:
    import keras # pylint: disable=import-outside-toplevel
    if version(tf.__version__).release >= version("2.13").release:
      keras.src.layers.core.dense.Dense.call = itex_dense_layer_call
      keras.src.layers.normalization.layer_normalization.LayerNormalization.call = itex_layer_norm_call
      keras.src.layers.normalization.layer_normalization.LayerNormalization.build = itex_layer_norm_build
      keras.src.layers.rnn.lstm.LSTM.call = itex_lstm_call
      keras.src.layers.rnn.lstm.LSTM.build = itex_lstm_build
    else:
      keras.layers.core.dense.Dense.call = itex_dense_layer_call
      keras.layers.LayerNormalization.call = itex_layer_norm_call
      keras.layers.LayerNormalization.build = itex_layer_norm_build
      keras.layers.LSTM.call = itex_lstm_call
      keras.layers.LSTM.build = itex_lstm_build
  
  except BaseException: # pylint: disable=broad-except
    logger.warning("itex experimental ops override: Keras is not installed.") # pylint: disable=line-too-long
