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
"""ITEX optimization for keras layers."""
import builtins
import logging
import types
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat

from intel_extension_for_tensorflow.python.ops.layer_norm import _layer_norm
from intel_extension_for_tensorflow.python.ops.activations import gelu
from intel_extension_for_tensorflow.python.ops.recurrent import ItexLSTM

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

def itex_dense_layer_call(self, inputs):
  r"""ITEX optimized dense layer"""
  def tensordot(a, b, axes, name=None):
    r"""Tensor contraction of a and b along specified axes and outer product.
    Tensordot (also known as tensor contraction) sums the product of elements
    from `a` and `b` over the indices specified by `axes`.
    This operation corresponds to `numpy.tensordot(a, b, axes)`.
    Example 1: When `a` and `b` are matrices (order 2), the case `axes=1`
    is equivalent to matrix multiplication.
    Example 2: When `a` and `b` are matrices (order 2), the case
    `axes = [[1], [0]]` is equivalent to matrix multiplication.
    Example 3: When `a` and `b` are matrices (order 2), the case `axes=0` gives
    the outer product, a tensor of order 4.
    Example 4: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
    tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
    \\(c_{jklm}\\) whose entry
    corresponding to the indices \\((j,k,l,m)\\) is given by:
    \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
    In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
    Args:
    a: `Tensor` of type `float32` or `float64`.
    b: `Tensor` with the same type as `a`.
    axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
      If axes is a scalar, sum over the last N axes of a and the first N axes of
      b in order. If axes is a list or `Tensor` the first and second row contain
      the set of unique integers specifying axes along which the contraction is
      computed, for `a` and `b`, respectively. The number of axes for `a` and
      `b` must be equal. If `axes=0`, computes the outer product between `a` and
      `b`.
    name: A name for the operation (optional).
    Returns:
    A `Tensor` with the same type as `a`.
    Raises:
    ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
    IndexError: If the values in axes exceed the rank of the corresponding
        tensor.
    """

    def _tensordot_reshape(a, axes, flipped=False):
      """Helper method to perform transpose and reshape for contraction op.
      This method is helpful in reducing `math_ops.tensordot` to
      `math_ops.matmul` using `array_ops.transpose` and
      `array_ops.reshape`. The method takes a tensor and performs the
      correct transpose and reshape operation for a given set of indices.
      It returns the reshaped tensor as well as a list of indices
      necessary to reshape the tensor again after matrix multiplication.
      Args:
      a: `Tensor`.
      axes: List or `int32` `Tensor` of unique indices specifying valid axes of
          `a`.
      flipped: An optional `bool`. Defaults to `False`. If `True`, the method
          assumes that `a` is the second argument in the contraction operation.
      Returns:
      A tuple `(reshaped_a, free_dims, free_dims_static)` where `reshaped_a` is
      the tensor `a` reshaped to allow contraction via `matmul`, `free_dims` is
      either a list of integers or an `int32` `Tensor`, depending on whether
      the shape of a is fully specified, and free_dims_static is either a list
      of integers and None values, or None, representing the inferred
      static shape of the free dimensions
      """
      if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
        shape_a = a.get_shape().as_list()
        axes = [i if i >= 0 else i + len(shape_a) for i in axes]
        free = [i for i in builtins.range(len(shape_a)) if i not in axes]
        free_dims = [shape_a[i] for i in free]
        prod_free = int(np.prod([shape_a[i] for i in free]))
        prod_axes = int(np.prod([shape_a[i] for i in axes]))
        perm = list(axes) + free if flipped else free + list(axes)
        new_shape = [prod_axes, prod_free] if flipped else [
            prod_free, prod_axes]
        if (perm != np.arange(len(shape_a))).any():
          a_trans = array_ops.transpose(a, perm)
        else:
          a_trans = a
        if a_trans.get_shape().as_list() != new_shape:
          reshaped_a = array_ops.reshape(a_trans, new_shape)
        else:
          reshaped_a = a_trans
        return reshaped_a, free_dims, free_dims
      else:
        if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
          shape_a = a.get_shape().as_list()
          axes = [i if i >= 0 else i + len(shape_a) for i in axes]
          free = [i for i in builtins.range(len(shape_a)) if i not in axes]
          axes_dims = [shape_a[i] for i in axes]
          free_dims = [shape_a[i] for i in free]
          free_dims_static = free_dims
          axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name="axes")
          free = ops.convert_to_tensor(free, dtype=dtypes.int32, name="free")
          shape_a = array_ops.shape(a)
        else:
          free_dims_static = None
          shape_a = array_ops.shape(a)
          rank_a = array_ops.rank(a)
          axes = ops.convert_to_tensor(axes, dtype=dtypes.int32, name="axes")
          axes = array_ops.where(axes >= 0, axes, axes + rank_a)
          free, _ = gen_array_ops.list_diff(range(rank_a), axes, dtypes.int32)
        free_dims = array_ops.gather(shape_a, free)
        axes_dims = array_ops.gather(shape_a, axes)
        prod_free_dims = tf.math.reduce_prod(free_dims)
        prod_axes_dims = tf.math.reduce_prod(axes_dims)
        if flipped:
          perm = array_ops.concat([axes, free], 0)
          new_shape = array_ops.stack([prod_axes_dims, prod_free_dims])
        else:
          perm = array_ops.concat([free, axes], 0)
          new_shape = array_ops.stack([prod_free_dims, prod_axes_dims])
        reshaped_a = array_ops.reshape(array_ops.transpose(a, perm), new_shape)
        return reshaped_a, free_dims, free_dims_static

    def _tensordot_axes(a, axes):
      """Generates two sets of contraction axes for the two tensor arguments."""
      a_shape = a.get_shape()
      if isinstance(axes, compat.integral_types):
        if axes < 0:
          raise ValueError(f"`axes` must be at least 0. Received: {axes}.")
        if a_shape.ndims is not None:
          if axes > a_shape.ndims:
            raise ValueError(f"`axes` must not be larger than the number of "
                             f"dimensions of tensor {a}.  Received {axes}, vs "
                             f"tensor dimensions {a_shape.ndims}.")
          return (list(builtins.range(a_shape.ndims - axes,
                        a_shape.ndims)), list(builtins.range(axes)))
        else:
          rank = array_ops.rank(a)
          return (range(rank - axes, rank,
                  dtype=dtypes.int32), range(axes, dtype=dtypes.int32))
      elif isinstance(axes, (list, tuple)):
        if len(axes) != 2:
          raise ValueError(
              f"`axes` must be an integer or have length 2. Received {axes}.")
        a_axes = axes[0]
        b_axes = axes[1]
        if isinstance(a_axes, compat.integral_types) and \
                isinstance(b_axes, compat.integral_types):
          a_axes = [a_axes]
          b_axes = [b_axes]
        if len(a_axes) != len(b_axes):
          raise ValueError(f"Different number of contraction axes `a` and `b`, "
                           f"{len(a_axes)} != {len(b_axes)}.")
        return a_axes, b_axes
      else:
        axes = ops.convert_to_tensor(axes, name="axes", dtype=dtypes.int32)
        return axes[0], axes[1]

    with ops.name_scope(name, "Tensordot", [a, b, axes]) as name:
      a = ops.convert_to_tensor(a, name="a")
      b = ops.convert_to_tensor(b, name="b")
      a_axes, b_axes = _tensordot_axes(a, axes)
      a_reshape, a_free_dims, a_free_dims_static = _tensordot_reshape(a, a_axes)
      b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
          b, b_axes, True)
      ab_matmul = tf.matmul(a_reshape, b_reshape)
    if self.use_bias:
      ab_matmul = tf.nn.bias_add(ab_matmul, self.bias)

    with ops.name_scope(name, "Tensordot", [a, b, axes]) as name:
      if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
        if (ab_matmul.get_shape().is_fully_defined() and
                ab_matmul.get_shape().as_list() == a_free_dims + b_free_dims):
          return ab_matmul
        else:
          return array_ops.reshape(
              ab_matmul, a_free_dims + b_free_dims, name=name)
      else:
        a_free_dims = ops.convert_to_tensor(a_free_dims, dtype=dtypes.int32)
        b_free_dims = ops.convert_to_tensor(b_free_dims, dtype=dtypes.int32)
        product = array_ops.reshape(
          ab_matmul, array_ops.concat([a_free_dims, b_free_dims], 0), name=name)
        if a_free_dims_static is not None and b_free_dims_static is not None:
          product.set_shape(a_free_dims_static + b_free_dims_static)
        return product

  if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
    inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

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
    outputs = tensordot(inputs, self.kernel, [[rank - 1], [0]])
    # TODO(itex): We cannot do softmax/sigmoid before reshape in tensordot.
    # It will affect loss functions with logits. We should use BMM instead.
    if self.activation is not None:
      outputs = self.activation(outputs)
    # Reshape the output back to the original ndim of the input.
    if not tf.executing_eagerly():
      shape = inputs.shape.as_list()
      output_shape = shape[:-1] + [self.kernel.shape[-1]]
      outputs.set_shape(output_shape)

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
    from keras.utils import tf_utils
    from pkg_resources import packaging # pylint: disable=import-outside-toplevel
    version = packaging.version.parse
    if version(tf.__version__) < version("2.9.0"):
      return
    tf_ln_call = copy_func(tf.keras.layers.LayerNormalization.call)
  except BaseException: # pylint: disable=broad-except
    return
  def itex_layer_norm_build(self, input_shape):
    self._param_dtype = None
    # Raise parameters of fp16 layer tch norm to fp32
    if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16: # pylint: disable=no-else-return
      self._param_dtype = dtypes.float32
    else:
      self._param_dtype =  self.dtype or dtypes.float32
    self.axis = tf_utils.validate_axis(self.axis, input_shape)
    input_shape = tf.TensorShape(input_shape)
    param_shape = [input_shape[dim] for dim in self.axis]
    self._use_layernorm = _can_use_onednn_layer_norm(self, len(input_shape))
    if self.scale:
        self.gamma = self.add_weight(
            name="gamma",
            shape=param_shape,
            dtype=self._param_dtype if self._use_layernorm else None,
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
            dtype=self._param_dtype if self._use_layernorm else None,
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
      return tf_ln_call(self, inputs)
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

    beta = self.beta if self.beta is not None else self._beta_const
    gamma = self.gamma if self.gamma is not None else self._gamma_const
    if self._is_one_axis_len:
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
      beta = array_ops.reshape(self.beta, broadcast_shape)
      outputs = outputs + math_ops.cast(beta, outputs.dtype)

    return outputs


  try:
    import tensorflow_addons as tfa
    tfa.layers.InstanceNormalization.call = itex_instance_norm_call
  except BaseException: # pylint: disable=broad-except
    logger.warning("itex experimental ops override: tensorflow_addons is not installed.") # pylint: disable=line-too-long
  try:
    tf.keras.layers.Dense.call = itex_dense_layer_call
    tf.keras.layers.LayerNormalization.call = itex_layer_norm_call
    tf.keras.layers.LayerNormalization.build = itex_layer_norm_build
    tf.nn.gelu = gelu
    tf.keras.layers.LSTM = ItexLSTM
    from tensorflow.python import keras
    keras.layers.LSTM = ItexLSTM
    logger.info("itex experimental ops override is enabled.")
  except BaseException: # pylint: disable=broad-except
    logger.error("Cannot override itex ops.")
  try:
    import keras # pylint: disable=import-outside-toplevel
    keras.layers.core.dense.Dense.call = itex_dense_layer_call
    keras.layers.LayerNormalization.call = itex_layer_norm_call
    keras.layers.LayerNormalization.build = itex_layer_norm_build
    keras.layers.LSTM = ItexLSTM
  except BaseException: # pylint: disable=broad-except
    logger.warning("itex experimental ops override: Keras is not installed.") # pylint: disable=line-too-long

