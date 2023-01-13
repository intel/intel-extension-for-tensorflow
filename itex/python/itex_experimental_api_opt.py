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
import logging
import types
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from intel_extension_for_tensorflow.python.ops.layer_norm import _layer_norm

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

def itex_experimental_api_opt():
  '''
  using itex api in some tf and keras functions.
  '''
  try:
    from pkg_resources import packaging # pylint: disable=import-outside-toplevel
    version = packaging.version.parse
    if version(tf.__version__) < version("2.9.0"):
      return
    tf_ln_call = copy_func(tf.keras.layers.LayerNormalization.call)
    tf_ln_build = copy_func(tf.keras.layers.LayerNormalization.build)
  except BaseException: # pylint: disable=broad-except
    return
  def itex_layer_norm_build(self, input_shape):
    tf_ln_build(self, input_shape)
    self._use_layernorm = _can_use_onednn_layer_norm(self, len(input_shape))

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

  try:
    tf.keras.layers.LayerNormalization.call = itex_layer_norm_call
    tf.keras.layers.LayerNormalization.build = itex_layer_norm_build
    logger.info("itex experimental api optimization enabled.")
  except BaseException: # pylint: disable=broad-except
    logger.error("Cannot do optimization for itex experimental api.")
  try:
    import keras # pylint: disable=import-outside-toplevel
    keras.layers.LayerNormalization.call = itex_layer_norm_call
    keras.layers.LayerNormalization.build = itex_layer_norm_build
  except BaseException: # pylint: disable=broad-except
    logger.warning("itex experimental api optimization: Keras is not installed.") # pylint: disable=line-too-long
