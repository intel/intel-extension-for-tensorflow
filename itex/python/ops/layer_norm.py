# Copyright (c) 2021-2022 Intel Corporation
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
"""Layer Normalization layers."""
# pylint: disable=g-classes-have-attributes, missing-function-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import Layer
try:
  from keras.utils import control_flow_util
except ImportError:
  from keras.src.utils import control_flow_util

def _layer_norm(
    x,
    scale,
    offset,  # pylint: disable=invalid-name
    epsilon=0.001,
    is_training=True,
    data_format="NHWC"):
  x = ops.convert_to_tensor(x, name="input")
  scale = ops.convert_to_tensor(scale, name="scale")
  offset = ops.convert_to_tensor(offset, name="offset")

  # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
  # prevent exception (see cudnn.h).
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon

  y, running_mean, running_var = load_ops_library.itex_layer_norm(
      x,
      scale,
      offset,
      epsilon=epsilon,
      is_training=is_training,
      data_format=data_format)
  return y, running_mean, running_var

@keras.utils.generic_utils.register_keras_serializable(package="Itex")
class LayerNormalization(Layer):
  """Layer normalization layer (Ba et al., 2016).

  Normalize the activations of the previous layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within each
  example close to 0 and the activation standard deviation close to 1.

  Given a tensor `inputs`, moments are calculated and normalization
  is performed across the axes specified in `axis`.

  >>> import intel_extension_for_tensorflow as itex
  >>> data = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
  >>> print(data)
  tf.Tensor(
  [[ 0. 10.]
   [20. 30.]
   [40. 50.]
   [60. 70.]
   [80. 90.]], shape=(5, 2), dtype=float32)

  >>> layer = itex.ops.LayerNormalization(axis=1)
  >>> output = layer(data)
  >>> print(output)
  tf.Tensor(
  [[-0.99998  0.99998]
   [-0.99998  0.99998]
   [-0.99998  0.99998]
   [-0.99998  0.99998]
   [-0.99998  0.99998]], shape=(5, 2), dtype=float32)

  Notice that with Layer Normalization the normalization happens across the
  axes *within* each example, rather than across different examples in the
  batch.

  If `scale` or `center` are enabled, the layer will scale the normalized
  outputs by broadcasting them with a trainable variable `gamma`, and center
  the outputs by broadcasting with a trainable variable `beta`. `gamma` will
  default to a ones tensor and `beta` will default to a zeros tensor, so that
  centering and scaling are no-ops before training has begun.
  Notice that with Layer Normalization the normalization happens across the
  axes *within* each example, rather than across different examples in the
  batch.

  If `scale` or `center` are enabled, the layer will scale the normalized
  outputs by broadcasting them with a trainable variable `gamma`, and center
  the outputs by broadcasting with a trainable variable `beta`. `gamma` will
  default to a ones tensor and `beta` will default to a zeros tensor, so that
  centering and scaling are no-ops before training has begun.

  So, with scaling and centering enabled the normalization equations
  are as follows:

  Let the intermediate activations for a mini-batch to be the `inputs`.

  For each sample `x_i` in `inputs` with `k` features, we compute the mean and
  variance of the sample:

  ```python
  mean_i = sum(x_i[j] for j in range(k)) / k
  var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k
  ```

  and then compute a normalized `x_i_normalized`, including a small factor
  `epsilon` for numerical stability.

  ```python
  x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
  ```

  And finally `x_i_normalized ` is linearly transformed by `gamma` and `beta`,
  which are learned parameters:

  ```python
  output_i = x_i_normalized * gamma + beta
  ```

  `gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
  this part of the inputs' shape must be fully defined.

  For example:

  >>> layer = itex.ops.LayerNormalization(axis=[1, 2, 3])
  >>> layer.build([5, 20, 30, 40])
  >>> print(layer.beta.shape)
  (20, 30, 40)
  >>> print(layer.gamma.shape)
  (20, 30, 40)

  If you want to run inference, set training to False, By default, when you call
  itex.ops.LayerNormalization, it will use training mode.

  For example:
  >>> import intel_extension_for_tensorflow as itex
  >>> data = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
  >>> layer = itex.ops.LayerNormalization(axis=1)
  >>> output = layer(data, training=False)
  >>> print(output)
  tf.Tensor(
  [[-0.99998  0.99998]
   [-0.99998  0.99998]
   [-0.99998  0.99998]
   [-0.99998  0.99998]
   [-0.99998  0.99998]], shape=(5, 2), dtype=float32)

  Note that other implementations of layer normalization may choose to define
  `gamma` and `beta` over a separate set of axes from the axes being
  normalized across. For example, Group Normalization
  ([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
  corresponds to a Layer Normalization that normalizes across height, width,
  and channel and has `gamma` and `beta` span only the channel dimension.
  So, this Layer Normalization implementation will not match a Group
  Normalization layer with group size set to 1.

  Arguments:
    axis: Integer or List/Tuple. The axis or axes to normalize across. Typically
      this is the features axis/axes. The left-out axes are typically the batch
      axis/axes. This argument defaults to `-1`, the last dimension in the
      input.
    epsilon: Small float added to variance to avoid dividing by zero. Defaults
      to 1e-3
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored. Defaults to True.
    scale: If True, multiply by `gamma`. If False, `gamma` is not used. Defaults
      to True. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling will be done by the next layer.
    beta_initializer: Initializer for the beta weight. Defaults to zeros.
    gamma_initializer: Initializer for the gamma weight. Defaults to ones.
    beta_regularizer: Optional regularizer for the beta weight. None by default.
    gamma_regularizer: Optional regularizer for the gamma weight. None by
      default.
    beta_constraint: Optional constraint for the beta weight. None by default.
    gamma_constraint: Optional constraint for the gamma weight. None by default.
    trainable: Boolean, if `True` the variables will be marked as trainable.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape` (tuple of
    integers, does not include the samples axis) when using this layer as the
    first layer in a model.

  Output shape:
    Same shape as input.

  Reference:
    - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).
  """

  def __init__(self,
               axis=-1,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               trainable=True,
               **kwargs):
    super(LayerNormalization, self).__init__(**kwargs)
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('Expected an int or a list/tuple of ints for the '
                      'argument \'axis\', but received: %r' % axis)

    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)
    self.trainable = trainable

    self.supports_masking = True

    # Indicates whether a faster fused implementation can be used. This will be
    # set to True or False in build()"
    self._use_layernorm = None

  def _can_use_onednn_layer_norm(self, ndims):
    """Return false if Itex layernorm implementation cannot be used.

    Check if the axis is contiguous and can be collapsed into the last axis.
    The self.axis is assumed to have no duplicates.
    """
    self._data_format = "NHWC"
    self._is_one_axis_len = None
    can_use_onednn_layer_norm = True
    axis = sorted(self.axis)
    if axis[-1] != ndims - 1 or ndims < 2 or ndims > 4 or axis[-1] - axis[0] != len(axis) - 1: # pylint: disable=line-too-long
      can_use_onednn_layer_norm = False

    if can_use_onednn_layer_norm and (axis[-1] == 3 or self.axis[-1] == -1):
      self.data_format = 'NHWC'

    if len(axis) == 1:
      self._is_one_axis_len = True
    else:
      self._is_one_axis_len = False

    if self.dtype == 'float64':
      raise ValueError('Itex Layernorm only support float32, \
                        bfloat16 and float16.')

    return can_use_onednn_layer_norm

  @property
  def _param_dtype(self):
    # Raise parameters of fp16 layer tch norm to fp32
    if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16: # pylint: disable=no-else-return
      return dtypes.float32
    else:
      return self.dtype or dtypes.float32

  def build(self, input_shape):
    ndims = len(input_shape)
    if ndims is None:
      raise ValueError('Input shape %s has undefined rank.' % input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]
    elif isinstance(self.axis, tuple):
      self.axis = list(self.axis)
    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))

    param_shape = [input_shape[dim] for dim in self.axis]

    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.gamma = None
      self._gamma_const = K.constant(
          1.0, dtype=self._param_dtype, shape=param_shape)

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True,
          experimental_autocast=False)
    else:
      self.beta = None
      self._beta_const = K.constant(
          0.0, dtype=self._param_dtype, shape=param_shape)

    self._use_layernorm = self._can_use_onednn_layer_norm(ndims)

    self.built = True

  def _layer_norm_inference_or_training(self, inputs, gamma, beta, training):
    """Returns the output of layer norm."""

    def _layer_norm_training():
      return _layer_norm(
          inputs,
          scale=gamma,
          offset=beta,
          epsilon=self.epsilon,
          is_training=True,
          data_format=self._data_format)

    def _layer_norm_inference():
      return _layer_norm(
          inputs,
          scale=gamma,
          offset=beta,
          epsilon=self.epsilon,
          is_training=False,
          data_format=self._data_format)

    output, _, _ = control_flow_util.smart_cond(
        training, _layer_norm_training, _layer_norm_inference)
    return output

  def _get_training_value(self, training=None):
    if training is None:
      training = True
    if isinstance(training, int):
      training = bool(training)
    if not self.trainable:
      # When the layer is not trainable, it overrides the value passed from
      # model.
      training = False
    return training

  def call(self, inputs, training=None): # pylint: disable=arguments-differ
    is_training = self._get_training_value(training)
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

    if not self._use_layernorm:
      input_dtype = inputs.dtype
      if input_dtype in ('float16', 'bfloat16'):
        # If mixed precision is used, cast inputs to float32 so that this is at
        # least as numerically stable as the fused version.
        inputs = math_ops.cast(inputs, 'float32')

      # Calculate the moments on the last axis (layer activations).
      mean, variance = nn.moments(inputs, self.axis, keep_dims=True)

      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

      # Compute layer normalization using the batch_normalization function.
      outputs = nn.batch_normalization(
          inputs,
          mean,
          variance,
          offset=offset,
          scale=scale,
          variance_epsilon=self.epsilon)
      outputs = math_ops.cast(outputs, input_dtype)
    else:
      beta = self.beta if self.beta is not None else self._beta_const
      gamma = self.gamma if self.gamma is not None else self._gamma_const
      if self._is_one_axis_len:
        outputs = self._layer_norm_inference_or_training(inputs, gamma, beta,
                                                         is_training)
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
        outputs = self._layer_norm_inference_or_training(inputs, scale,
                                                         offset, is_training)
        outputs = array_ops.reshape(outputs, tensor_shape)
        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        if scale is not None:
          outputs = outputs * math_ops.cast(scale, outputs.dtype)
        if offset is not None:
          outputs = outputs + math_ops.cast(offset, outputs.dtype)

    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'axis': self.axis,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint)
    }
    base_config = super(LayerNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
