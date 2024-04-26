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
"""RMS Normalization layers."""
# pylint: disable=g-classes-have-attributes, missing-function-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from keras import constraints
from keras import initializers
from keras import ops
from keras import regularizers
from keras.layers import Layer
from keras.src.saving import object_registration


@object_registration.register_keras_serializable(package="Itex")
class RMSNormalization(Layer):
  """Root Mean Square Layer Normalization (B.Zhang et al., 2019).

  Given a tensor `inputs`, normalization is performed across the axes specified 
  in `axis`. Notice that with RMSNormalization the normalization happens across 
  the axes *within* each example, rather than across different examples in the
  batch. Different from LayerNorm, RMSNorm regularizes the summed inputs to a 
  neuron in layer according to root mean square (RMS), and thus more 
  computationally simpler and efficient than LayerNorm.

  If `scale` or `center` are enabled, the layer will scale the normalized
  outputs by broadcasting them with a trainable variable `gamma`, and center
  the outputs by broadcasting with a trainable variable `beta`. `gamma` will
  default to a ones tensor and `beta` will default to a zeros tensor, so that
  centering and scaling are no-ops before training has begun.

  For each sample `x_i` in `inputs` with `k` features, we compute the mean
  square of the sample:

  ```python
  ms = sqrt(sum(x_i[j] ** 2 for j in range(k)) / k)
  ```

  and then compute a normalized `x_i_normalized`, including a small factor
  `epsilon` for numerical stability.

  ```python
  rms = 1 / sqrt(ms + epsilon)
  x_i_normalized = x_i * rms
  ```

  And finally `x_i_normalized ` is linearly transformed by `gamma` and `beta`,
  which are learned parameters:

  ```python
  output_i = x_i_normalized * gamma + beta
  ```

  `gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
  this part of the inputs' shape must be fully defined.

  For example:

  >>> layer = itex.ops.RMSNormalization(axis=[1, 2, 3])
  >>> layer.build([5, 20, 30, 40])
  >>> print(layer.beta.shape)
  (20, 30, 40)
  >>> print(layer.gamma.shape)
  (20, 30, 40)

  If you want to run inference, set training to False, By default, when you call
  itex.ops.RMSNormalization, it will use inference mode.

  For example:
  >>> import intel_extension_for_tensorflow as itex
  >>> data = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
  >>> layer = itex.ops.RMSNormalization()
  >>> output = layer(data, training=False)
  [[0.         1.4141995 ]
   [0.78446394 1.176696  ]
   [0.883452   1.104315  ]
   [0.9203579  1.0737509 ]
   [0.93955225 1.0569963 ]], shape=(5, 2), dtype=float32)


  Arguments:
    axis: Integer or List/Tuple. The axis or axes to normalize across. Typically
      this is the features axis/axes. RMSNorm only support to norm last continous
      n dimension. This argument defaults to `-1`, the last dimension in the
      input.
    epsilon: Small float added to variance to avoid dividing by zero. Defaults
      to 1e-3
    scale: If True, multiply by `gamma`. If False, `gamma` is not used. Defaults
      to True. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling will be done by the next layer.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored. Defaults to False.
    gamma_initializer: Initializer for the gamma weight. Defaults to ones.
    gamma_regularizer: Optional regularizer for the gamma weight. None by default.
    gamma_constraint: Optional constraint for the gamma weight. None by default.
    beta_initializer: Initializer for the beta weight. Defaults to zeros.
    beta_regularizer: Optional regularizer for the beta weight. None by default.
    beta_constraint: Optional constraint for the beta weight. None by default.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape` (tuple of
    integers, does not include the samples axis) when using this layer as the
    first layer in a model.

  Output shape:
    Same shape as input.

  Reference:
    - [B.Zhang et al., 2019](https://openreview.net/pdf?id=SygkZ3MTJE).
  """

  def __init__(self,
               axis=-1,
               epsilon=1e-3,
               scale=True,
               center=False,
               gamma_initializer='ones',
               gamma_regularizer=None,
               gamma_constraint=None,
               beta_initializer='zeros',
               beta_regularizer=None,
               beta_constraint=None,
               **kwargs):
    super(RMSNormalization, self).__init__(**kwargs)
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('Expected an int or a list/tuple of ints for the '
                      'argument \'axis\', but received: %r' % axis)

    self.epsilon = epsilon
    self.scale = scale
    self.center = center
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.gamma_constraint = constraints.get(gamma_constraint)
    self.beta_initializer = initializers.get(beta_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)

    self.supports_masking = True
    # Indicates whether a faster fused implementation can be used. This will be
    # set to True or False in build()"        
    self.use_fused_rms_norm = False
    
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
    axis = sorted(self.axis)
    if axis[0] != ndims - len(axis) or axis[-1] != ndims - 1:
      raise ValueError('ITEX RMSNorm uses last n axis of input tensor, '
                       'but got axis: {}'.format(",".join(str(n) for n in axis)))

    # Reduce axis'shape into one dim
    elem_count = 1
    for dim in self.axis:
      elem_count = elem_count * input_shape[dim]
    param_shape = [elem_count]

    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=True)
    else:
      self.gamma = None
      self._gamma_const = ops.ones(
          dtype=self._param_dtype, shape=param_shape)

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=True)
    else:
      self.beta = None
      self._beta_const = ops.zeros(
          dtype=self._param_dtype, shape=param_shape)

    # fused_rms_norm only support XPU backend currently
    self.use_fused_rms_norm = len(tf.config.list_physical_devices("XPU")) > 0
    self.built = True

  def call(self, inputs, training=False): # pylint: disable=arguments-differ
    input_shape = inputs.shape
    ndims = len(input_shape)
    beta = self.beta if self.beta is not None else self._beta_const
    gamma = self.gamma if self.gamma is not None else self._gamma_const

    # Collapse dims before self.axis, and dims in self.axis
    pre_dim, in_dim = (1, 1)
    axis = sorted(self.axis)
    tensor_shape = inputs.shape
    for dim in range(0, ndims):
        dim_tensor = tensor_shape[dim]
        if dim < axis[0]:
            pre_dim = pre_dim * dim_tensor
        else:
            assert dim in axis
            in_dim = in_dim * dim_tensor

    squeezed_shape = [pre_dim, in_dim]
    inputs = ops.reshape(inputs, squeezed_shape)
    # Compute RMS normalization.
    if self.use_fused_rms_norm and not training:
        # fused kernel only support inference on xpu.
        outputs = load_ops_library.itex_rms_norm(
                                    inputs,
                                    gamma,
                                    beta,
                                    epsilon=self.epsilon,
                                    use_scale=self.scale,
                                    use_center=self.center)
    else:
        # use several math kernels to emulate RMSNorm
        ms = ops.mean(inputs ** 2, axis=-1, keepdims=True)
        rms = ops.rsqrt(ms + self.epsilon)
        outputs = inputs * rms
        if self.scale:
          outputs = outputs * gamma
        if self.center:
          outputs = outputs + beta
  
    outputs = ops.reshape(outputs, tensor_shape)
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'axis': self.axis,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'gamma_constraint': constraints.serialize(self.gamma_constraint),
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint)
    }
    base_config = super(RMSNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
