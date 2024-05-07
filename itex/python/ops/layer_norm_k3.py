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
from keras import constraints
from keras import initializers
from keras import ops
from keras import regularizers
from keras.layers import Layer
from keras.src.saving import object_registration

import tensorflow as tf


def _layer_norm(
        x,
        scale,
        offset,  # pylint: disable=invalid-name
        epsilon=0.001,
        is_training=True,
        data_format="NHWC"):
    x = ops.convert_to_tensor(x)
    scale = ops.convert_to_tensor(scale)
    offset = ops.convert_to_tensor(offset)

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


@object_registration.register_keras_serializable(package="Itex")
class LayerNormalization(Layer):
    """Layer normalization layer (Ba et al., 2016).

    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.

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

    >>> layer = keras.layers.LayerNormalization(axis=[1, 2, 3])
    >>> layer.build([5, 20, 30, 40])
    >>> print(layer.beta.shape)
    (20, 30, 40)
    >>> print(layer.gamma.shape)
    (20, 30, 40)

    Note that other implementations of layer normalization may choose to define
    `gamma` and `beta` over a separate set of axes from the axes being
    normalized across. For example, Group Normalization
    ([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
    corresponds to a Layer Normalization that normalizes across height, width,
    and channel and has `gamma` and `beta` span only the channel dimension.
    So, this Layer Normalization implementation will not match a Group
    Normalization layer with group size set to 1.

    Args:
        axis: Integer or List/Tuple. The axis or axes to normalize across.
            Typically, this is the features axis/axes. The left-out axes are
            typically the batch axis/axes. `-1` is the last dimension in the
            input. Defaults to `-1`.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-3.
        center: If True, add offset of `beta` to normalized tensor. If False,
            `beta` is ignored. Defaults to `True`.
        scale: If True, multiply by `gamma`. If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`), this can be
            disabled since the scaling will be done by the next layer.
            Defaults to `True`.
        rms_scaling: If True, `center` and `scale` are ignored, and the
            inputs are scaled by `gamma` and the inverse square root
            of the square of all inputs. This is an approximate and faster
            approach that avoids ever computing the mean of the input.
        beta_initializer: Initializer for the beta weight. Defaults to zeros.
        gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        beta_regularizer: Optional regularizer for the beta weight.
            None by default.
        gamma_regularizer: Optional regularizer for the gamma weight.
            None by default.
        beta_constraint: Optional constraint for the beta weight.
            None by default.
        gamma_constraint: Optional constraint for the gamma weight.
            None by default.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).


    Reference:

    - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).
    """

    def __init__(
        self,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        rms_scaling=False,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_jit = False
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.rms_scaling = rms_scaling
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.supports_masking = True
        self.autocast = False

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
        if axis[-1] != ndims - 1 or ndims < 2 or ndims > 4 or axis[-1] - axis[0] != len(axis) - 1:  # pylint: disable=line-too-long
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
        if self.compute_dtype == "float16" or self.compute_dtype == "bfloat16":  # pylint: disable=no-else-return
            return "float32"
        else:
            return self.dtype or dtypes.float32

    def build(self, input_shape):
        ndims = len(input_shape)
        if ndims is None:
            raise ValueError(
                'Input shape %s has undefined rank.' % input_shape)
        if isinstance(self.axis, list):
            shape = tuple([input_shape[dim] for dim in self.axis])
        else:
            shape = (input_shape[self.axis],)
            self.axis = [self.axis]
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x
        param_shape = [input_shape[dim] for dim in self.axis]
        if self.scale or self.rms_scaling:
            self.gamma = self.add_weight(
                name="gamma",
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                dtype=self._param_dtype,
            )
        else:
            self.gamma = None
            self._gamma_const = ops.ones(
                dtype=self._param_dtype, shape=param_shape)

        if self.center and not self.rms_scaling:
            self.beta = self.add_weight(
                name="beta",
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                dtype=self._param_dtype,
            )
        else:
            self.beta = None
            self._beta_const = ops.zeros(
                dtype=self._param_dtype, shape=param_shape)
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

        output, _, _ = tf.__internal__.smart_cond.smart_cond(
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
        return training

    def call(self, inputs, training=None):
        is_training = self._get_training_value(training)
        inputs = ops.cast(inputs, self.compute_dtype)
        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)

        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape[dim]

        def _broadcast(v):
            if (
                v is not None
                and len(v.shape) != ndims
                and self.axis != [ndims - 1]
            ):
                return ops.reshape(v, broadcast_shape)
            return v

        input_dtype = inputs.dtype
        if input_dtype in (tf.float16, tf.bfloat16) and self.dtype == "float32" and not self._use_layernorm:
            # If mixed precision is used, cast inputs to float32 so that
            # this is at least as numerically stable as the fused version.
            inputs = ops.cast(inputs, "float32")

        if self.rms_scaling:
            # Calculate outputs with only variance and gamma if rms scaling
            # is enabled
            # Calculate the variance along self.axis (layer activations).
            variance = ops.var(inputs, axis=self.axis, keepdims=True)
            inv = ops.rsqrt(variance + self.epsilon)

            outputs = inputs * inv * ops.cast(self.gamma, inputs.dtype)
        else:
            if self._use_layernorm:
                beta = self.beta if self.beta is not None else self._beta_const
                gamma = self.gamma if self.gamma is not None else self._gamma_const
                if self._is_one_axis_len:
                    outputs = self._layer_norm_inference_or_training(inputs, gamma, beta,
                                                                     is_training)
                    return outputs
                else:
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

                    squeezed_shape = [1, pre_dim, in_dim]
                    inputs = ops.reshape(inputs, squeezed_shape)

                    # self.gamma and self.beta have the wrong shape for layer_norm, so
                    # we cannot pass them as the scale and offset parameters. Therefore, we
                    # create two constant tensors in correct shapes for layer_norm and
                    # later construct a separate calculation on the scale and offset.
                    scale = ops.ones([in_dim], dtype="float32")
                    offset = ops.zeros([in_dim], dtype="float32")

                    # Compute layer normalization.
                    outputs = self._layer_norm_inference_or_training(inputs, scale,
                                                                     offset, is_training)
                    outputs = ops.reshape(outputs, tensor_shape)
                    scale, offset = _broadcast(
                        self.gamma), _broadcast(self.beta)

                    if scale is not None:
                        outputs = outputs * ops.cast(scale, outputs.dtype)
                    if offset is not None:
                        outputs = outputs + ops.cast(offset, outputs.dtype)
                    return outputs

            # Calculate the mean & variance along self.axis (layer activations).
            mean, variance = ops.moments(inputs, axes=self.axis, keepdims=True)
            gamma, beta = _broadcast(self.gamma), _broadcast(self.beta)

            inv = ops.rsqrt(variance + self.epsilon)
            if gamma is not None:
                gamma = ops.cast(gamma, inputs.dtype)
                inv = inv * gamma

            res = -mean * inv
            if beta is not None:
                beta = ops.cast(beta, inputs.dtype)
                res = res + beta

            outputs = inputs * inv + res

        return ops.cast(outputs, input_dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
