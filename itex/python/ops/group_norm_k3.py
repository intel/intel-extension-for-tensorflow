# Copyright 2023 The Keras Authors. All Rights Reserved.
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
"""Group normalization layer"""

import tensorflow as tf
from keras import constraints
from keras import initializers
from keras import ops
from keras import regularizers
from keras.src.saving import object_registration
from keras.src.layers.input_spec import InputSpec
from keras import Layer

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from intel_extension_for_tensorflow.python.ops.layer_norm_k3 import _layer_norm


@object_registration.register_keras_serializable(package="Itex")
class GroupNormalization(Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes nearly
    identical to Layer Normalization (see Layer Normalization docs for details).

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization. You can achieve this via `groups=-1`.

    Args:
        groups: Integer, the number of groups for Group Normalization. Can be in
            the range `[1, N]` where N is the input dimension. The input
            dimension must be divisible by the number of groups.
            Defaults to 32.
        axis: Integer or List/Tuple. The axis or axes to normalize across.
            Typically, this is the features axis/axes. The left-out axes are
            typically the batch axis/axes. -1 is the last dimension in the
            input. Defaults to `-1`.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-3.
        center: If `True`, add offset of `beta` to normalized tensor.
            If `False`, `beta` is ignored. Defaults to `True`.
        scale: If `True`, multiply by `gamma`. If `False`, `gamma` is not used.
            When the next layer is linear (also e.g. `relu`), this can be
            disabled since the scaling will be done by the next layer.
            Defaults to `True`.
        beta_initializer: Initializer for the beta weight. Defaults to zeros.
        gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        beta_regularizer: Optional regularizer for the beta weight. None by
            default.
        gamma_regularizer: Optional regularizer for the gamma weight. None by
            default.
        beta_constraint: Optional constraint for the beta weight.
            None by default.
        gamma_constraint: Optional constraint for the gamma weight. None by
            default.  Input shape: Arbitrary. Use the keyword argument
            `input_shape` (tuple of integers, does not include the samples
            axis) when using this layer as the first layer in a model.
            Output shape: Same shape as input.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).

    Reference:

    - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
        self,
        groups=32,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_jit = False
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        # Indicates whether a faster fused implementation can be used. This will be
        # set to True or False in build()"        
        self.use_fused_group_norm = None
        self.use_gpu = None

    def build(self, input_shape):
        self.use_gpu = len(tf.config.list_physical_devices("XPU")) > 0
        dim = input_shape[self.axis]

        rank = len(input_shape)
        self.axis = (self.axis + rank) % rank
        
        # fused_group_norm only support GPU backend and NHWC and axis=-1 currently
        # TODO(itex): support channel first and rank==any
        self.use_fused_group_norm = self.use_gpu and (rank == 4) and (self.axis == rank - 1)

        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}."
            )

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim})."
            )

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim})."
            )

        self.input_spec = InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs, training=False):
        if self.use_fused_group_norm and training == False:
            inputs = ops.cast(inputs, self.compute_dtype)
            normalized_inputs = load_ops_library.itex_group_norm(
                inputs,
                [0.0] if self.gamma is None else self.gamma,
                [0.0] if self.beta is None else self.beta,
                num_groups=self.groups,
                epsilon=self.epsilon,
                use_scale=self.scale,
                use_center=self.center,
            )
            return normalized_inputs
        else:
            normalized_inputs = self.itex_group_norm_call(inputs)
            return normalized_inputs
        # fall back path
        # reshaped_inputs = self._reshape_into_groups(inputs)
        # normalized_inputs = self._apply_normalization(
        #     reshaped_inputs, inputs.shape
        # )
        # return ops.reshape(normalized_inputs, ops.shape(inputs))

    def itex_group_norm_call(self, inputs):
        """
        This is implemented by itex_layer_norm.
        The GroupNormalization takes the axis as input,
        and the size of gamma and beta is the size of the specified axis.
        """
        input_shape = inputs.shape
        reshaped_inputs = self._reshape_into_groups(inputs)
        group_shape = reshaped_inputs.shape
        ndims = len(input_shape)
        group_ndims = len(group_shape)
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
            perm_shape = list(range(0, group_ndims))
            perm_shape.pop(axis)
            perm_shape.insert(1, axis)
            reshaped_inputs = ops.transpose(reshaped_inputs, perm_shape)

        # Collapse dims after 1.
        in_dim = 1
        tensor_shape = reshaped_inputs.shape
        for dim in range(0, group_ndims):
            dim_tensor = tensor_shape[dim]
            if dim > 1:
                in_dim = in_dim * dim_tensor

        squeezed_shape = [tensor_shape[0], tensor_shape[1], in_dim]
        inputs = ops.reshape(reshaped_inputs, squeezed_shape)

        # self.gamma and self.beta have the wrong shape for layer_norm, so
        # we cannot pass them as the scale and offset parameters. Therefore, we
        # create two constant tensors in correct shapes for layer_norm and
        # later construct a separate calculation on the scale and offset.
        scale = ops.ones([in_dim], dtype="float32")
        offset = ops.zeros([in_dim], dtype="float32")

        outputs, _, _ = _layer_norm(inputs,
                                    scale=scale,
                                    offset=offset,
                                    epsilon=self.epsilon,
                                    is_training=True)
        outputs = ops.reshape(outputs, tensor_shape)
        if axis != 1:
            perm_back_shape = list(range(0, group_ndims))
            perm_back_shape.pop(1)
            perm_back_shape.insert(axis, 1)
            outputs = ops.transpose(outputs, perm_back_shape)

        outputs = ops.reshape(outputs, input_shape)

        if self.scale:
            gamma = ops.reshape(self.gamma, broadcast_shape)
            outputs = outputs * ops.cast(gamma, outputs.dtype)
        if self.center:
            if axis == ndims - 1:
                # Use biasadd to avoid Sum in bwd process to improve perf.
                outputs = tf.nn.bias_add(outputs, ops.cast(self.beta, outputs.dtype))
            else:
                beta = ops.reshape(self.beta, broadcast_shape)
                outputs = outputs + ops.cast(beta, outputs.dtype)

        return outputs

    def _reshape_into_groups(self, inputs):
        input_shape = ops.shape(inputs)
        group_shape = list(inputs.shape)
        group_shape[0] = -1
        for i, e in enumerate(group_shape[1:]):
            if e is None:
                group_shape[i + 1] = input_shape[i + 1]

        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        reshaped_inputs = ops.reshape(inputs, group_shape)
        return reshaped_inputs

    # def _apply_normalization(self, reshaped_inputs, input_shape):
    #     group_reduction_axes = list(range(1, len(reshaped_inputs.shape)))

    #     axis = -2 if self.axis == -1 else self.axis - 1
    #     group_reduction_axes.pop(axis)

    #     broadcast_shape = self._create_broadcast_shape(input_shape)
    #     mean, variance = ops.moments(
    #         reshaped_inputs, axes=group_reduction_axes, keepdims=True
    #     )

    #     # Compute the batch normalization.
    #     inv = ops.rsqrt(variance + self.epsilon)
    #     if self.scale:
    #         gamma = ops.reshape(self.gamma, broadcast_shape)
    #         gamma = ops.cast(gamma, reshaped_inputs.dtype)
    #         inv = inv * gamma

    #     res = -mean * inv
    #     if self.center:
    #         beta = ops.reshape(self.beta, broadcast_shape)
    #         beta = ops.cast(beta, reshaped_inputs.dtype)
    #         res = res + beta

    #     normalized_inputs = reshaped_inputs * inv + res
    #     return normalized_inputs

    # def _create_broadcast_shape(self, input_shape):
    #     broadcast_shape = [1] * len(input_shape)
    #     broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
    #     broadcast_shape.insert(self.axis, self.groups)
    #     return broadcast_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
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
