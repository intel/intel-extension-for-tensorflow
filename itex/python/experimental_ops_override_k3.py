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
import logging
import os
import types
import tensorflow as tf


from keras import ops


from intel_extension_for_tensorflow.python.ops.layer_norm_k3 import _layer_norm
from intel_extension_for_tensorflow.python.ops.group_norm_k3 import GroupNormalization
from intel_extension_for_tensorflow.python.ops.optimizers_k3 import Adam
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library


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
    self._data_format = "NHWC"  # pylint: disable=protected-access
    self._is_one_axis_len = None  # pylint: disable=protected-access
    can_use_onednn_layer_norm = True
    axis = sorted(self.axis)
    if axis[-1] != ndims - 1 or ndims < 2 or ndims > 4 or axis[-1] - axis[0] != len(axis) - 1:  # pylint: disable=line-too-long
        can_use_onednn_layer_norm = False

    if can_use_onednn_layer_norm and (axis[-1] == 3 or self.axis[-1] == -1):
        self.data_format = 'NHWC'

    if len(axis) == 1:
        self._is_one_axis_len = True  # pylint: disable=protected-access
    else:
        self._is_one_axis_len = False  # pylint: disable=protected-access

    if self.dtype == 'float64':
        raise ValueError(
            'Itex Layernorm only support float32, bfloat16 and float16.')  # pylint: disable=line-too-long

    return can_use_onednn_layer_norm


def experimental_ops_override():
    '''
    using itex api in some tf and keras functions.
    '''
    try:
        from pkg_resources import packaging  # pylint: disable=import-outside-toplevel
        version = packaging.version.parse
        if version(tf.__version__) < version("2.16.1"):
            return

        from keras.src.backend.tensorflow.core import convert_to_tensor
        from keras.src.backend import standardize_dtype
        from keras.src.backend.common import dtypes

        import keras
        tf_ln_call = copy_func(keras.layers.LayerNormalization.call)
        tf_bn_call = copy_func(keras.layers.BatchNormalization.call)
        tf_bn_build = copy_func(keras.layers.BatchNormalization.build)
        tf_gn_call = copy_func(keras.layers.GroupNormalization.call)
        tf_mean = copy_func(keras.src.backend.numpy.mean)

    except BaseException:  # pylint: disable=broad-except
        return

    def itex_layer_norm_build(self, input_shape):
        self.supports_jit = False
        if self.compute_dtype == "float16" or self.compute_dtype == "bfloat16":  # pylint: disable=no-else-return
            self._param_dtype = "float32"
        else:
            self._param_dtype = self.dtype or dtypes.float32
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
        self._use_layernorm = _can_use_onednn_layer_norm(self, ndims)
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

    def itex_layer_norm_call(self, inputs, training=None):
        if not self._use_layernorm:  # pylint: disable=protected-access
            return tf_ln_call(self, inputs)  # pylint: disable=not-callable
        if self.rms_scaling:  # pylint: disable=protected-access
            return tf_ln_call(self, inputs)  # pylint: disable=not-callable 
        is_training = True
        if training is None:
            is_training = True
        if isinstance(training, int):
            is_training = bool(training)
        elif isinstance(training, bool):
            is_training = training
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed from
            # model.
            is_training = False
        # Compute the axes along which to reduce the mean / variance
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

        beta = self.beta if self.beta is not None else self._beta_const
        gamma = self.gamma if self.gamma is not None else self._gamma_const
        if self._is_one_axis_len:
            outputs = _layer_norm_inference_or_training(self, inputs, gamma, beta,
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
            outputs = _layer_norm_inference_or_training(self, inputs, scale,
                                                                     offset, is_training)
            outputs = ops.reshape(outputs, tensor_shape)
            scale, offset = _broadcast(
                self.gamma), _broadcast(self.beta)

            if scale is not None:
                outputs = outputs * ops.cast(scale, outputs.dtype)
            if offset is not None:
                outputs = outputs + ops.cast(offset, outputs.dtype)
            return outputs

    def itex_batch_norm_build(self, input_shape):
        tf_bn_build(self, input_shape)
        rank = len(input_shape)
        if self.axis in (1, -3) and rank == 4:
            self.fused = True
            self._data_format = "NCHW"
        elif self.axis in (1, -4) and rank == 5:
            self.fused = True
            self._data_format = "NCDHW"
        elif self.axis in (-1, 3) and rank == 4:
            self.fused = True
            self._data_format = "NHWC"
        elif self.axis in (-1, 4) and rank == 5:
            self.fused = True
            self._data_format = "NDHWC"
        else:
            self.fused = False
        
        self.supports_jit = False
        self._param_shape = (input_shape[self.axis],)
        if self.compute_dtype == "float16" or self.compute_dtype == "bfloat16":  # pylint: disable=no-else-return
            self._param_dtype = "float32"
        else:
            self._param_dtype = self.dtype or dtypes.float32

    def itex_batch_norm_call(self, inputs, training=None, mask = None):
        if (not self.fused) or mask is not None:
            return tf_bn_call(self, inputs)

        inputs = tf.cast(inputs, self.compute_dtype)
        if self.center:
            beta = self.beta
        else:
            beta = ops.zeros(
                dtype=self._param_dtype, shape=self._param_shape
        )
        if self.scale:
            gamma = self.gamma
        else:
            gamma = ops.ones(
                dtype=self._param_dtype, shape=self._param_shape
        )
        def _fused_batch_norm_training():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=True,
                data_format=self._data_format,
                exponential_avg_factor=1-self.momentum,
                )

        def _fused_batch_norm_inference():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format,)


        output, mean, variance = tf.__internal__.smart_cond.smart_cond(
            bool(training) and self.trainable, _fused_batch_norm_training, _fused_batch_norm_inference
        )
        if bool(training) and self.trainable:
            self.moving_mean.assign(mean)
            self.moving_variance.assign(variance)
        return output

    def itex_group_norm_call(self,inputs):
        if self.use_fused_group_norm:
            shape = inputs.shape
            inputs = ops.cast(inputs, self.compute_dtype)
            normalized_inputs, _, _ = load_ops_library.itex_group_norm(
                inputs,
                ops.zeros((shape[-1],),self.compute_dtype) if self.gamma is None else self.gamma,
                ops.zeros((shape[-1],),self.compute_dtype) if self.beta is None else self.beta,
                num_groups=self.groups,
                epsilon=self.epsilon,
                use_scale=self.scale,
                use_center=self.center,
            )
            return normalized_inputs
        elif not self.use_gpu:
            normalized_inputs = GroupNormalization.itex_group_norm_call(self, inputs)
            return normalized_inputs
        else:
            return tf_gn_call(self, inputs)

    def itex_mean(x, axis=None, keepdims=False):
        if isinstance(x, tf.IndexedSlices):
            return tf_mean(x, axis, keepdims)
        x = convert_to_tensor(x)
        ori_dtype = standardize_dtype(x.dtype)
        compute_dtype = dtypes.result_type(x.dtype, "float32")
        if ori_dtype == "float16" or ori_dtype == "bfloat16":
            compute_dtype = ori_dtype
        if "int" in ori_dtype or ori_dtype == "bool":
            result_dtype = compute_dtype
        else:
            result_dtype = ori_dtype
        output = tf.reduce_mean(
            tf.cast(x, compute_dtype), axis=axis, keepdims=keepdims
        )
        return tf.cast(output, result_dtype)

    def itex_var(x, axis=None, keepdims=False):
        x = convert_to_tensor(x)
        ori_dtype = standardize_dtype(x.dtype)
        compute_dtype = dtypes.result_type(x.dtype, "float32")
        if ori_dtype == "float16" or ori_dtype == "bfloat16":
            compute_dtype = ori_dtype
        result_dtype = dtypes.result_type(x.dtype, float)
        x = tf.cast(x, compute_dtype)
        return tf.cast(
            tf.math.reduce_variance(x, axis=axis, keepdims=keepdims),
            result_dtype,
        )

    try:
        keras.layers.LayerNormalization.call = itex_layer_norm_call
        keras.layers.LayerNormalization.build = itex_layer_norm_build
        keras.layers.BatchNormalization.call = itex_batch_norm_call
        keras.layers.BatchNormalization.build = itex_batch_norm_build
        keras.layers.GroupNormalization.call = itex_group_norm_call
        keras.layers.GroupNormalization.build = GroupNormalization.build
        keras.optimizers.Adam.update_step = Adam.update_step
        
    except BaseException:  # pylint: disable=broad-except
        logger.error("Cannot override itex ops.")
    try:
        import keras  # pylint: disable=import-outside-toplevel
        keras.src.layers.normalization.layer_normalization.LayerNormalization.call = itex_layer_norm_call
        keras.src.layers.normalization.layer_normalization.LayerNormalization.build = itex_layer_norm_build
        keras.src.layers.normalization.batch_normalization.BatchNormalization.call = itex_batch_norm_call
        keras.src.layers.normalization.batch_normalization.BatchNormalization.build = itex_batch_norm_build
        keras.src.layers.normalization.group_normalization.GroupNormalization.call = itex_group_norm_call
        keras.src.layers.normalization.group_normalization.GroupNormalization.build = GroupNormalization.build
        keras.src.backend.numpy.mean = itex_mean
        keras.src.backend.numpy.var = itex_var
        keras.src.optimizers.adam.Adam.update_step = Adam.update_step
        logger.info("itex experimental ops override is enabled.")
    except BaseException:  # pylint: disable=broad-except
        logger.warning(
            "Cannot override itex ops.")  # pylint: disable=line-too-long
