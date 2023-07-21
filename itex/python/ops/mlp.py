# /* Copyright (c) 2023 Intel Corporation

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/

"""Contains the Dense layer."""

import tensorflow as tf
from tensorflow.python.framework import config

from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from intel_extension_for_tensorflow.python.device import is_xehpc
from intel_extension_for_tensorflow.python.ops.activations import gelu
from tensorflow.python import keras
from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import Dense
import functools
# isort: off


@keras.utils.generic_utils.register_keras_serializable(package="Itex")
class FusedDenseBiasAddGelu(tf.compat.v1.layers.Dense):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). These are all attributes of
    `Dense`.

    Note: If the input to the layer has a rank greater than 2, then `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors).  The output in this case will have
    shape `(batch_size, d0, units)`.

    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    When a popular kwarg `input_shape` is passed, then keras will create
    an input layer to insert before the current layer. This can be treated
    equivalent to explicitly defining an `InputLayer`.

    Example:

    >>> # Create a `Sequential` model and add a Dense layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(tf.keras.layers.Dense(32))
    >>> model.output_shape
    (None, 32)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(FusedDenseBiasAddGelu, self).__init__(
            units,
            activation=functools.partial(gelu, approximate=True),
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs
        )
        self._could_use_fused_matmul_biasadd_gelu = is_xehpc()

    def standard_dense(self, inputs):
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
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def call(self, inputs, training=None):
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
        self._could_use_fused_matmul_biasadd_gelu = (
            self._could_use_fused_matmul_biasadd_gelu
            and (not isinstance(inputs, tf.SparseTensor))
            and (
                self._compute_dtype_object == tf.bfloat16
                or self._compute_dtype_object == tf.float16
            )
        )
        if self._could_use_fused_matmul_biasadd_gelu:
            k = inputs.shape[-1]
            inputs = tf.reshape(inputs, [-1, k])            
            outputs, _ = load_ops_library.fused_dense_bias_add_gelu(
                input=inputs, weights=self.kernel, bias=self.bias, is_training=training
            )
        else:
            outputs = self.standard_dense(inputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the input shape of a Dense layer "
                "should be defined. Found None. "
                f"Received: input_shape={input_shape}"
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        base_config = super(Dense, self).get_config()
        derive_config = {
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        return dict(list(base_config.items()) + list(derive_config.items()))
