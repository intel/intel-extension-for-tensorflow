# Copyright (c) 2023 Intel Corporation
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for itex recurrent layers."""

import os
import tempfile
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import intel_extension_for_tensorflow as itex
from intel_extension_for_tensorflow.python.test_func import test as test_lib
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from keras.regularizers import l2
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from keras import layers as keras_layers

from keras.layers import (
    Input,
    Dense,
    Activation,
)


def convert_model_weights(source_model, target_model):
    _, fname = tempfile.mkstemp(".h5")
    source_model.save_weights(fname)
    target_model.load_weights(fname)
    os.remove(fname)  

class FusedDenseBiasAddGeluTest(keras_parameterized.TestCase):
    def assert_allclose(self, expected, actual, dtype):
        if dtype in [
            tf.float32,
        ]:
            rtol = 1e-3
            atol = 1e-3
            np.testing.assert_allclose(expected, actual, rtol=rtol, atol=atol)
        elif dtype in [tf.bfloat16, tf.float16]:
            rtol = 1e-2
            atol = 1e-2
            np.testing.assert_allclose(
                tf.cast(expected, tf.float32),
                tf.cast(actual, tf.float32),
                rtol=rtol,
                atol=atol,
            )
        else:
            print("not supported data type")

    def gelu(self, x):
        return tf.nn.gelu(x, approximate=True)

    @parameterized.named_parameters(
        *testing_utils.generate_combinations_with_testcase_name(
            to_itex=[True, False], model_nest_level=[1, 2], dtype=[tf.float16]
        )
    )
    def test_load_weights_between_nonitex_rnn(self, to_itex, model_nest_level, dtype):
        if not test_lib.is_gpu_available():
            self.skipTest("Skip on CPU due to the pattern not supported")
        np.random.seed(0)
        units = 128
        activation = self.gelu
        m = 128
        k = 128
        input_shape = (m, k)
        inputs = np.random.random((m, k))

        layer = keras.layers.Dense(units, activation=activation, dtype=dtype)
        itex_layer = itex.ops.FusedDenseBiasAddGelu(units, dtype=dtype)

        model = self._make_nested_model(input_shape, layer, dtype, model_nest_level)
        itex_model = self._make_nested_model(
            input_shape, itex_layer, dtype, model_nest_level
        )

        if to_itex:
            convert_model_weights(model, itex_model)
        else:
            convert_model_weights(itex_model, model)

        x = tf.Variable(inputs, dtype=dtype)

        with tf.GradientTape(persistent=True) as tape1:
            outputs = model(x, training=True)
            loss = tf.reduce_sum(outputs * 2)

        with tf.GradientTape(persistent=True) as tape2:
            itex_outputs = itex_model(x, training=True)
            itex_loss = tf.reduce_sum(itex_outputs * 2)

        dx = tape1.gradient(loss, x)
        dwei = tape1.gradient(loss, model.trainable_variables)
        gradients = dict(dx=dx, dwei=dwei)

        dx_itex = tape2.gradient(itex_loss, x)
        dwei_itex = tape2.gradient(itex_loss, itex_model.trainable_variables)
        gradients_itex = dict(dx=dx_itex, dwei=dwei_itex)

        # double verify weight
        self.assert_allclose(
            model.trainable_variables[0], itex_model.trainable_variables[0], dtype
        )
        self.assert_allclose(
            model.trainable_variables[1], itex_model.trainable_variables[1], dtype
        )

        # verify forward result
        self.assert_allclose(outputs, itex_outputs, dtype)

        # verify backward result
        self.assert_allclose(gradients["dx"], gradients_itex["dx"], dtype)
        self.assert_allclose(gradients["dwei"][0], gradients_itex["dwei"][0], dtype)
        self.assert_allclose(gradients["dwei"][1], gradients_itex["dwei"][1], dtype)

    def _make_nested_model(self, input_shape, layer, dtype, level=1):
        # example: make_nested_seq_model((1,), Dense(10), level=2).summary()
        def make_nested_seq_model(input_shape, layer, dtype, level=1):
            model = layer
            for i in range(1, level + 1):
                layers = (
                    [tf.keras.layers.InputLayer(input_shape, dtype=dtype), model]
                    if (i == 1)
                    else [model]
                )
                model = tf.keras.models.Sequential(layers)
                if i > 1:
                    model.build((None,) + input_shape)
            return model

        return make_nested_seq_model(input_shape, layer, dtype, level)
        


if __name__ == "__main__":
    test.main()