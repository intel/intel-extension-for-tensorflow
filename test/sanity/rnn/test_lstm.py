# Copyright (c) 2022 Intel Corporation
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
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from keras.regularizers import l2
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils


from keras.layers import (
    LSTM,
    Input,
    Dense, Activation, Dropout, 
    TimeDistributed,
    )

def convert_model_weights(source_model, target_model):
    _, fname = tempfile.mkstemp('.h5')
    source_model.save_weights(fname)
    target_model.load_weights(fname)
    os.remove(fname)

class LSTMTest(keras_parameterized.TestCase):           
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
            np.testing.assert_allclose(tf.cast(expected, tf.float32),
                                       tf.cast(actual, tf.float32),
                                       rtol=rtol,
                                       atol=atol)
        else:
            print("not supported data type")
            
    @parameterized.named_parameters(
        *testing_utils.generate_combinations_with_testcase_name(
            to_itex=[True, False],
            model_nest_level=[1, 2],
            model_type=['seq'],
            dtype=[tf.float32, tf.float16, tf.bfloat16]))
    def test_load_weights_between_nonitex_rnn(self, to_itex, model_nest_level,
                                              model_type, dtype):
        np.random.seed(0)
        input_size = 3
        timesteps = 4
        input_shape = (timesteps, input_size)
        units = 2
        batch_size = 3
        inputs = np.random.random((batch_size, timesteps, input_size))

        rnn_layer_kwargs = dict(
            # return_sequences=True,
            # return_state=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            stateful=True,
            dropout=0,
            recurrent_dropout=0,
            kernel_regularizer=l2(0.001),
            recurrent_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
            time_major=True,
        )

        layer = keras.layers.LSTM(units, **rnn_layer_kwargs)
        itex_layer = itex.ops.ItexLSTM(units, **rnn_layer_kwargs)

        model = self._make_nested_model(input_shape, layer, model_nest_level,
                                        model_type)
        itex_model = self._make_nested_model(input_shape, itex_layer,
                                             model_nest_level, model_type)
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
        self.assert_allclose(model.trainable_variables[0],
                             itex_model.trainable_variables[0], dtype)
        self.assert_allclose(model.trainable_variables[1],
                             itex_model.trainable_variables[1], dtype)
        self.assert_allclose(model.trainable_variables[2],
                             itex_model.trainable_variables[2], dtype)

        # verify forward result
        self.assert_allclose(outputs, itex_outputs, dtype)

        # verify backward result
        self.assert_allclose(gradients["dx"], gradients_itex["dx"], dtype)
        self.assert_allclose(gradients["dwei"][0], gradients_itex["dwei"][0],
                             dtype)
        self.assert_allclose(gradients["dwei"][1], gradients_itex["dwei"][1],
                             dtype)
        self.assert_allclose(gradients["dwei"][2], gradients_itex["dwei"][2],
                             dtype)

    def _make_nested_model(self,
                           input_shape,
                           layer,
                           level=1,
                           model_type='func'):
        # example: make_nested_seq_model((1,), Dense(10), level=2).summary()
        def make_nested_seq_model(input_shape, layer, level=1):
            model = layer
            for i in range(1, level + 1):
                layers = [tf.keras.layers.InputLayer(input_shape), model
                          ] if (i == 1) else [model]
                model = tf.keras.models.Sequential(layers)
                if i > 1:
                    model.build((None, ) + input_shape)
            return model

        # example: make_nested_func_model((1,), Dense(10), level=2).summary()
        def make_nested_func_model(input_shape, layer, level=1):
            model_input = tf.keras.layers.Input(input_shape)
            model = layer
            for _ in range(level):
                model = tf.keras.models.Model(model_input, model(model_input))
            return model

        if model_type == 'func':
            return make_nested_func_model(input_shape, layer, level)
        elif model_type == 'seq':
            return make_nested_seq_model(input_shape, layer, level)

    def test_stacked_lstm_and_model_save(self):
        regularization = 0.01
        model_kwargs = dict(
                        return_sequences=True,
                        # return_state=True,
                        # stateful=True,
                        kernel_regularizer=l2(regularization),
                        recurrent_regularizer=l2(regularization),
                        bias_regularizer=l2(regularization),
                        dropout = 0.1,
                        recurrent_dropout = 0.1,
                        unroll=True,
                        )      
        batch_size = 3
        length = 2
        num_signals = 2
        units = 2
        batch_input_shape = (batch_size, length, num_signals)    
        x_input = tf.keras.layers.Input(batch_shape=batch_input_shape)
        x_in = x_input
       
        for i in range(2):
            x_in = itex.ops.ItexLSTM(units, **model_kwargs)(x_in, training=True)
            x_in = Dropout(0.1)(x_in)
        x_out = TimeDistributed(
                        Dense(1, activation="tanh"))(x_in)
        model = tf.keras.Model(inputs=x_input, outputs=x_out)
        model.compile(gradient_descent.GradientDescentOptimizer(0.001), 'mse')
        x = np.random.random((batch_size, length, num_signals))
        y = np.random.random((batch_size, length, units))
        loss = model.train_on_batch(x, y)        
        model.save("/tmp/",
                    overwrite=True,  # default
                    include_optimizer=True,  # default
                    save_format=None,  # default, 'h5' in r1.15. Else 'tf'
                    signatures=None,  # applicable to 'tf' SavedModel format only
                    )

     

if __name__ == '__main__':
    test.main()
