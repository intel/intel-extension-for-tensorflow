# Copyright (c) 2022 Intel Corporation
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
import os
os.environ['TF_USE_LEGACY_KERAS']='1'
os.environ['ITEX_ENABLE_NEXTPLUGGABLE_DEVICE']='0'
import tempfile
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import intel_extension_for_tensorflow as itex
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils


from tensorflow.keras.layers import (
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

class LSTMTrainingTest(test.TestCase):
    """Tests for itex recurrent layers."""
  
    def test_train_on_batch(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        batch_size = 128
        length = 128
        num_signals = 14
        units = 200
        def build_model(rnn_layer):
            regularization = 0.01
            model_kwargs = dict(
                            return_sequences=True,
                            stateful=True,
                            kernel_regularizer=l2(regularization),
                            recurrent_regularizer=l2(regularization),
                            bias_regularizer=l2(regularization),
                            dropout = 0,
                            recurrent_dropout = 0,
                            )      
            batch_input_shape = (batch_size, length, num_signals)    
            x_input = Input(batch_shape=batch_input_shape)
            x_in = x_input

            x_out = rnn_layer(units, **model_kwargs)(x_in, training=True)
            model = tf.keras.Model(inputs=x_input, outputs=x_out)
            model.reset_states()
            return model
                  
        itex_model = build_model(itex.ops.ItexLSTM)
        keras_model = build_model(LSTM)
        convert_model_weights(itex_model, keras_model)
        
        keras_model.compile(gradient_descent.GradientDescentOptimizer(0.001), 'mse')
        itex_model.compile(gradient_descent.GradientDescentOptimizer(0.001), 'mse')        

        x = np.random.random((batch_size, length, num_signals))
        y = np.random.random((batch_size, length, units)) 
        
        keras_loss = keras_model.train_on_batch(x, y)       
        itex_loss = itex_model.train_on_batch(x, y)   
        np.isclose(keras_loss, itex_loss, rtol=1e-3, atol=1e-3, equal_nan=False)
     

if __name__ == '__main__':
    test.main()

