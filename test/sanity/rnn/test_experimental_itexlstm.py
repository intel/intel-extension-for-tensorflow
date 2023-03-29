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
import numpy as np
import tensorflow as tf
import intel_extension_for_tensorflow as itex
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

itex.experimental_ops_override()

class LSTMTest(test_util.TensorFlowTestCase):          

    def test(self):
        if not config.list_logical_devices('XPU'):
            self.skipTest("Test requires XPU")
        assert(tf.keras.layers.LSTM == itex.ops.ItexLSTM)
        
        from tensorflow.python import keras
        assert(keras.layers.LSTM == itex.ops.ItexLSTM)

        import keras
        assert(keras.layers.LSTM == itex.ops.ItexLSTM)
        
        batch_input_shape = (2, 2, 2)    
        x_input = keras.layers.Input(batch_shape=batch_input_shape)
        x_in = x_input
        x_out = keras.layers.LSTM(20)(x_in, training=False)
        model = keras.Model(inputs=x_input, outputs=x_out)
        layers = model.layers
        assert(layers[1].name == "itex_lstm")
        x = np.random.random(batch_input_shape)
        y = model.predict_on_batch(x)
        print(y.shape)

     

if __name__ == '__main__':
    test.main()
