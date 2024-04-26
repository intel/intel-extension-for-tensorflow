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

import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
import numpy as np
import tensorflow as tf
import intel_extension_for_tensorflow as itex
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tf_keras.src.backend import set_session

itex.experimental_ops_override()

class LSTMTest(test_util.TensorFlowTestCase):          

    def test(self):
        if not config.list_logical_devices('XPU'):
            self.skipTest("Test requires XPU")
        from tensorflow.keras.layers import LSTM
        with self.session(use_gpu=True) as sess:
            set_session(sess)
            batch_input_shape = (2, 2, 2)
            x_input = tf.keras.Input(shape=(2,2,))
            x_out = LSTM(20)(x_input, training=False)
            model = tf.keras.Model(inputs=x_input, outputs=x_out)
            x = np.random.random(batch_input_shape).astype(np.float32)
            y = model.predict(x)
            assert "itex_lstm_call" == model.layers[1].call.__func__.__name__
            print(y.shape)

     

if __name__ == '__main__':
    test.main()
