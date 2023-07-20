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



import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, Flatten, Dense, MaxPool2D, AveragePooling2D,\
                                    Permute, Activation
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

import intel_extension_for_tensorflow as itex

import os

np.random.seed(34)
tf.random.set_seed(34)

os.environ["ITEX_ONEDNN_GRAPH"] = "1"
os.environ["_ITEX_ONEDNN_GRAPH_ALL_TYPE"] = "1"

def create_model():
    inputs = Input(shape=(8, 8, 3))
    
    t = AveragePooling2D()(inputs)
    t = Permute((3, 1, 2))(t)

    t = tf.reshape(t, [-1, 3, 16])
    t = Dense(64)(t)
    t = Activation(activations.relu)(t)
    t = Dense(64)(t)
    t = Activation(activations.elu)(t)
    t = Dense(64)(t)
    t = Activation(activations.tanh)(t)
    t = itex.ops.LayerNormalization()(t)
    t = tf.keras.activations.softmax(t)

    model = Model(inputs, t)

    return model

model = create_model()

model.summary() 

model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )

training_data = np.random.normal(0, 1, size=(2, 8, 8, 3)).astype("float32")
label_data = np.random.normal(0, 1, size=(2, 3, 64)).astype("float32")

batch_size = 1
history = model.fit(training_data, label_data, batch_size=batch_size, epochs=3)
