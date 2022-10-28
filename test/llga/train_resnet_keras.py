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
                                    Add, Flatten, Dense, MaxPool2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

np.random.seed(34)
tf.random.set_seed(34)

def bn_relu(inputs: Tensor) -> Tensor:
    bn = BatchNormalization()(inputs)
    relu = ReLU()(bn)
    return relu

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               use_bias=False,
               padding="same",
               kernel_regularizer=regularizers.l2(0.001))(x)
    y = bn_relu(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               use_bias=False,
               padding="same",
               kernel_regularizer=regularizers.l2(0.001))(y)
    y = BatchNormalization()(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   use_bias=False,
                   padding="same",
                   kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
    out = Add()([x, y])
    out = ReLU()(out)
    return out

def create_res_net():
    inputs = Input(shape=(64, 64, 3))
    
    t = MaxPool2D()(inputs)

    num_filters = 3
    outputs = residual_block(t, downsample=False, filters=num_filters)
    
    model = Model(inputs, outputs)

    return model


model = create_res_net()

model.summary() 

model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )

training_data = np.random.normal(0, 1, size=(2, 64, 64, 3)).astype("float32")
label_data = np.random.normal(0, 1, size=(2, 32, 32, 3)).astype("float32")

batch_size = 1
history = model.fit(training_data, label_data, batch_size=batch_size, epochs=3)
