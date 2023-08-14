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
# =============================================================================

import tensorflow.compat.v2 as tf
import numpy as np
from tensorflow import keras
from keras.initializers import Constant
try:
  from keras.testing_infra import test_combinations
  from keras.testing_infra import test_utils
except ImportError:
  from keras.src.testing_infra import test_combinations
  from keras.src.testing_infra import test_utils
import intel_extension_for_tensorflow as itex
from intel_extension_for_tensorflow.python.ops import RMSNormalization

def _build_rms_normalization_model(norm):
    model = keras.models.Sequential()
    model.add(norm)
    model.compile(
        loss="mse",
        optimizer="rmsprop",
        run_eagerly=test_utils.should_run_eagerly(),
    )

    return model


@test_utils.run_v2_only
class RMSNormalizationTest(test_combinations.TestCase):

    def test_trainable_weights(self):
        rows = 100
        cols = 100
        # Check if weights get initialized correctly
        layer = RMSNormalization(scale=False, center=False)
        layer.build((rows, cols))
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

        # Check if weights get initialized correctly
        layer = RMSNormalization(scale=True, center=True)
        layer.build((rows, cols))
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.weights), 2)
        self.assertEqual(len(layer.weights[0].shape), 1)
        self.assertEqual(len(layer.weights[1].shape), 1)
        self.assertEqual(layer.weights[0].shape[0], cols)
        self.assertEqual(layer.weights[1].shape[0], cols)

    @test_combinations.run_all_keras_modes
    def test_correctness_1d(self):
        layer = RMSNormalization(scale=False, center=False)
        input = tf.constant(
            [-1.0, -1.0, 1.0, 1.0, 2.0, 2.0, -2.0, -2.0], 
            shape=(1, 8)
        )
        expected_output = tf.constant(
            [-0.6323, -0.6323, 0.6323, 0.6323, 1.2646, 1.2646, -1.2646, -1.2646],
            shape=(1, 8),
        )
        self.assertAllClose(
            _build_rms_normalization_model(layer)(input),
            expected_output,
            atol=1e-3,
        )

    @test_combinations.run_all_keras_modes
    def test_correctness_with_scaling(self):
        normalization_layer = RMSNormalization(
            scale=True,
            center=False,
            gamma_initializer=Constant(2),
        )
        inputs = tf.constant(
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], shape=(1, 8)
        )
        expected_output = tf.constant(
            [2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0], shape=(1, 8)
        )
        self.assertAllClose(
            _build_rms_normalization_model(normalization_layer)(inputs),
            expected_output,
            atol=1e-3,
        )

    @test_combinations.run_all_keras_modes
    def test_correctness_with_centering(self):
        normalization_layer = RMSNormalization(
            scale=False,
            center=True,
            beta_initializer=Constant(10),
        )
        inputs = tf.constant(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], shape=(1, 8)
        )
        expected_output = tf.constant(
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0], shape=(1, 8)
        )
        self.assertAllClose(
            _build_rms_normalization_model(normalization_layer)(inputs),
            expected_output,
            atol=1e-3,
        )

    def _test_rms_norm(self, shape, dtype):
        np.random.seed(1)
        x_val = np.random.random_sample(shape).astype(np.float32)
        layer = itex.ops.RMSNormalization()
        inputs = tf.constant(x_val, shape=shape, dtype=dtype)
        outputs = layer(inputs, training=False)
        ref_outputs = layer(inputs, training=True)
        if dtype == tf.float32:
            self.assertAllClose(outputs, ref_outputs, atol=1e-5, rtol=1e-5)
        else:
            self.assertAllClose(outputs, ref_outputs, atol=1e-3, rtol=1e-3)

    def _runtests(self, shape):
        dtypes = [tf.float32, tf.float16, tf.bfloat16]
        for dtype in dtypes:
            self._test_rms_norm(shape, dtype)

    def testInference(self):
        shapes = [[1, 3],
                  [2, 8],
                  [4, 128],
                  [16, 128],
                  [512, 127],
                  [3, 129],
                  [4, 511],
                  [64,512],
                  [512,512],
                  [1,513],
                  [16, 1024],
                  [1023, 1023],
                  [1024, 1024]]
        for shape in shapes:
            self._runtests(shape)


if __name__ == "__main__":
    tf.test.main()
