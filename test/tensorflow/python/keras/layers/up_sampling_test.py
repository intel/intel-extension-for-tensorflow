# Copyright (c) 2022 Intel Corporation
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for up-sampling layers."""


import numpy as np
import tensorflow.compat.v2 as tf

import keras
try:
    from keras.src.testing_infra import test_combinations
    from keras.src.testing_infra import test_utils
except ImportError:
    from keras.testing_infra import test_combinations
    from keras.testing_infra import test_utils

from tensorflow.python.keras import combinations

# isort: off
from tensorflow.python.framework import (
    test_util as tf_test_utils,
)


@tf_test_utils.for_all_test_methods(
    tf_test_utils.disable_xla, "align_corners=False not supported by XLA"
)
# @test_combinations.run_all_keras_modes
class UpSamplingTest(test_combinations.TestCase):
    @combinations.generate(combinations.combine(mode=['graph']))
    def test_upsampling_3d(self):
        num_samples = 2
        stack_size = 2
        input_len_dim1 = 10
        input_len_dim2 = 11
        input_len_dim3 = 12

        for data_format in ["channels_last"]:
            if data_format == "channels_first":
                inputs = np.random.rand(
                    num_samples,
                    stack_size,
                    input_len_dim1,
                    input_len_dim2,
                    input_len_dim3,
                )
            else:
                inputs = np.random.rand(
                    num_samples,
                    input_len_dim1,
                    input_len_dim2,
                    input_len_dim3,
                    stack_size,
                )

            # basic test
            with self.cached_session():
                test_utils.layer_test(
                    keras.layers.UpSampling3D,
                    kwargs={"size": (2, 2, 2), "data_format": data_format},
                    input_shape=inputs.shape,
                )

                for length_dim1 in [2, 3]:
                    for length_dim2 in [2]:
                        for length_dim3 in [3]:
                            layer = keras.layers.UpSampling3D(
                                size=(length_dim1, length_dim2, length_dim3),
                                data_format=data_format,
                            )
                            layer.build(inputs.shape)
                            output = layer(keras.backend.variable(inputs))
                            if tf.executing_eagerly():
                                np_output = output.numpy()
                            else:
                                np_output = keras.backend.eval(output)
                            if data_format == "channels_first":
                                assert (
                                    np_output.shape[2]
                                    == length_dim1 * input_len_dim1
                                )
                                assert (
                                    np_output.shape[3]
                                    == length_dim2 * input_len_dim2
                                )
                                assert (
                                    np_output.shape[4]
                                    == length_dim3 * input_len_dim3
                                )
                            else:  # tf
                                assert (
                                    np_output.shape[1]
                                    == length_dim1 * input_len_dim1
                                )
                                assert (
                                    np_output.shape[2]
                                    == length_dim2 * input_len_dim2
                                )
                                assert (
                                    np_output.shape[3]
                                    == length_dim3 * input_len_dim3
                                )

                            # compare with numpy
                            if data_format == "channels_first":
                                expected_out = np.repeat(
                                    inputs, length_dim1, axis=2
                                )
                                expected_out = np.repeat(
                                    expected_out, length_dim2, axis=3
                                )
                                expected_out = np.repeat(
                                    expected_out, length_dim3, axis=4
                                )
                            else:  # tf
                                expected_out = np.repeat(
                                    inputs, length_dim1, axis=1
                                )
                                expected_out = np.repeat(
                                    expected_out, length_dim2, axis=2
                                )
                                expected_out = np.repeat(
                                    expected_out, length_dim3, axis=3
                                )

                            np.testing.assert_allclose(np_output, expected_out)

if __name__ == "__main__":
    tf.test.main()