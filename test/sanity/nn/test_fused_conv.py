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

"""Tests for _FusedITEXConv2D related functionality in tensorflow.ops.nn."""
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

tf.compat.v1.disable_eager_execution()

@test_util.run_all_in_native_and_block_format
class FusedConv2DTest(test_util.TensorFlowTestCase):
    """test _FusedITEXConv2D"""
    def test_conv_biasadd(self):
        x = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
        w = np.ones([1, 2, 1, 1]).astype(np.float32)
        b = np.array([1], dtype=np.float32)
        expected_result = np.array([[[[4.], [6.], [4.]],
                                    [[10.], [12.], [7.]],
                                    [[16.], [18.], [10.]]]])
        with self.cached_session(use_gpu=True) as sess:
            fused_graph = array_ops.identity(tf.nn.bias_add(nn_ops.Conv2D(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC'), b))
            result = sess.run(array_ops.identity(fused_graph))
            self.assertAllClose(expected_result, result)


if __name__ == '__main__':
    test.main()
