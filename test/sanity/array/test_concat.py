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
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops


class ConcatOpTest(test_util.TensorFlowTestCase):
    """test concat op"""

    @test_util.run_deprecated_v1
    def test4DStackBf16(self):
        with self.session(use_gpu=True):
            p1 = array_ops.placeholder(dtypes.bfloat16, shape=[2, 3, 1, 1])
            p2 = array_ops.placeholder(dtypes.bfloat16, shape=[2, 3, 4, 1])
            c = array_ops.concat([p1, p2], 2)
            params = {
                p1: np.random.rand(2, 3, 1, 1).astype("f"),
                p2: np.random.rand(2, 3, 4, 1).astype("f")
            }
            result = c.eval(feed_dict=params)

        self.assertEqual(result.shape, c.get_shape())
        self.assertAllEqual(result[:, :, :1, :], tf.cast(params[p1], dtypes.bfloat16))
        self.assertAllEqual(result[:, :, 1:, :], tf.cast(params[p2], dtypes.bfloat16))
        

if __name__ == '__main__':
    test.main()
