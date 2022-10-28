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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

DEFAULT_DATA_TYPES=[tf.dtypes.bfloat16, tf.dtypes.float16, tf.dtypes.float32]


class CumsumTest(test.TestCase):
    def _test_impl(self,  in_size, kaxis,  dtype):
        in_array = np.random.uniform(0, 1, in_size)
        np_res = np.cumsum(in_array, axis=kaxis, dtype=np.float32)

        with test_util.use_gpu():
            array = tf.constant(in_array, dtype=dtype)
            out_gpu = tf.math.cumsum(
                array, axis=kaxis)

        self.assertAllClose(np_res, out_gpu, 1e-2, 1e-2)
    
    def test_full_scan(self):
        for dtype in DEFAULT_DATA_TYPES:
                self._test_impl([3], 0,  dtype)

    
    def test_partial_scan(self):
      for dtype in DEFAULT_DATA_TYPES:
        for kaxis in [0, 1]:
              self._test_impl([2, 2], kaxis,  dtype)



if __name__ == "__main__":
  test.main()
