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
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size, broadcast_binary_size_x, broadcast_binary_size_y

try:
    from intel_extension_for_tensorflow.python.test_func import test
    INT_COMPUTE_TYPE = [dtypes.int32]
except ImportError:
    from tensorflow.python.platform import test
    INT_COMPUTE_TYPE = [dtypes.int32]

ITERATION = 5

class TruncateModTest(test.TestCase):
    def _test_impl(self, x_size, y_size, dtype):
        x = tf.random.uniform(shape=x_size, minval=0, maxval=100, dtype=dtype)
        y = tf.random.uniform(shape=y_size, minval=1, maxval=100, dtype=dtype)
        flush_cache()
        out_gpu = tf.raw_ops.TruncateMod(x=x, y=y)

    @add_profiling
    @multi_run(ITERATION)
    def testTruncateMod(self):
        for dtype in INT_COMPUTE_TYPE:
            for in_size in tailed_no_tailed_size:
                self._test_impl([in_size], [in_size], dtype)
            # test broadcast_binary_size
            for in_size in zip(broadcast_binary_size_x, broadcast_binary_size_y):
                self._test_impl(in_size[0], in_size[1], dtype)
            

if __name__ == '__main__':
    test.main()   
