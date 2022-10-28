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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size, broadcast_binary_size_x, broadcast_binary_size_y
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class FloorModTest(test.TestCase):
    def _test_impl(self, x_shape, y_shape, dtype):
        np.random.seed(4)
        x1 = tf.constant(np.random.normal(size=x_shape), dtype=dtype)
        np.random.seed(8)
        x2 = tf.constant(tf.ones(y_shape, dtype=dtype), dtype=dtype)
        flush_cache()
        tf_val = math_ops.floor_mod(x1, x2)
            
    @add_profiling
    @multi_run(ITERATION)
    def testOp(self):
        for dtype in [tf.int64]:
            # test tailed_no_tailed_size
            for in_size in tailed_no_tailed_size:
                self._test_impl([in_size], [in_size], dtype)
            # test broadcast_binary_size
            for in_size in zip(broadcast_binary_size_x, broadcast_binary_size_y):
                self._test_impl(in_size[0], in_size[1], dtype)

if __name__ == '__main__':
    test.main()  
