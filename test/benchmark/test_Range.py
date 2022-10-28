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
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class RangeTest(test.TestCase):
    def _test_impl(self, x_start, x_end, x_range1, x_range2, dtype):
        x1 = tf.constant(x_start, dtype=dtype)
        x2 = tf.constant(x_end, dtype=dtype)
        np.random.seed(4)
        x3 = tf.constant(x_range1+x_range2*np.random.random(), dtype=dtype)
        flush_cache()
        tf_val = math_ops.range(x1, x2, x3, dtype=dtype)

    @add_profiling
    @multi_run(ITERATION)
    def test(self):
        try:
            from intel_extension_for_tensorflow.python.test_func import test
            DEFAULT_DATA_TYPES = [tf.dtypes.float32, tf.dtypes.bfloat16]
        except ImportError:
            DEFAULT_DATA_TYPES=[tf.dtypes.float32]
        for dtype in DEFAULT_DATA_TYPES: 
            self._test_impl(1, 819200, 1, 1, dtype=dtype)
            self._test_impl(1024, 1024000, 1, 5, dtype=dtype)


if __name__ == '__main__':
    test.main()  
