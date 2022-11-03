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

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.int32] 
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.int32]  # BF16 is not supported by CUDA

ITERATION = 5

class BincountTest(test.TestCase):
    def _test_impl(self, m, n, dtype):
        np.random.seed(0)
        x = np.random.choice(m, n, replace=True)
        y = np.random.choice(m, n, replace=True)
        values = constant_op.constant(x, dtype=tf.int32)
        weights = tf.constant(y, dtype=dtype)
        flush_cache()
        out_gpu = math_ops.dense_bincount(values, size=n, weights=weights, binary_output=False)

    @add_profiling
    @multi_run(ITERATION)
    def testBincount(self):
        # argument dtype of dense_bincount
        # weight: int32, float32
        for dtype in FLOAT_COMPUTE_TYPE:
            for size in [8192, 8193, 16384, 16385]:
                self._test_impl(size, size, dtype)



if __name__ == '__main__':
    test.main()   
