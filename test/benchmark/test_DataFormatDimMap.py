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
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
import tensorflow as tf
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [tf.int32, tf.int64]  
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [tf.int32, tf.int64]  # BF16 is not supported by CUDA

ITERATION = 5

class DataFormatDimMapTest(test.TestCase):
    def _test_impl(self, m, n, dtype):
        x = np.random.randint(low=-4, high=3, size = (m,n))
        x = tf.constant(x, dtype=dtype)
        flush_cache()
        out_gpu = tf.raw_ops.DataFormatDimMap(x=x, src_format='NHWC', dst_format='NCHW', name=None)

    @add_profiling
    @multi_run(ITERATION)
    def testDataFormatDimMap(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl(8192,4096, dtype)

if __name__ == '__main__':
    test.main()    
