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
from tensorflow.python.ops import nn_ops
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

class DataFormatVecPermuteTest(test.TestCase):
    def _test_1D(self, m, dtype):
        x = np.random.randint(1000, size = (m))
        x = tf.constant(x, dtype=dtype)
        flush_cache()
        out_gpu = nn_ops.data_format_vec_permute(
            x=x, src_format='NHWC', dst_format='NCHW')

    def _test_2D(self, m, n, dtype):
        x = np.random.randint(1000, size = (m,n))
        x = tf.constant(x, dtype=dtype)
        flush_cache()
        out_gpu = nn_ops.data_format_vec_permute(
            x=x, src_format='NDHWC', dst_format='NCDHW')

    @add_profiling
    @multi_run(ITERATION)
    def testDataFormatVecPermute(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_2D(5,2, dtype)
            self._test_1D(4, dtype)

if __name__ == '__main__':
    test.main()    
