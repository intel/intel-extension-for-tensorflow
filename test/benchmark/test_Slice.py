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
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size, broadcast_binary_size_x, broadcast_binary_size_y

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class SliceTest(test.TestCase):
    def _test_impl(self, in_size, begin, size, dtype):
        x = np.random.normal(size=in_size)
        x = constant_op.constant(x, dtype=dtype)
        begin = constant_op.constant(begin, dtype=dtypes.int32)
        size = constant_op.constant(size, dtype=dtypes.int32)
        flush_cache()
        out_gpu = array_ops.slice(x, begin, size)

    @add_profiling
    @multi_run(ITERATION)
    def testSlice(self):
        case_list = [[[64, 179, 32], [0, 0, 0], [2, 10, 10]],
                     [[64, 179, 32], [0, 0, 0], [2, 10, 16]],
                     [[8, 179, 128], [0, 0, 0], [2, 10, 121]],
                     [[32, 1024, 64],[0, 0, 0],[32, 1023, 10]],
                     [[32, 1024, 64],[0, 0, 0],[32, 512, 61]],
                     [[32, 1024, 64],[0, 0, 0],[32, 512, 64]],
                     [[64, 1024, 1024], [0, 0, 0], [64, 1024, 512]]]
        for dtype in FLOAT_COMPUTE_TYPE:
            for case in case_list:
                self._test_impl(case[0], case[1], case[2], dtype)

if __name__ == '__main__':
    test.main()
