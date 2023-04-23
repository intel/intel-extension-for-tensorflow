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


import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class SegmentSumTest(test.TestCase):
    def _test_impl(self, outer_dim, ratio, inner_dim, dtype):
        output_outer_dim = int(outer_dim / ratio)
        const = np.random.randint(5, size=(outer_dim, inner_dim))
        seg_ids = np.sort(np.random.randint(output_outer_dim, size=outer_dim))
        ids = constant_op.constant(seg_ids.astype(np.int32))
        data = constant_op.constant(const, dtype=dtype)
        flush_cache()
        out_gpu = math_ops.segment_sum(data, ids)

    @add_profiling
    @multi_run(ITERATION)
    def testSegmentSum(self):
        #from tf benchmark test
        outer_dim_options = [2**x for x in range(9, 14, 2)]
        ratio_options = [2**x for x in range(1, 6, 2)]
        inner_dim_options = [2**x for x in range(9, 14, 2)]
        options = (outer_dim_options, ratio_options, inner_dim_options, FLOAT_COMPUTE_TYPE)
        for outer_dim, ratio, inner_dim, dtype in itertools.product(*options):
            self._test_impl(outer_dim, ratio, inner_dim, dtype)
            
if __name__ == '__main__':
    np.random.seed(0)
    test.main()    
