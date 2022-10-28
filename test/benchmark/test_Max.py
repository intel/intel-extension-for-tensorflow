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
from utils import reduction_size, reduction_axis

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class MaxTest(test.TestCase):
    def _test_impl(self, size, axis, dtype):
        x = np.random.normal(size=size)
        x = constant_op.constant(x, dtype=dtype)
        flush_cache()
        out_gpu = tf.compat.v1.raw_ops.Max(input=x,axis=axis,keep_dims=False)

    @add_profiling
    @multi_run(ITERATION)
    def testMax(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            # test tailed_no_tailed_size
            for in_size in reduction_size:
                for axis in reduction_axis:
                    self._test_impl(in_size, axis, dtype)

if __name__ == '__main__':
    test.main()  
