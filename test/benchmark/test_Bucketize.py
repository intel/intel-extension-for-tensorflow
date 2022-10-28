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
except ImportError:
    from tensorflow.python.platform import test

FLOAT_COMPUTE_TYPE = [dtypes.float32] # Bucketize op does not support bf16  
ITERATION = 5

class BucketizeTest(test.TestCase):
    def _test_impl(self, input_size, boundary_size, dtype):
        input_array = np.random.normal(size=input_size)
        boundary_array = np.sort(np.random.normal(size=boundary_size))
        input_tensor = constant_op.constant(input_array, dtype=dtype)
        boundary_list  = list(boundary_array)
        flush_cache()
        out_gpu = math_ops.bucketize(input_tensor, boundary_list)

    @add_profiling
    @multi_run(ITERATION)
    def testBucketize(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            # test bucketize_ops
            self._test_impl(30523, 1024, dtype)

if __name__ == '__main__':
    test.main()
