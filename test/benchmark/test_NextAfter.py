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
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32] # stock tf does not register bf16 or half op
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32]  # BF16 is not supported by CUDA

ITERATION = 5

class NextAfterTest(test.TestCase):
    def _test_impl(self, size, dtype):
        in_array_1 = np.random.normal(size=size)
        in_array_2 = np.random.normal(size=size)
        array_1 = constant_op.constant(in_array_1, dtype=dtype)
        array_2 = constant_op.constant(in_array_2, dtype=dtype)
        flush_cache()
        out_gpu = math_ops.nextafter(array_1, array_2)

    @add_profiling
    @multi_run(ITERATION)
    def testNextAfter(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl([1024], dtype)
            
if __name__ == '__main__':
    test.main()    
