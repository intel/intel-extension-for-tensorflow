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
from tensorflow.python.ops import special_math_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class Besseli0Test(test.TestCase):
    def _test_impl(self, size, dtype):
        in_array = np.random.uniform(-1., 1., size=size)
        array = constant_op.constant(in_array, dtype=dtype)
        flush_cache()
        out_gpu = special_math_ops.bessel_i0e(array)

    @add_profiling
    @multi_run(ITERATION)
    def testBesseli0(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl([8192, 8192], dtype)
            self._test_impl([8193, 8192], dtype)
            
if __name__ == '__main__':
    test.main()    
