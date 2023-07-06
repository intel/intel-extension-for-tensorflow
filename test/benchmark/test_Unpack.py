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
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class UnPackTest(test.TestCase):
    def _test_impl(self, x_size, num, axis, dtype):
        x_array = np.random.normal(size=x_size)
        x_tensor = constant_op.constant(x_array, dtype=dtype)
        flush_cache()
        out_gpu1 = array_ops.unpack(x_tensor, num, axis)

    @add_profiling
    @multi_run(ITERATION)
    def testUnPack(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl([2, 8192, 8192], 2, 0, dtype)
            self._test_impl([16,3,224,224], 3, 1, dtype)
            self._test_impl([4,19200], 4, 0, dtype)
            self._test_impl([1,300,300,3], 1, 0, dtype)
            
if __name__ == '__main__':
    test.main() 
