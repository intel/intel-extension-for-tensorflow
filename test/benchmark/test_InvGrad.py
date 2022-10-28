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

class InvGradTest(test.TestCase):
    def _test_impl(self, size, dtype):
        array = np.random.rand(*size)
        array2 = np.random.rand(*size)
        in_array = constant_op.constant(array, dtype=dtype)
        in_array2 = constant_op.constant(array2, dtype=dtype)
        flush_cache()
        out_gpu = tf.raw_ops.InvGrad(y=in_array, dy=in_array2)

    @add_profiling
    @multi_run(ITERATION)
    def testInvGrad(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl([30523], dtype)
            self._test_impl([1,128,128,128,3], dtype)
            self._test_impl([1024,2], dtype)
            self._test_impl([1024,99], dtype)
            self._test_impl([16,17,33,33,1], dtype)
            self._test_impl([256,1], dtype)
            self._test_impl([4,128,128,3], dtype)
            self._test_impl([4,128,28,28], dtype)
            self._test_impl([4,16,16,3], dtype)
            self._test_impl([4,256,256,3], dtype)
            self._test_impl([4,32,32,3], dtype)
            self._test_impl([4,64,64,3], dtype)
            self._test_impl([], dtype)
            
if __name__ == '__main__':
    test.main()    
