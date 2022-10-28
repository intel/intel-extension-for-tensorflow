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
from tensorflow.python.ops import variables
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size, broadcast_binary_size_x, broadcast_binary_size_y

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class StridedSliceGradTest(test.TestCase):
    def _test_impl(self, in_size, dtype):
        grad = variables.Variable([1,2,3], dtype=dtype)
        varshape = variables.Variable(in_size, dtype=dtypes.int32)
        begin = constant_op.constant([0], dtype=dtypes.int32)
        end = constant_op.constant([3], dtype=dtypes.int32)
        strides = constant_op.constant([1], dtype=dtypes.int32)
        flush_cache()
        out_gpu = array_ops.strided_slice_grad(varshape, begin, end, strides, grad)


    @add_profiling
    @multi_run(ITERATION)
    def testStridedSliceGrad(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in tailed_no_tailed_size:
                self._test_impl([in_size], dtype)

if __name__ == '__main__':
    test.main()   
