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
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import dtypes
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA
    
ITERATION = 5
size = [3, 7, 7, 3]
ksize = [1, 1, 1, 1]
strides = [1, 1, 1, 1]
padding = "SAME"
origin_input = np.random.normal(size=size)
origin_input = constant_op.constant(origin_input)
_, _argmax = gen_nn_ops.max_pool_with_argmax(origin_input, ksize, strides, padding)

class MaxPoolGradWithArgmaxTest(test.TestCase):

    def _test_impl(self, dtype):
        origin_input = np.random.normal(size=size)
        grad = np.random.normal(size=size)
        origin_input = constant_op.constant(origin_input, dtype=dtype)
        grad = constant_op.constant(grad, dtype=dtype)
        flush_cache()
        _result = gen_nn_ops.max_pool_grad_with_argmax(origin_input, grad, _argmax, ksize, strides, padding)

    @add_profiling
    @multi_run(ITERATION)
    def testMaxPoolGradWithArgmax(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl(dtype)


if __name__ == "__main__":
    test.main()
