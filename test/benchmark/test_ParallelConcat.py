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
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework.ops import disable_eager_execution
from utils import multi_run, add_profiling, flush_cache, common_2d_input_size, broadcast_binary_size_y
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class ParallelConcatTest(test.TestCase):
    def _test_impl(self, x_size, out_size, dtype):
        x_array = np.random.normal(size=x_size)
        x_tensor = constant_op.constant(x_array, dtype=dtype)
        y_array = np.random.normal(size=x_size)
        y_tensor = constant_op.constant(y_array, dtype=dtype)
        flush_cache()
        out_gpu = gen_array_ops.parallel_concat(values=list([x_tensor, y_tensor]), shape=out_size)

    @add_profiling
    @multi_run(ITERATION)
    def testParallelConcat(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl([1,8192, 8192], [2,8192, 8192], dtype)

if __name__ == '__main__':
    test.main()  
