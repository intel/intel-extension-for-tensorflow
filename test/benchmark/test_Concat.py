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

concat_shape_x = [[8192, 8192], [16,28,28,168], [16,256,32,32,32], [16,256,256,64]]
concat_shape_y = [[8192, 8192], [16,28,28,160], [16,128,32,32,32], [16,256,256,64]]
concat_axis = [0, -1, 1, 2]

class ConcatTest(test.TestCase):
    def _test_impl(self, x_size, y_size, axis, dtype):
        x_array = np.random.normal(size=x_size)
        x_tensor = constant_op.constant(x_array, dtype=dtype)
        y_array = np.random.normal(size=y_size)
        y_tensor = constant_op.constant(y_array, dtype=dtype)
        flush_cache()
        out_gpu = array_ops.concat([x_tensor, y_tensor],axis)

    @add_profiling
    @multi_run(ITERATION)
    def testConcat(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for i in range(len(concat_shape_x)):
                self._test_impl(concat_shape_x[i], concat_shape_y[i], concat_axis[i], dtype)

if __name__ == '__main__':
    test.main()  
