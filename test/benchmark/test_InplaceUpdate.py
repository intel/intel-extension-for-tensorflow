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
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5
X_SHAPES = [[1024, 100], [1024, 32], [800, 256, 256]]
V_SHAPES = [[1, 100], [1, 32], [1, 256, 256]]
class InpalceUpdateTest(test.TestCase):
    def _test_impl(self, x, i, v):
        flush_cache()
        gen_array_ops.inplace_update(x, i, v)

    def _test_float32(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for shapes in zip(X_SHAPES, V_SHAPES):
                x = array_ops.ones(shape=shapes[0], dtype=dtype)
                i = np.random.randint(1,1024)
                v = array_ops.ones(shape=shapes[1], dtype=dtype)
                self._test_impl(x, [i], v)

    @add_profiling
    @multi_run(ITERATION)
    def testInplaceUpdate(self):
        self._test_float32()

if __name__ == '__main__':
    test.main()
