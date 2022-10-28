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
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import gradient_checker_v2
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test, test_util
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class ResizeNearestNeighborGradTest(test.TestCase):
    def _test_impl(self, in_size, out_size, dtype):
        input_tensor = np.random.normal(size=in_size)
        input_tensor = constant_op.constant(input_tensor, dtype=dtype)
        def resize_nn(t, shape=out_size):
            return image_ops.resize_nearest_neighbor(t, shape[1:3])
        flush_cache()
        with self.cached_session():
            err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(
                resize_nn, [input_tensor], delta=1 / 8))

    @add_profiling
    @multi_run(ITERATION)
    @test_util.run_deprecated_v1
    def testResizeNearestNeighborGrad(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            # should not use too large size, otherwise it will fail to allocate memory
            for in_size in zip([[1,4,6,1], [5,7,9,1]], [[1,2,3,1], [5,4,6,5]]):
                self._test_impl(in_size[0], in_size[1], dtype)

if __name__ == '__main__':
    test.main()
