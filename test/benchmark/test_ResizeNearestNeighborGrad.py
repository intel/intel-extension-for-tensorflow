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
from tensorflow.python.ops import gen_image_ops
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class ResizeNearestNeighborGradTest(test.TestCase):
    def _test_impl(self, in_size, original_size, dtype):
        grad = np.random.normal(size=in_size)
        grad = constant_op.constant(grad, dtype=dtype)
        original_shape = constant_op.constant(original_size, dtype=dtypes.int32)
        flush_cache()
        gen_image_ops.resize_nearest_neighbor_grad(grad, original_shape)

    @add_profiling
    @multi_run(ITERATION)
    def testResizeNearestNeighborGrad(self):
        cases = [[[512,32,32,3], [64,64]],
                 [[256,64,64,3], [256,256]],
                 [[32,128,128,3], [512,512]],
                 [[64,9,9,91], [38,38]]]
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in cases:
                self._test_impl(in_size[0], in_size[1], dtype)

if __name__ == '__main__':
    test.main()
