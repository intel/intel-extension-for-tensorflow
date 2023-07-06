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

class ResizeNearestNeighborTest(test.TestCase):
    def _test_impl(self, in_size, out_size, dtype):
        input_tensor = np.random.normal(size=in_size)
        input_tensor = constant_op.constant(input_tensor, dtype=dtype)
        output_shape =  constant_op.constant(out_size, dtype=dtypes.int32)
        flush_cache()
        gen_image_ops.resize_nearest_neighbor(input_tensor, output_shape)

    @add_profiling
    @multi_run(ITERATION)
    def testResizeNearestNeighbor(self):
        cases = [[[512,64,64,3], [32,32]],
                 [[256,256,256,3], [64,64]],
                 [[32,512,512,3], [128,128]],
                 [[64,38,38,91], [9,9]]]
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in cases:
                self._test_impl(in_size[0], in_size[1], dtype)

if __name__ == '__main__':
    test.main()
