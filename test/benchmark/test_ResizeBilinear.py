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
from tensorflow.python.ops import image_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import common_2d_input_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
except ImportError:
    from tensorflow.python.platform import test

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA
    
ITERATION = 5

class ResizeBilinearTest(test.TestCase):
    def _test_impl(self, size, target_size, dtype):
        np.random.seed(4)
        in_array = np.random.normal(size=size)
        in_array = constant_op.constant(in_array, dtype=dtype)
        flush_cache()
        out_gpu = image_ops.resize_images_v2(in_array, target_size, method=image_ops.ResizeMethod.BILINEAR)

    @add_profiling
    @multi_run(ITERATION)
    def testResizeBilinear(self):
        image_input_size = [[1, 1080, 1920, 3], [3, 768, 1024, 3]] # NHWC
        target_size = [540, 960]
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in image_input_size:
                self._test_impl(in_size, target_size, dtype)

if __name__ == '__main__':
    test.main()
