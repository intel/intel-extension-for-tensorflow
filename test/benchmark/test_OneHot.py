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

try:
    from intel_extension_for_tensorflow.python.test_func import test

    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test

    FLOAT_COMPUTE_TYPE = [
        dtypes.float32,
        dtypes.float16,
    ]  # BF16 is not supported by CUDA

ITERATION = 5


class OneHotTest(test.TestCase):
    def _test_impl(self, in_size, depth, axis, dtype):
        indices = np.random.randint(depth, size=in_size)
        indices = constant_op.constant(indices)
        on_value = 1.0
        flush_cache()
        out_gpu = array_ops.one_hot(
            indices=indices, depth=depth, on_value=on_value, axis=axis, dtype=dtype
        )

    @add_profiling
    @multi_run(ITERATION)
    def testOneHot(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size, depth in [([2048], 2), ([4, 512], 91), ([512], 1100), ([1024], 93184), ([2048], 186368)]:
                for axis in [-1, 0]:
                    self._test_impl(in_size, depth, axis, dtype)


if __name__ == "__main__":
    test.main()
