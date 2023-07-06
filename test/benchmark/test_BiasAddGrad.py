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


from turtle import ycor
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import broadcast_binary_size_x

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    # BF16 is not supported by CUDA
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]

ITERATION = 5
bias_add_grad_shapes = [
    [32, 118, 1, 256],
    [32, 128],
    [32, 1],
    [32, 2014, 1, 128],
    [32, 29, 1, 128],
    [32, 29, 1, 64],
    [32, 473, 1, 256],
    [32, 8177, 1, 64],
    [1, 17, 33, 33, 1],
    [1, 17, 33, 33, 32]
]
bias_add_grad_shapes += broadcast_binary_size_x


class BiasAddGradTest(test.TestCase):
    def _test_impl(self, size, dtype):
        np.random.seed(1)
        x = np.random.normal(size=size)
        x = constant_op.constant(x, dtype=dtype)
        flush_cache()
        out = gen_nn_ops.bias_add_grad(x)

    @add_profiling
    @multi_run(ITERATION)
    def testBiasAddGrad(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            # should not use too large size, otherwise it will fail to allocate memory
            for in_size in bias_add_grad_shapes:
                self._test_impl(in_size, dtype)


if __name__ == '__main__':
    test.main()
