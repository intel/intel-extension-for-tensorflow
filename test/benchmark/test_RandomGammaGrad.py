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
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import random_ops
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test, test_util
    FLOAT_COMPUTE_TYPE = [dtypes.float32]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32]

ITERATION = 5
class RandomGammaGradTest(test.TestCase):
    def _test_impl(self, x_size, y_size, dtype):
        shape = []
        alpha = np.random.normal(size=x_size)
        alpha = constant_op.constant(alpha, dtype=dtype)
        beta = np.random.normal(size=y_size)
        beta = constant_op.constant(beta, dtype=dtype)
        flush_cache()
        sample = random_ops.random_gamma(shape, alpha, beta, seed=12345)
        grads_alpha, grads_beta = gradients_impl.gradients(sample, [alpha, beta])
        self.evaluate(grads_alpha)
        self.evaluate(grads_beta)

    @add_profiling
    @multi_run(ITERATION)
    @test_util.run_deprecated_v1
    def testRandomGammaGrad(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in [[2,2], [3,3], [254,254], [255,255], [1024,1024], [1025,1025]]:
                self._test_impl(in_size, in_size, dtype)
            for in_size in [[2,2,2,2], [3,3,3,3], [88,88,88,88], [89,89,89,89]]:
                self._test_impl(in_size, in_size, dtype)

if __name__ == '__main__':
    test.main()
