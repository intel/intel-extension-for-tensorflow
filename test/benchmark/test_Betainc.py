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
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32]

ITERATION = 5

class BetaincTest(test.TestCase):
    def _test_impl(self, a_size, b_size, x_size, dtype):
        a = np.random.normal(size=a_size)
        a = constant_op.constant(a, dtype=dtype)
        b = np.random.normal(size=b_size)
        b = constant_op.constant(b, dtype=dtype)
        x = np.random.normal(size=x_size)
        x = constant_op.constant(x, dtype=dtype)
        flush_cache()
        out_gpu = math_ops.betainc(a, b, x)

    @add_profiling
    @multi_run(ITERATION)
    def testBetainc(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            # test tailed_no_tailed_size
            for in_size in tailed_no_tailed_size:
                self._test_impl([in_size], [in_size], [in_size], dtype)

if __name__ == '__main__':
    test.main()
