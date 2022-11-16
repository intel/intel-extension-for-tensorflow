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
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

# size used in models
FLOAT_LOGITS_SIZE = [[1024, 1001], [1024, 1, 100], [1024, 99, 2], [32, 1001], [8, 1000, 91]]
HALF_LOGITS_SIZE = [[1024, 1001], [32, 1001]]
BFLOAT16_LOGITS_SIZE = [[1024, 1001]]

class SoftmaxTest(test.TestCase):
    def _test_impl(self, logits, dtype):
        logits = constant_op.constant(logits, dtype=dtype)
        flush_cache()
        gen_nn_ops.softmax(logits)
    
    @add_profiling
    @multi_run(ITERATION)
    def testSoftmax(self):
        for in_size in FLOAT_LOGITS_SIZE:
            self._test_impl(np.random.normal(size=in_size), dtypes.float32)
        for in_size in HALF_LOGITS_SIZE:
            self._test_impl(np.random.normal(size=in_size), dtypes.half)
        if dtypes.bfloat16 in FLOAT_COMPUTE_TYPE:
            for in_size in BFLOAT16_LOGITS_SIZE:
                self._test_impl(np.random.normal(size=in_size), dtypes.bfloat16)

if __name__ == "__main__":
    test.main()
