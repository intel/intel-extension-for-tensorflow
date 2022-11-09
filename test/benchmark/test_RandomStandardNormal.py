# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
    INT_COMPUTE_TYPE = [tf.half, tf.bfloat16, tf.float32]
except ImportError:
    from tensorflow.python.platform import test
    INT_COMPUTE_TYPE = [tf.half, tf.bfloat16, tf.float32]

ITERATION = 5

class RandomStandardNormalTest(test.TestCase):
    def _test_impl(self, size, dtype):
        seed1, seed2 = 79, 25
        flush_cache()
        out_gpu = gen_random_ops.random_standard_normal(shape=size, dtype=dtype, seed=seed1, seed2=seed2)

    @add_profiling
    @multi_run(ITERATION)
    def testRandomStandardNormal(self):
        for dtype in INT_COMPUTE_TYPE:
            # test tailed_no_tailed_size
            for in_size in tailed_no_tailed_size:
                self._test_impl([in_size], dtype)

if __name__ == '__main__':
    test.main()  
