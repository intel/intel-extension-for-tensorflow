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
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size
try:
    from intel_extension_for_tensorflow.python.test_func import test
    INT_COMPUTE_TYPE = [dtypes.int32, dtypes.int64]
except ImportError:
    from tensorflow.python.platform import test
    INT_COMPUTE_TYPE = [dtypes.int32, dtypes.int64]

ITERATION = 5

class RightShiftTest(test.TestCase):
    def _test_impl(self, size, dtype):
        x = np.random.rand(*size)
        x = constant_op.constant(x, dtype=dtype)
        y = np.random.rand(*size)
        y = constant_op.constant(x, dtype=dtype)

        flush_cache()
        _ = gen_bitwise_ops.right_shift(x, y)

    @add_profiling
    @multi_run(ITERATION)
    def testRightShift(self):
        for dtype in INT_COMPUTE_TYPE:
          # test tailed_no_tailed_size
          for in_size in [32]:
            self._test_impl([in_size], dtype)
            
if __name__ == '__main__':
    test.main()    
