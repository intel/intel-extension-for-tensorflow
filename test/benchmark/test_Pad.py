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
from utils import tailed_no_tailed_size, broadcast_binary_size_x, broadcast_binary_size_y

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.double]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class PadTest(test.TestCase):
    def _test_impl(self, in_size, dtype):
        x = np.random.normal(size=in_size)
        x = constant_op.constant(x, dtype=dtype)
        paddings = tf.constant([[1, 1,], [2, 2], [1, 1], [2, 2]])
        flush_cache()
        out_gpu = array_ops.pad(x, paddings)


    @add_profiling
    @multi_run(ITERATION)
    def testPad(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in [[4,128,128,256], [4,16,16,256], [16,512,14,14], [16,128,56,56]]:
                self._test_impl(in_size, dtype)

if __name__ == '__main__':
    test.main()
