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
from utils import multi_run, add_profiling, flush_cache, broadcast_binary_size_x

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class SpaceToBatchNDTest(test.TestCase):
    def _test_impl(self, size, block_shape, paddings, dtype):
        in_array = np.random.normal(size=size)
        in_array = constant_op.constant(in_array, dtype=dtype)
        flush_cache()
        out_gpu = tf.raw_ops.SpaceToBatchND(input=in_array, block_shape=block_shape, paddings=paddings, name=None)
    
    @add_profiling
    @multi_run(ITERATION)
    def testSpaceToBatchND(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl([32,2044,1,64], [2,1], [[0,0],[0,0]],dtype)
            self._test_impl([32,503,1,128], [2,1], [[1,0],[0,0]],dtype)
            self._test_impl([32,118,1,256], [2,1], [[31,31],[0,0]],dtype)
            self._test_impl([32,29,1,256], [2,1], [[4,3],[0,0]],dtype)
            self._test_impl([32,29,1,128], [2,1], [[4,3],[0,0]],dtype)
            self._test_impl([32,29,1,128], [2,1], [[2,1],[0,0]],dtype)
            self._test_impl([32,29,1,64], [2,1], [[1,0],[0,0]],dtype)
            self._test_impl([32,29,1,128], [2,1], [[1,0],[0,0]],dtype)
            self._test_impl([32,118,1,256], [2,1], [[0,0],[0,0]],dtype)
            self._test_impl([32,473,1,256], [2,1], [[1,0],[0,0]],dtype)
            self._test_impl([32,2014,1,256], [2,1], [[0,0],[0,0]],dtype)

if __name__ == '__main__':
    test.main()
