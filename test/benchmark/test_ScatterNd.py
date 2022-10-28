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
from utils import multi_run, add_profiling, flush_cache
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class ScatterNdTest(test.TestCase):
    def _test1d(self,  m, n, dtype):
        np.random.seed(0)
        indice = constant_op.constant(np.random.choice(m, n, replace=False).reshape(n,1), dtype=tf.int64)
        update = constant_op.constant(np.random.choice(m, n, replace=False), dtype=dtype)
        shape = constant_op.constant([m], dtype=tf.int64) 
        flush_cache()
        op = array_ops.scatter_nd(indices=indice, updates=update, shape=shape)
            
    def _test2d(self,  m, n, dtype):
        np.random.seed(0)
        indice = constant_op.constant(np.random.choice(m, n, replace=False).reshape(n,1), dtype=tf.int64) 
        update = constant_op.constant(np.random.choice(m, n, replace=False).repeat(m).reshape(n,m), dtype=dtype)
        shape = constant_op.constant([m, m], dtype=tf.int64) 
        flush_cache()
        op = array_ops.scatter_nd(indices=indice, updates=update, shape=shape)

    @add_profiling
    @multi_run(ITERATION)
    def test(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test1d(1024, 256, dtype)
            self._test1d(8192, 4095, dtype)
            self._test2d(1024, 256, dtype)
            self._test2d(8192, 4095, dtype)


if __name__ == '__main__':
    test.main()  
