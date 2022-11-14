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
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from utils import multi_run, add_profiling, flush_cache
from tensorflow.python.framework import constant_op
from utils import tailed_no_tailed_size
try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5
tile_size = [[[8,128,1],[16,32,3]], [[8,32,1],[2,256,1]], [[8,512,1],[3,2,1024]], [[8,8,1],[3,128,3]]] 

class TileTest(test.TestCase):
    def _test_impl(self, in_size, mul_times, dtype):
        x = np.random.normal(size=in_size)
        x = constant_op.constant(x, dtype=dtype)
        y = constant_op.constant(mul_times, dtype=dtypes.int32)
        flush_cache()
        out_gpu = array_ops.tile(x, y)

    @add_profiling
    @multi_run(ITERATION)
    def testTile(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size, in_size2 in tile_size:
                self._test_impl(in_size, in_size2, dtype)

if __name__ == '__main__':
    test.main()  
