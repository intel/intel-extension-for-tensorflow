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

class BatchToSpacdNDTest(test.TestCase):
  def _test_impl(self, input_size, block, crop, dtype):
    input_array = np.random.normal(size=input_size)
    input_tensor = constant_op.constant(input_array, dtype=dtype)
    flush_cache()
    _ = array_ops.batch_to_space_nd(input_tensor, block, crop)

  @add_profiling
  @multi_run(ITERATION)
  def testBatchToSpaceND(self):
    for dtype in FLOAT_COMPUTE_TYPE:
      self._test_impl([64,1007,1,128], [2,1], [[0,0],[0,0]], dtype)
      self._test_impl([64,237,1,256], [2,1], [[1,0],[0,0]], dtype)
      self._test_impl([64,59,1,256], [2,1], [[0,0],[0,0]], dtype)
      self._test_impl([64,15,1,128], [2,1], [[1,0],[0,0]], dtype)
      self._test_impl([64,15,1,64], [2,1], [[1,0],[0,0]], dtype)
      self._test_impl([64,16,1,128], [2,1], [[2,1],[0,0]], dtype)
      self._test_impl([64,18,1,128], [2,1], [[4,3],[0,0]], dtype)
      self._test_impl([64,18,1,256], [2,1], [[4,3],[0,0]], dtype)
      self._test_impl([64,90,1,256], [2,1], [[31,31],[0,0]], dtype)
      self._test_impl([64,252,1,128], [2,1], [[1,0],[0,0]], dtype)
      self._test_impl([64,1022,1,128], [2,1], [[0,0],[0,0]], dtype)

if __name__ == '__main__':
  test.main()
  
