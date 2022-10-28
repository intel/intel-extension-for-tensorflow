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
  def _test_impl(self, input_size, block_size, crop_size, dtype):
    input_array = np.random.normal(size=input_size)
    input_tensor = constant_op.constant(input_array, dtype=dtype)
    block_array = np.ones(block_size)
    block_tensor = constant_op.constant(block_array, dtype=dtypes.int32)
    crop_array = np.array([[1, input_size[ind+1]-1]
                            for ind in range(0, crop_size[0])])
    crop_tensor = constant_op.constant(crop_array, dtype=dtypes.int32)
    flush_cache()
    _ = array_ops.batch_to_space_nd(input_tensor, block_tensor,
                                    crop_tensor)

  @add_profiling
  @multi_run(ITERATION)
  def testBatchToSpaceND(self):
    for dtype in FLOAT_COMPUTE_TYPE:
      self._test_impl([64,252,1,128], [2], [2,2], dtype)
      self._test_impl([64,1007,1,128], [2], [2,2], dtype)

if __name__ == '__main__':
  test.main()
  
