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

class GatherV2Test(test.TestCase):
  def _test_impl(self, input_size, indices_size, axis, dtype):
    input_array = np.random.normal(size=input_size)
    input_tensor = constant_op.constant(input_array, dtype=dtype)
    indices_array = np.random.uniform(size=indices_size,
                                      low=0, high=input_size[0]-1)
    indices_tensor = constant_op.constant(indices_array, dtype=dtypes.int32)
    axis_tensor = constant_op.constant(axis, dtype=dtypes.int32)
    flush_cache()
    _ = array_ops.gather_v2(input_tensor, indices_tensor, axis_tensor)

  @add_profiling
  @multi_run(ITERATION)
  def testGatherV2(self):
    for dtype in FLOAT_COMPUTE_TYPE:
      self._test_impl([1310720,256], [401408], 0, dtype)
      self._test_impl([261888,4], [49152], 0, dtype)

if __name__ == '__main__':
  test.main()
