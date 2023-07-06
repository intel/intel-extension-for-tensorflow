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


import copy
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
try:
  from intel_extension_for_tensorflow.python.test_func import test
  FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16, dtypes.int32]
except ImportError:
  from tensorflow.python.platform import test
  FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.int32]  # BF16 is not supported by CUDA

ITERATION = 5

gather_shape_x = [[4,128,91,28,28], [16,4,300,512], [16,8,1,512], [17451,80]]
gather_shape_y = [[4,128,3], [16,8,2], [16,4,2], [24000,2]]

class GatherNdTest(test.TestCase):
  def _test_impl(self, input_size, indices_size, dtype):
    input_array = np.random.normal(size=input_size)
    input_tensor = constant_op.constant(input_array, dtype=dtype)
    sub_size = copy.deepcopy(indices_size)
    sub_size[-1] = 1
    indices_array = np.random.uniform(size=sub_size,
                                      low=0, high=input_size[0]-1)
    for ind in range(1, indices_size[-1]):
      tmp_array = np.random.uniform(size=sub_size,
                                    low=0, high=input_size[ind]-1)
      indices_array = np.concatenate((indices_array, tmp_array), axis=-1)
    indices_tensor = constant_op.constant(indices_array, dtype=dtypes.int32)
    flush_cache()
    _ = array_ops.gather_nd(input_tensor, indices_tensor)

  @add_profiling
  @multi_run(ITERATION)
  def testGatherNd(self):
    for dtype in FLOAT_COMPUTE_TYPE:
      for i in range(len(gather_shape_x)):
        self._test_impl(gather_shape_x[i], gather_shape_y[i], dtype)

if __name__ == '__main__':
  test.main()
