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
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache

try:
  from intel_extension_for_tensorflow.python.test_func import test
  FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
  from tensorflow.python.platform import test
  FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class Dilation2DTest(test.TestCase):
  def _test_impl(self, input_size, filter_size, strides, dilations, dtype):
    input_array = np.random.normal(size=input_size)
    input_tensor = constant_op.constant(input_array, dtype=dtype)
    filter_array = np.random.normal(size=filter_size)
    filter_tensor = constant_op.constant(filter_array, dtype=dtype)
    flush_cache()
    _ = nn_ops.dilation2d(input_tensor, filter_tensor,
                          strides, dilations, "VALID")

  @add_profiling
  @multi_run(ITERATION)
  def testDilation2D(self):
    for dtype in FLOAT_COMPUTE_TYPE:
      self._test_impl([128, 1024, 1024, 3], [3, 3, 3],
                      [1, 1, 1, 1], [1, 1, 1, 1], dtype)
      self._test_impl([128, 512, 512, 3], [18, 18, 3],
                      [1, 1, 1, 1], [1, 1, 1, 1], dtype)

if __name__ == '__main__':
  test.main()
