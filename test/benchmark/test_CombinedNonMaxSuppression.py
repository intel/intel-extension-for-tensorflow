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
from tensorflow.python.ops import image_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache

try:
  from intel_extension_for_tensorflow.python.test_func import test
  FLOAT_COMPUTE_TYPE = [dtypes.float32]  # FP16, BF16 is not supported by itex
except ImportError:
  from tensorflow.python.platform import test
  FLOAT_COMPUTE_TYPE = [dtypes.float32]  # FP16, BF16 is not supported by CUDA

ITERATION = 5

class CombinedNonMaxSuppressionTest(test.TestCase):
  def _test_impl(self, boxes_size, scores_size, max_output_size_per_class,
                 max_total_size, dtype):
    boxes_array = np.random.uniform(size=boxes_size)
    boxes_tensor = constant_op.constant(boxes_array, dtype=dtype)
    scores_array = np.random.uniform(size=scores_size)
    scores_tensor = constant_op.constant(scores_array, dtype=dtype)
    flush_cache()
    _ = image_ops.combined_non_max_suppression(boxes_tensor, scores_tensor,
                                               max_output_size_per_class,
                                               max_total_size)

  @add_profiling
  @multi_run(ITERATION)
  def testCombinedNonMaxSuppression(self):
    for dtype in FLOAT_COMPUTE_TYPE:
      self._test_impl([4, 12288, 1, 4], [4, 12288, 1], 1000, 1000, dtype)
      self._test_impl([4, 49152, 1, 4], [4, 49152, 1], 1000, 1000, dtype)

if __name__ == '__main__':
  test.main()
