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
from tensorflow.python.ops import gen_training_ops as training_ops
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

class ResourceApplyMomentumTest(test.TestCase):
    def _test_impl(self, size, dtype):        
        np.random.seed(0)
        var = resource_variable_ops.ResourceVariable(np.random.normal(size=size), dtype=dtype) 
        np.random.seed(4)
        accum = resource_variable_ops.ResourceVariable(np.random.normal(size=size), dtype=dtype)
        np.random.seed(8)
        grad = constant_op.constant(np.random.normal(size=size), dtype = dtype)
        flush_cache()
        op_out = training_ops.resource_apply_momentum(var=var.handle, accum=accum.handle, lr=0.001, grad=grad, momentum=0.6)

    @add_profiling
    @multi_run(ITERATION)
    def test(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            # test tailed_no_tailed_size
            for in_size in tailed_no_tailed_size:
                self._test_impl([in_size], dtype)

if __name__ == "__main__":
  test.main()
