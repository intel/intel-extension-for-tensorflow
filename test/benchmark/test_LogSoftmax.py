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
from utils import tailed_no_tailed_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

FLOAT_LOGITS = [[152,30522], [304,30522], [2,2], [4,2], [8,2], [32,2]]

class LogSoftmaxTest(test.TestCase):
    def _test_impl(self, features, dtype):
        features = constant_op.constant(features, dtype=dtype)
        flush_cache()
        nn_ops.log_softmax(features)

    @add_profiling
    @multi_run(ITERATION)
    def testLogSoftmax(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in tailed_no_tailed_size:
                logits = np.random.normal(size=in_size)
                self._test_impl(logits, dtype)
        for logits in FLOAT_LOGITS:
            self._test_impl(np.array(logits).astype(np.float32), dtypes.float32)

if __name__ == "__main__":
    test.main()
