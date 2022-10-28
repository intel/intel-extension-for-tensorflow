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
from utils import broadcast_binary_size_x
from utils import broadcast_binary_size_y

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5
FLOAT_LABELS = [[2048,91]]
FLOAT_LOGITS = [[2048,91]]

class SoftmaxCrossEntropyWithLogitsTest(test.TestCase):
    def _test_impl(self, labels, logits, dtype):
        labels = constant_op.constant(labels, dtype=dtype)
        logits = constant_op.constant(logits, dtype=dtype)
        flush_cache()
        nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    @add_profiling
    @multi_run(ITERATION)
    def testSoftmaxCrossEntropyWithLogits(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for in_size in zip(broadcast_binary_size_x, broadcast_binary_size_x):
                lables = np.random.normal(size=in_size[0])
                logits = np.random.normal(size=in_size[1])
                self._test_impl(lables, logits, dtype=dtype)
        for input in zip(FLOAT_LABELS, FLOAT_LOGITS):
            self._test_impl(input[0], input[1], dtype=dtypes.float32)

if __name__ == "__main__":
    test.main()
