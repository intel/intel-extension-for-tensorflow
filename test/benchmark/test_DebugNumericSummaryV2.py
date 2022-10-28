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
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_debug_ops
from tensorflow.core.protobuf import debug_event_pb2
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

OUTPUT_DTYPE = [dtypes.float32, dtypes.float64]
DEBUG_MODE = [
    debug_event_pb2.TensorDebugMode.CURT_HEALTH,
    debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
    debug_event_pb2.TensorDebugMode.FULL_HEALTH,
    debug_event_pb2.TensorDebugMode.SHAPE,
    debug_event_pb2.TensorDebugMode.REDUCE_INF_NAN_THREE_SLOTS,
]

ITERATION = 5

class DebugNumericSummaryV2Test(test.TestCase):
    def _test_impl(self, size, input_dtype, output_dtype, debug_mode):
        np.random.seed(0)
        x = constant_op.constant(np.random.normal(size=size), dtype=input_dtype) 
        flush_cache()
        out_gpu = gen_debug_ops.debug_numeric_summary_v2(x, output_dtype=output_dtype, tensor_debug_mode=debug_mode)

    @add_profiling
    @multi_run(ITERATION)
    def testDebugNumericSummaryV2(self):
        for debug_mode in DEBUG_MODE:
            for output_dtype in OUTPUT_DTYPE:
                for dtype in FLOAT_COMPUTE_TYPE:
                    # test tailed_no_tailed_size
                    for size in tailed_no_tailed_size:
                        self._test_impl([size], dtype, output_dtype, debug_mode)

if __name__ == '__main__':
    test.main()   

