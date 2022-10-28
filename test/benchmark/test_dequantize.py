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
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
except ImportError:
    from tensorflow.python.platform import test

FLOAT_COMPUTE_TYPE = [dtypes.float32]

ITERATION = 5

class DequantizeTest(test.TestCase):
    def _test_sym(self, size, dtype):
        in_array = np.random.uniform(low=-128, high=127, size=[size]).astype(np.int8)
        x = constant_op.constant(in_array, dtype=dtypes.qint8)
        x_min = -5.0
        x_max = 5.0
        flush_cache()
        dequantize_op = array_ops.dequantize(x, x_min, x_max, mode="SCALED")
        out_gpu = array_ops.identity(dequantize_op)

    def _test_asym(self, size, dtype):
        in_array = np.random.uniform(low=0, high=255, size=[size]).astype(np.uint8)
        x = constant_op.constant(in_array, dtype=dtypes.quint8)
        x_min = -5.0
        x_max = 5.0
        flush_cache()
        dequantize_op = array_ops.dequantize(x, x_min, x_max, mode="MIN_FIRST")
        out_gpu = array_ops.identity(dequantize_op)

    @add_profiling
    @multi_run(ITERATION)
    def testdequantizesym(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_sym(102760448, dtype)
            self._test_sym(3211264, dtype)

    # ITEX currently doesn't support GPU asymmetric quantization
    # Since oneDNN doesn't support GPU reorder with zeropoint
    # @add_profiling
    # @multi_run(ITERATION)
    # def testdequantizeasym(self):
    #     for dtype in FLOAT_COMPUTE_TYPE:
    #         self._test_asym(102760448, dtype)
    #         self._test_asym(3211264, dtype)
            
if __name__ == '__main__':
    test.main()    
