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
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class DoubleTest(test_util.TensorFlowTestCase):
    """ test kernels with double data type."""

    def testCastInt64ToDouble(self):
        with self.session(use_gpu=True):
            x = np.array([100, 200]).astype(np.int64)
            val = constant_op.constant(x, x.dtype)
            cast = math_ops.cast(val, dtypes.float64, name="cast")
            self.assertAllClose(x.astype(np.float64), self.evaluate(cast))

    def testCastDoubleToFloat(self):
        with self.session(use_gpu=True):
            x = np.array([100, 200]).astype(np.float64)
            val = constant_op.constant(x, x.dtype)
            cast = math_ops.cast(val, dtypes.float32, name="cast")
            self.assertAllClose(x.astype(np.float32), self.evaluate(cast))

    def _compareGpuUnary(self, x, np_func, tf_func):
        np_ans = np_func(x)
        with self.session(use_gpu=True):
            result = tf_func(ops.convert_to_tensor(x))
            tf_gpu = self.evaluate(result)
            self.assertAllClose(np_ans, tf_gpu)

    def _compareGpuBinary(self, x, y, np_func, tf_func):
        np_ans = np_func(x, y)
        with self.session(use_gpu=True):
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = tf_func(inx, iny)
            tf_gpu = self.evaluate(out)
        self.assertAllClose(np_ans, tf_gpu)
        self.assertShapeEqual(np_ans, out)

    def testExp(self):
        x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
        self._compareGpuUnary(x, np.exp, math_ops.exp)

    def testAddV2(self):
        x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.float64)
        y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.float64)
        self._compareGpuBinary(x, y, np.add, math_ops.add_v2)

    def testAddV2_Complex64(self):
        x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.complex64)
        y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.complex64)
        self._compareGpuBinary(x, y, np.add, math_ops.add_v2)

    def testAddV2_Complex128(self):
        x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.complex128)
        y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.complex128)
        self._compareGpuBinary(x, y, np.add, math_ops.add_v2)       

    def testRealDiv(self):
        nums = np.arange(-10, 10, .25).astype(np.float64).reshape(80, 1)
        divs = np.arange(-3, 0, .25).astype(np.float64).reshape(1, 12)
        tf_result = math_ops.realdiv(nums, divs)
        np_result = np.divide(nums, divs)
        self.assertAllClose(tf_result, np_result)


if __name__ == '__main__':
    test.main()
