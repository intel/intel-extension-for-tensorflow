import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32]  # only float32 is supported


ITERATION = 5

class RintTest(test.TestCase):
    def _test_impl(self, size, dtype):
        array = np.random.rand(*size)*10
        in_array = constant_op.constant(array, dtype=dtype)
        flush_cache()
        out_gpu = math_ops.rint(in_array)

    @add_profiling
    @multi_run(ITERATION)
    def testRound(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            self._test_impl([3, 7, 7, 3], dtype)
            self._test_impl([30523], dtype)
            self._test_impl([1,128,128,128,3], dtype)
            self._test_impl([1024,2], dtype)
            self._test_impl([1024,99], dtype)
            self._test_impl([16,17,33,33,1], dtype)
            self._test_impl([256,1], dtype)
            self._test_impl([4,128,128,3], dtype)
            self._test_impl([4,128,28,28], dtype)
            self._test_impl([4,16,16,3], dtype)
            self._test_impl([4,256,256,3], dtype)
            self._test_impl([4,32,32,3], dtype)
            self._test_impl([4,64,64,3], dtype)


if __name__ == '__main__':
    test.main()
