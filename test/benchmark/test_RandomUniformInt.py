import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
    INT_COMPUTE_TYPE = [dtypes.int32, dtypes.int64]
except ImportError:
    from tensorflow.python.platform import test
    INT_COMPUTE_TYPE = [dtypes.int32, dtypes.int64]  # BF16 is not supported by CUDA

ITERATION = 5

class RandomUniformIntTest(test.TestCase):
    def _test_impl(self, size, dtype):
        shape = constant_op.constant([4, 7])
        seed1, seed2 = 79, 25
        minval2 = constant_op.constant(1, dtype=dtype)
        maxval2 = constant_op.constant(50, dtype=dtype)
        flush_cache()
        out_gpu = gen_random_ops.RandomUniformInt(shape=shape, minval=minval2, maxval=maxval2, seed=seed1, seed2=seed2)

    @add_profiling
    @multi_run(ITERATION)
    def testRandomUniformInt(self):
        for dtype in INT_COMPUTE_TYPE:
            # test tailed_no_tailed_size
            for in_size in tailed_no_tailed_size:
                self._test_impl([in_size], dtype)
print('xuerui\n')

if __name__ == '__main__':
    test.main()  
