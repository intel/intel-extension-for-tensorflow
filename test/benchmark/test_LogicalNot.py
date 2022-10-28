import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size

try:
    from intel_extension_for_tensorflow.python.test_func import test
except ImportError:
    from tensorflow.python.platform import test

ITERATION = 5

class LogicalNotTest(test.TestCase):
    def _test_impl(self, size):
        x = np.random.normal(size=size)
        y = constant_op.constant(tf.where(x<0, False, True))
        flush_cache()
        out_gpu = math_ops.logical_not(y)

    @add_profiling
    @multi_run(ITERATION)
    def testLogicalNot(self):  # bool
        # test tailed_no_tailed_size
        for in_size in tailed_no_tailed_size:
            self._test_impl([in_size])

if __name__ == '__main__':
    test.main()  