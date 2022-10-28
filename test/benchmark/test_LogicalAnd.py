import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache
from utils import tailed_no_tailed_size, broadcast_binary_size_x, broadcast_binary_size_y

try:
    from intel_extension_for_tensorflow.python.test_func import test
except ImportError:
    from tensorflow.python.platform import test
    
ITERATION = 5

class LogicalAndTest(test.TestCase):
    def _test_impl(self, x_size, y_size):
        x = np.random.normal(size=x_size)
        logicalx = constant_op.constant(tf.where(x<0, False, True))
        y = np.random.normal(size=y_size)
        logicaly = constant_op.constant(tf.where(y<0, False, True))
        flush_cache()
        out_gpu = math_ops.logical_and(logicalx, logicaly)

    @add_profiling
    @multi_run(ITERATION)
    def testLogicalAnd(self):  # bool
        # test tailed_no_tailed_size
        for in_size in tailed_no_tailed_size:
            self._test_impl([in_size], [in_size])
        # test broadcast_binary_size
        for in_size in zip(broadcast_binary_size_x, broadcast_binary_size_y):
            self._test_impl(in_size[0], in_size[1])
            
if __name__ == '__main__':
    test.main()    
