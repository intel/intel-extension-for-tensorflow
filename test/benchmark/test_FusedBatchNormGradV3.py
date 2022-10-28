import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # only float32 is supported


ITERATION = 5

class FusedBatchNormGradV3Test(test.TestCase):
    def _test_impl(self, x_shape, scale_shape, dtype):
        data_format = "NHWC"
        grad_val = np.random.random_sample(x_shape).astype(dtype.as_numpy_dtype)
        x_val = np.random.random_sample(x_shape).astype(dtype.as_numpy_dtype)
        scale_val = np.random.random_sample(scale_shape).astype(dtype.as_numpy_dtype)
        mean_val = np.random.random_sample(scale_shape).astype(dtype.as_numpy_dtype)
        var_val = np.random.random_sample(scale_shape).astype(dtype.as_numpy_dtype)
        epsilon = 0.001
        var_reciprocal_val = np.reciprocal(np.sqrt(var_val + epsilon))
        
        reserve_space_3_val = np.random.random_sample(scale_shape).astype(dtype.as_numpy_dtype)
        
        grad = constant_op.constant(grad_val, dtype=dtype)
        x = constant_op.constant(x_val, dtype=dtype)
        scale = constant_op.constant(scale_val, dtype=dtypes.float32)
        mean = constant_op.constant(mean_val, dtype=dtypes.float32)
        var_reciprocal = constant_op.constant(var_reciprocal_val, dtype=dtypes.float32)
        reserve_space_1 = mean
        reserve_space_2 = var_reciprocal
        reserve_space_3 = reserve_space_3_val

        flush_cache()
        
        grad_x, grad_scale, grad_offset, _, _ = gen_nn_ops.fused_batch_norm_grad_v3(
          grad,
          x,
          scale,
          reserve_space_1,
          reserve_space_2,
          reserve_space_3,
          data_format=data_format,
          is_training=True)

    @add_profiling
    @multi_run(ITERATION)
    def testFusedBatchNormGradV3(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            for x_shape in [
                [4, 5, 5, 48],
                [4, 8, 8, 84],
                [4, 17, 17, 48],
                [4, 9, 27, 8],
                [4, 31, 31, 7],
                [4, 35, 35, 2],
                [4, 147, 147, 2],
                [3, 299, 299, 3],
                [5, 183, 183, 1],
                [5, 183, 183, 1],
                [5, 41, 35, 2],
                [2, 3, 4, 5],
                [2, 2, 6, 3]
            ]:
                scale_shape = [x_shape[-1]]
                self._test_impl(x_shape, scale_shape, dtype)


if __name__ == '__main__':
    test.main()    
