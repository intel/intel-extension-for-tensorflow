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
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.framework import constant_op
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA

ITERATION = 5

class SparseTensorDenseMatMulTest(test.TestCase):
    def _test_impl(self, in_size, dtype, 
                   adjoint_a=False, adjoint_b=False, indices_dtype=np.int64):
        np.random.seed(127)
        x = []
        x = constant_op.constant(x, dtype=dtype)
        x = np.random.rand(in_size, in_size)
        x[np.abs(x) < 0.5] = 0  # Make it sparse
        y = []
        y = constant_op.constant(y, dtype=dtype)
        y = np.random.rand(in_size, in_size)
        flush_cache()
        x_mat = np.matrix(x)
        if adjoint_a:
            x_mat = x_mat.H
        y_mat = np.matrix(y)
        if adjoint_b:
            y_mat = y_mat.H

        x_indices = np.vstack(np.where(x)).astype(indices_dtype).T
        x_values = x[np.where(x)]
        x_shape = x.shape

        sp_x_value = sparse_tensor.SparseTensorValue(
        indices=x_indices, values=x_values, dense_shape=x_shape)
        tf_value_ans = sparse_ops.sparse_tensor_dense_matmul(
            sp_x_value, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
        tf_tensor_ans = sparse_ops.sparse_tensor_dense_matmul(
            sparse_tensor.SparseTensor.from_value(sp_x_value),
            y,
            adjoint_a=adjoint_a,
            adjoint_b=adjoint_b)    

    @add_profiling
    @multi_run(ITERATION)
    def testSparseTensorDenseMatMul(self):
        for dtype in FLOAT_COMPUTE_TYPE:
            # too large size may make memory allocation failure
            for in_size in [9, 10, 128, 129, 512, 513, 1023, 1024, 2048, 2049]:
                self._test_impl(in_size, dtype)

if __name__ == '__main__':
    test.main()   
