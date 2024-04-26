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
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_training_ops as training_ops
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA  

INDICES_TYPE=[dtypes.int32] # , dtypes.int64]
slice_1d_list=[1024,16384*16,4096,8193,256,15,8]
in_size_list = [[8192],[16384*16383],[8192, 8192*2], [16384, 32], [8193, 64],[32,16,512,512],[16, 16384]]

ITERATION = 3


class SparseApplyAdagradV2Test(test.TestCase):
    def _test_impl(self, ref_size, dtype, index_dtype, slice_1d):
        x = resource_variable_ops.ResourceVariable(np.random.normal(size=ref_size), dtype=dtype) 
        y = resource_variable_ops.ResourceVariable(np.random.normal(size=ref_size), dtype=dtype) 
        lr = tf.constant(2.0, dtype=dtype)
        epsicon = tf.constant(1e-6, dtype=dtype)
        indices = np.random.choice(ref_size[0], slice_1d, replace=False)
        indices_con = tf.constant(indices, dtype=index_dtype)
        update_size= ref_size[:]
        update_size[0]=slice_1d
        updates = np.random.normal(size=update_size)
        updates_con = tf.expand_dims(tf.constant(updates, dtype=dtype),-1)
        flush_cache()
        out_gpu = training_ops.resource_sparse_apply_adagrad_v2(x.handle, y.handle, lr, epsicon, updates_con, indices_con)

    @add_profiling
    @multi_run(ITERATION)
    def testSparseApplyAdagradV2(self):
        for index_dtype in INDICES_TYPE:
            for dtype in FLOAT_COMPUTE_TYPE:
                for i in range(len(in_size_list)): 
                    self._test_impl(in_size_list[i], dtype, index_dtype, slice_1d_list[i])

if __name__ == '__main__':
    test.main()