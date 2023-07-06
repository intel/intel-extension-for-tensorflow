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
from tensorflow.python.training import training_ops
from utils import multi_run, add_profiling, flush_cache

try:
    from intel_extension_for_tensorflow.python.test_func import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16, dtypes.bfloat16]
except ImportError:
    from tensorflow.python.platform import test
    FLOAT_COMPUTE_TYPE = [dtypes.float32, dtypes.float16]  # BF16 is not supported by CUDA  

INDICES_TYPE=[dtypes.int32] # , dtypes.int64]
type_from_model=[[128,1001],[1966080,3],[96,1001]]

ITERATION = 5

class SparseSoftmaxCrossEntropyWithLogitsTest(test.TestCase):
    def _test_impl(self, batch_size, num_entries, dtype, index_dtype):
        labels = tf.constant(np.random.randint(num_entries, size=batch_size), dtype=index_dtype)
        logits = tf.constant(np.random.randn(batch_size, num_entries), dtype=dtype)
        flush_cache()
        out_gpu = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="SequenceLoss/CrossEntropy")

    @add_profiling
    @multi_run(ITERATION)
    def testSparseSoftmaxCrossEntropyWithLogits(self):
        for index_dtype in INDICES_TYPE:
            for dtype in FLOAT_COMPUTE_TYPE:
                for batch_size in (32, 64, 128):
                    for num_entries in (100, 1000, 10000):
                        self._test_impl(batch_size, num_entries, dtype, index_dtype)
                for i in range(len(type_from_model)):
                    self._test_impl(type_from_model[i][0], type_from_model[i][1], dtype, index_dtype)

if __name__ == '__main__':
    test.main()