# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.cast."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import os
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

tf.compat.v1.disable_eager_execution()
# Test plain format.
os.environ['ITEX_LAYOUT_OPT']="0"

class CastTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testNativeCast(self):
        x = np.array([1.8, 1.1]).astype(dtypes.bfloat16.as_numpy_dtype)
        with self.session(use_gpu = False) as sess:
            itex_result = array_ops.identity(math_ops.cast(array_ops.identity(array_ops.identity(x)*2.)*1.,tf.float32)*2.)
            np_result = x.astype(np.float32)*4.
            self.assertAllEqual(np_result, itex_result)

if __name__ == "__main__":
  test.main()
