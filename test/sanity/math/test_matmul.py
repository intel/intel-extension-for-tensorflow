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
import tensorflow as tf


SHAPE1 = [64, 16]
SHAPE2 = [16, 64]

np.random.seed(1)
input_1 = np.reshape(np.random.normal(size=np.prod(SHAPE1)), newshape=SHAPE1)
input_2 = np.reshape(np.random.normal(size=np.prod(SHAPE2)), newshape=SHAPE2)


class MatmulTest(test_util.TensorFlowTestCase):
  """test Matmul op."""

  @test_util.run_deprecated_v1
  def testMatmulFp16(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")

    with self.session(use_gpu=False):
        ans_cpu = self.evaluate(tf.matmul(input_1, input_2))

    with self.session(use_gpu=True):
        dtype = tf.float16  # TODO(itex): bf16 accuracy issue
        x1 = tf.cast(input_1, dtype)
        x2 = tf.cast(input_2, dtype)
        y_gpu = self.evaluate(tf.matmul(x1, x2))

    self.assertAllClose(tf.cast(ans_cpu, dtype), y_gpu, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test.main()
