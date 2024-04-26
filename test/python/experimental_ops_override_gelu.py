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


import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
import intel_extension_for_tensorflow as itex
import numpy as np
import tensorflow as tf

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

SHAPE = (1, 16)
np.random.seed(1)
tf.compat.v1.disable_eager_execution()
class GELUTest(test_util.TensorFlowTestCase):
  """test GELU op"""

  @test_util.run_deprecated_v1
  def testGELU(self):
    x = tf.compat.v1.placeholder(tf.float32, shape=SHAPE)
    x_arr = np.reshape(np.random.normal(size=np.prod(SHAPE)) * 100,
            newshape=SHAPE)
    with self.session(use_gpu=True) as sess:
      tf_gelu = tf.nn.gelu(x)
      tf_result = sess.run(tf_gelu, feed_dict={x:x_arr})
      itex.experimental_ops_override()
      itex_gelu = tf.nn.gelu(x)
      itex_result = sess.run(itex_gelu, feed_dict={x:x_arr})
      self.assertAllClose(itex_result, tf_result, rtol=1e-2, atol=1e-2)

    from tensorflow.nn import gelu # pylint: disable=import-outside-toplevel
    assert gelu == itex.ops.gelu

if __name__ == "__main__":
  test.main()
