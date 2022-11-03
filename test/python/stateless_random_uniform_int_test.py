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

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_stateless_random_ops

class StatelessRandomUniformIntTest(test_util.TensorFlowTestCase):
  """test StatelessRandomUniformInt op"""

  def testStatelessRandomUniformInt(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      shape = constant_op.constant((2, 5), dtype=dtype)
      seed = constant_op.constant([1,2], dtype=dtype)
      minval = constant_op.constant(2, dtype=dtype)
      maxval = constant_op.constant(7, dtype=dtype)
      output = gen_stateless_random_ops.stateless_random_uniform_int(
        shape, seed, minval, maxval)
      self.assertEqual(output.dtype, dtype)
    
if __name__ == "__main__":
  test.main()
