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
from tensorflow.python.ops import gen_stateless_random_ops_v2

class StatelessRandomGetKeyCounterAlgTest(test_util.TensorFlowTestCase):
  """test StatelessRandomGetKeyCounterAlg op"""

  def testStatelessRandomGetKeyCounterAlg(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      seed_t = constant_op.constant([0x12345678, 0xabcdef1], dtype=dtype)
      key, counter, alg = (gen_stateless_random_ops_v2.
                          stateless_random_get_key_counter_alg(seed_t))
      self.assertEqual(key.dtype, dtypes.uint64)
      self.assertEqual(counter.dtype, dtypes.uint64)
      self.assertEqual(alg.dtype, dtypes.int32)
    
if __name__ == "__main__":
  test.main()
