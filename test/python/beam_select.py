# Copyright (c) 2023 Intel Corporation
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
import intel_extension_for_tensorflow as itex

class BeamSelectTest(test_util.TensorFlowTestCase):
    """test layer normalization op"""

    def _testForwardPass(self, input_shape, next_shape, dtype, tol):
        beam_indices=[[3,1,2,0],[0,3,2,1]]
        k=tf.random.uniform(input_shape,dtype=dtype)
        first_tokens=tf.concat([k,k,k,k],axis=1)
        next_tokens=tf.random.uniform(next_shape,dtype=dtype)
        cache = tf.concat([first_tokens,next_tokens],axis=3) if next_shape[3] != 0 else first_tokens

        result = tf.gather(params=cache, indices=beam_indices, axis=1, batch_dims=1)
        output=itex.ops.beam_select_kv_cache(cache,beam_indices,input_length=input_shape[3])
        # We use absolute tolerances in addition to relative tolerances, because
        # some of the values are very close to zero.
        self.assertAllClose(output, result, rtol=tol, atol=tol)

    def testRestForward(self):
        for i in [(tf.float32,1e-6),(tf.float16,1e-2),(tf.bfloat16,1e-2)]:
            d,t=i
            self._testForwardPass((2,1,16,1024,256), (2,4,16,0,256),d,t)
            self._testForwardPass((2,1,16,1024,256), (2,4,16,1,256),d,t)
            self._testForwardPass((2,1,16,1024,256), (2,4,16,4,256),d,t)


if __name__ == "__main__":
    test.main()
