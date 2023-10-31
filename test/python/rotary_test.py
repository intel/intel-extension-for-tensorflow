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

def rotate_every_two(x: tf.Tensor) -> tf.Tensor:
    rotate_half_tensor = tf.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), axis=-1)
    new_shape = rotate_half_tensor.get_shape().as_list()[:-2] + [tf.math.reduce_prod(rotate_half_tensor.get_shape().as_list()[-2:])]
    rotate_half_tensor = tf.reshape(rotate_half_tensor, new_shape)
    return rotate_half_tensor

def apply_rotary_pos_emb(tensor,sin,cos):
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

class RotaryTest(test_util.TensorFlowTestCase):
    """test layer normalization op"""

    def _testForwardPass(self, input_shape, sin_shape, dtype, tol):

        q=tf.random.uniform(input_shape,dtype=dtype)
        k=tf.random.uniform(input_shape,dtype=dtype)

        sin=tf.random.uniform(sin_shape,dtype=dtype)
        cos=tf.random.uniform(sin_shape,dtype=dtype)

        k_rot = k[:, :, :, : 64]
        k_pass = k[:, :, :, 64 :]

        q_rot = q[:, :, :, : 64]
        q_pass = q[:, :, :, 64 :]

        k_rot = apply_rotary_pos_emb(k_rot, sin,cos)
        q_rot = apply_rotary_pos_emb(q_rot, sin,cos)

        result_k = tf.concat((k_rot, k_pass), axis=-1)
        result_q = tf.concat((q_rot, q_pass), axis=-1)
        output_q,output_k=itex.ops.qk_rotary_positional_embedding(q,k,sin,cos,rotary_dim=64,num_attention_heads=16,head_dim=256)
        # We use absolute tolerances in addition to relative tolerances, because
        # some of the values are very close to zero.
        self.assertAllClose(output_q, result_q, rtol=tol, atol=tol)
        self.assertAllClose(output_k, result_k, rtol=tol, atol=tol)

    def testRestForward(self):
        for i in [(tf.float32,1e-6),(tf.float16,1e-2),(tf.bfloat16,1e-2)]:
            d,t=i
            self._testForwardPass((4,1,16,256), (1,1,1,64),d,t)
            self._testForwardPass((12,1,16,256), (12,1,1,64),d,t)
    def testFirstForward(self):
        for i in [(tf.float32,1e-6),(tf.float16,1e-2),(tf.bfloat16,1e-2)]:
            d,t=i
            self._testForwardPass((1,1024,16,256), (1,1024,1,64),d,t)
            self._testForwardPass((3,1024,16,256), (3,1024,1,64),d,t)


if __name__ == "__main__":
    test.main()
