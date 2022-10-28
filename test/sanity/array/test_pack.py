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



import tensorflow as tf
from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_array_ops


class PackTest(test_util.TensorFlowTestCase):
    """test Pack / Unpack op"""

    def _packHelper(self, dtype):
        with self.session(force_gpu=True):
            x = constant_op.constant([1, 4], dtype=dtype)
            y = constant_op.constant([2, 5], dtype=dtype)
            z = constant_op.constant([3, 6], dtype=dtype)
            pack_0 = constant_op.constant([[1, 4],
                                           [2, 5],
                                           [3, 6]], dtype=dtype)
            pack_1 = constant_op.constant([[1, 2, 3],
                                           [4, 5, 6]], dtype=dtype)
            self.assertAllEqual(pack_0, gen_array_ops.pack([x, y, z]))
            self.assertAllEqual(pack_1, gen_array_ops.pack([x, y, z], axis=1))

    def testPack(self):
        for dtype in [tf.float32, tf.bfloat16, tf.float16, tf.int32, tf.int64]:
            self._packHelper(dtype)

    def _unpackHelper(self, dtype):
        with self.cached_session(force_gpu=True):
            pack = constant_op.constant([[1, 4],
                                         [2, 5],
                                         [3, 6]], dtype=dtype)

            unpack_0_0 = constant_op.constant([1, 4], dtype=dtype)
            unpack_0_1 = constant_op.constant([2, 5], dtype=dtype)
            unpack_0_2 = constant_op.constant([3, 6], dtype=dtype)
            a, b, c = self.evaluate(gen_array_ops.unpack(pack, 3))
            self.assertAllEqual(unpack_0_0, a)
            self.assertAllEqual(unpack_0_1, b)
            self.assertAllEqual(unpack_0_2, c)

            unpack_1_0 = constant_op.constant([1, 2, 3], dtype=dtype)
            unpack_1_1 = constant_op.constant([4, 5, 6], dtype=dtype)
            d, e = self.evaluate(gen_array_ops.unpack(pack, 2, axis=1))
            self.assertAllEqual(unpack_1_0, d)
            self.assertAllEqual(unpack_1_1, e)

    def testUnpack(self):
        for dtype in [tf.float32, tf.bfloat16, tf.float16]:
            self._unpackHelper(dtype)


if __name__ == '__main__':
    test.main()
