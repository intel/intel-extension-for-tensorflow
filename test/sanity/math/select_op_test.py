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

"""Select Op test"""
from intel_extension_for_tensorflow.python.test_func import test_util

from tensorflow import test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_math_ops


class SelectTest(test_util.TensorFlowTestCase):
    """test Select Op."""

    def testSelect(self):
        x = constant_op.constant([True, False])
        a = constant_op.constant([[1, 2], [3, 4]])
        b = constant_op.constant([[5, 6], [7, 8]])
        expected = constant_op.constant([[1, 2], [7, 8]])
        self.assertAllEqual(expected, gen_math_ops.select(x, a, b))


if __name__ == '__main__':
    test.main()
