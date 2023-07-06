# Copyright (c) 2023 Intel Corporation
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for cast_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

@test_util.run_all_in_native_and_block_format
class CastTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testCast(self):
    with self.session(use_gpu=True) as sess:
      inputs = constant_op.constant([[[[-0.20470765, 0.47894335, -0.51943874],
                                       [-0.5557303,  1.9657806,   1.3934058 ],
                                       [ 0.09290788, 0.28174615,  0.7690226 ]],
                                      [[ 1.2464347,  1.0071894,  -1.2962211 ],
                                       [ 0.27499163, 0.22891288,  1.3529168 ],
                                       [ 0.88642937,-2.0016372,  -0.37184253]]]])
      weights = constant_op.constant([[[[ 1.6690253, -0.43856972, -0.53974146],
                                        [ 0.476985,   3.2489438,  -1.0212275 ],
                                        [-0.5770873,  0.12412128,  0.30261356]]]])
      scale = constant_op.constant([[[[0.523438, 0.000938416, 1.34375],
                                      [-0.714844, -0.832031, -2.375],
                                      [-1.85938, -0.859375, 0.558594]],
                                     [[-1.26562, 0.119629, -1.0625],
                                      [0.332031, -2.35938, -0.199219],
                                      [-1.53906, -0.972656, -1.30469]]]],
                                    dtype=dtypes.bfloat16)
      expected = constant_op.constant([[[[0.0976562, 0.0014801, -0.71875],
                                         [0.566406, -5.65625, 3.0625],
                                         [0.287109, -0.832031, -0.0585938]],
                                        [[-4.1875, 0.306641, 2.21875],
                                         [-0.0708008, -1.86719, -0.00543213],
                                         [-1.13281, 6.75, -1.89844]]]],
                                      dtype=dtypes.bfloat16)

      conv = nn_ops.conv2d(
          inputs, filter=weights, strides=[1, 1, 1, 1], padding="VALID")
      cast = math_ops.cast(conv, dtype=dtypes.bfloat16)
      actual = self.evaluate(array_ops.identity(cast * scale))

      self.assertAllClose(expected, actual)


if __name__ == "__main__":
  test.main()
