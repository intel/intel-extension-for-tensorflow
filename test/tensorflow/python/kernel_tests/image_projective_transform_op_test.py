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
"""Tests for tensorflow.raw_ops.ImageProjectiveTransformV2."""

import numpy as np

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed

class ImageProjectiveTransformV2Test(test.TestCase):

  def _base_test(self, images, transforms,
                 output_shape, interpolation, fill_mode, use_gpu):
    with test_util.device(use_gpu=use_gpu):
      output = tf.raw_ops.ImageProjectiveTransformV2(
        images = images,
        transforms = transforms,
        output_shape = output_shape,
        interpolation = interpolation,
        fill_mode = fill_mode)
      return output

  def testImageProjectiveTransformV2(self):
    if not test.is_gpu_available():
      self.skipTest("Need XPU for testing.")
    random_seed.set_random_seed(6)

    transform_matrix_list = [
      np.asarray([[1., 0., 0., 0., 1., -1., 0., 0.]], dtype = np.float32), # down shift by 1
      np.asarray([[1., 0., 0., 0., 1., 1., 0., 0.]], dtype = np.float32), # up shift by 1
      np.asarray([[1., 0., 1., 0., 1., 0., 0., 0.]], dtype = np.float32), # left shift by 1
      np.asarray([[1., 0., -1., 0., 1., 0., 0., 0.]], dtype = np.float32), # right shift by 1
    ]
    dtypes = [
      np.uint8,
      np.int32,
      np.int64,
      np.float16,
      np.float32,
      np.float64
    ]
    test_shapes = [
      ([1, 640, 480, 3], [1920, 1080]),
      ([2, 1920, 1080, 3], [640, 480]),
    ]
    interpolations = ["NEAREST", "BILINEAR"]
    fill_modes = ["CONSTANT", "REFLECT", "WRAP", "CONSTANT"]
    for interpolation in interpolations:
      for fill_mode in fill_modes:
        for (input_shape, output_shape) in test_shapes:
          for transforms in transform_matrix_list:
            transforms = ops.convert_to_tensor(transforms)
            for dtype in dtypes:
              image = np.random.randint(0, 256, input_shape).astype(dtype)
              image = ops.convert_to_tensor(image)
              ref_out = self._base_test(
                image,transforms, output_shape, interpolation, fill_mode, False)
              out = self._base_test(
                image, transforms, output_shape, interpolation, fill_mode, True)
              self.assertAllClose(out, ref_out, atol=1e-6)

if __name__ == "__main__":
  test.main()
