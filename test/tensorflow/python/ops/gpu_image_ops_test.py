# Copyright (c) 2022 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


"""Tests for tensorflow.ops.image_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import colorsys
import functools
import itertools
import math
import os
import time

from absl.testing import parameterized
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.ops import get_single_element
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

class ResizeBilinearBenchmark(test.Benchmark):
  
  def _benchmarkResize(self, image_size, num_channels):
    batch_size = 1
    num_ops = 1000
    img = variables.Variable(
        random_ops.random_normal(
            [batch_size, image_size[0], image_size[1], num_channels]),
        name="img")

    deps = []
    for _ in xrange(num_ops):
      with ops.control_dependencies(deps):
        resize_op = image_ops.resize_bilinear(
            img, [299, 299], align_corners=False)
        deps = [resize_op]
      benchmark_op = control_flow_ops.group(*deps)

    with self.benchmark_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      results = self.run_op_benchmark(
          sess,
          benchmark_op,
          name=("resize_bilinear_%s_%s_%s" % (image_size[0], image_size[1],
                                              num_channels)))
      print("%s   : %.2f ms/img" %
            (results["name"],
             1000 * results["wall_time"] / (batch_size * num_ops)))

  def benchmarkSimilar3Channel(self):
    self._benchmarkResize((183, 229), 3)

  def benchmarkScaleUp3Channel(self):
    self._benchmarkResize((141, 186), 3)

  def benchmarkScaleDown3Channel(self):
    self._benchmarkResize((749, 603), 3)

  def benchmarkSimilar1Channel(self):
    self._benchmarkResize((183, 229), 1)

  def benchmarkScaleUp1Channel(self):
    self._benchmarkResize((141, 186), 1)

  def benchmarkScaleDown1Channel(self):
    self._benchmarkResize((749, 603), 1)

class ResizeImagesV2Test(test_util.TensorFlowTestCase, parameterized.TestCase):
  
  METHODS = [
      image_ops.ResizeMethod.BILINEAR, image_ops.ResizeMethod.NEAREST_NEIGHBOR,
      image_ops.ResizeMethod.BICUBIC, image_ops.ResizeMethod.AREA,
      image_ops.ResizeMethod.LANCZOS3, image_ops.ResizeMethod.LANCZOS5,
      image_ops.ResizeMethod.GAUSSIAN, image_ops.ResizeMethod.MITCHELLCUBIC
  ]

  # Some resize methods, such as Gaussian, are non-interpolating in that they
  # change the image even if there is no scale change, for some test, we only
  # check the value on the value preserving methods.
  INTERPOLATING_METHODS = [
      image_ops.ResizeMethod.BILINEAR, image_ops.ResizeMethod.NEAREST_NEIGHBOR,
      image_ops.ResizeMethod.BICUBIC, image_ops.ResizeMethod.AREA,
      image_ops.ResizeMethod.LANCZOS3, image_ops.ResizeMethod.LANCZOS5
  ]

  TYPES = [
      np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.float16,
      np.float32, np.float64
  ]

  def _assertShapeInference(self, pre_shape, size, post_shape):
    # Try single image resize
    single_image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.resize_images_v2(single_image, size)
    self.assertEqual(y.get_shape().as_list(), post_shape)
    # Try batch images resize with known batch size
    images = array_ops.placeholder(dtypes.float32, shape=[99] + pre_shape)
    y = image_ops.resize_images_v2(images, size)
    self.assertEqual(y.get_shape().as_list(), [99] + post_shape)
    # Try batch images resize with unknown batch size
    images = array_ops.placeholder(dtypes.float32, shape=[None] + pre_shape)
    y = image_ops.resize_images_v2(images, size)
    self.assertEqual(y.get_shape().as_list(), [None] + post_shape)

  # XLA doesn't implement half_pixel_centers
  @test_util.disable_xla("b/127616992")
  def testLegacyBicubicMethodsMatchNewMethods(self):
    img_shape = [1, 3, 2, 1]
    data = [64, 32, 32, 64, 50, 100]
    target_height = 6
    target_width = 4
    methods_to_test = ((gen_image_ops.resize_bilinear, "triangle"),
                       (gen_image_ops.resize_bicubic, "keyscubic"))
    for legacy_method, new_method in methods_to_test:
      with self.cached_session(use_gpu=True):
        img_np = np.array(data, dtype=np.float32).reshape(img_shape)
        image = constant_op.constant(img_np, shape=img_shape)
        legacy_result = legacy_method(
            image,
            constant_op.constant([target_height, target_width],
                                 dtype=dtypes.int32),
            half_pixel_centers=True)
        scale = (
            constant_op.constant([target_height, target_width],
                                 dtype=dtypes.float32) /
            math_ops.cast(array_ops.shape(image)[1:3], dtype=dtypes.float32))
        new_result = gen_image_ops.scale_and_translate(
            image,
            constant_op.constant([target_height, target_width],
                                 dtype=dtypes.int32),
            scale,
            array_ops.zeros([2]),
            kernel_type=new_method,
            antialias=False)
        self.assertAllClose(
            self.evaluate(legacy_result), self.evaluate(new_result), atol=1e-04)
  
class AdjustContrastTest(test_util.TensorFlowTestCase):

  def _testContrast(self, x_np, y_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
      self.assertAllClose(y_tf, y_np, 1e-6)

  def testDoubleContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 62, 169, 255, 28, 0, 255, 135, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testDoubleContrastFloat(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float).reshape(x_shape) / 255.

    y_data = [
        -45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
        134.75, 409.25, -116.5
    ]
    y_np = np.array(y_data, dtype=np.float).reshape(x_shape) / 255.

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testHalfContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [22, 52, 65, 49, 118, 172, 41, 54, 176, 67, 178, 59]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=0.5)

  def testBatchDoubleContrast(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 81, 200, 255, 10, 0, 255, 116, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def _adjustContrastNp(self, x_np, contrast_factor):
    mean = np.mean(x_np, (1, 2), keepdims=True)
    y_np = mean + contrast_factor * (x_np - mean)
    return y_np

  def _adjustContrastTf(self, x_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
    return y_tf

  def testRandomContrast(self):
    x_shapes = [
        [1, 2, 2, 3],
        [2, 1, 2, 3],
        [1, 2, 2, 3],
        [2, 5, 5, 3],
        [2, 1, 1, 3],
    ]
    for x_shape in x_shapes:
      x_np = np.random.rand(*x_shape) * 255.
      contrast_factor = np.random.rand() * 2.0 + 0.1
      y_np = self._adjustContrastNp(x_np, contrast_factor)
      y_tf = self._adjustContrastTf(x_np, contrast_factor)
      self.assertAllClose(y_tf, y_np, rtol=1e-5, atol=1e-5)

  def testContrastFactorShape(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "contrast_factor must be scalar|"
                                "Shape must be rank 0 but is rank 1"):
      image_ops.adjust_contrast(x_np, [2.0])

class AdjustSaturationTest(test_util.TensorFlowTestCase):

  def testHalfSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testTwiceSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 2.0
    y_data = [0, 5, 13, 0, 106, 226, 30, 0, 234, 89, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBatchSaturation(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def _adjustSaturationNp(self, x_np, scale):
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in xrange(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      s *= scale
      s = min(1.0, max(0.0, s))
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def testAdjustRandomSaturation(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    with self.cached_session(use_gpu=True):
      for x_shape in x_shapes:
        for test_style in test_styles:
          x_np = np.random.rand(*x_shape) * 255.
          scale = np.random.rand()
          if test_style == "all_random":
            pass
          elif test_style == "rg_same":
            x_np[..., 1] = x_np[..., 0]
          elif test_style == "rb_same":
            x_np[..., 2] = x_np[..., 0]
          elif test_style == "gb_same":
            x_np[..., 2] = x_np[..., 1]
          elif test_style == "rgb_same":
            x_np[..., 1] = x_np[..., 0]
            x_np[..., 2] = x_np[..., 0]
          else:
            raise AssertionError("Invalid test style: %s" % (test_style))
          y_baseline = self._adjustSaturationNp(x_np, scale)
          y_fused = self.evaluate(image_ops.adjust_saturation(x_np, scale))
          self.assertAllClose(y_fused, y_baseline, rtol=2e-5, atol=1e-5)

class AdjustContrastTest(test_util.TensorFlowTestCase):

  def _testContrast(self, x_np, y_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
      self.assertAllClose(y_tf, y_np, 1e-6)

  def testDoubleContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 62, 169, 255, 28, 0, 255, 135, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testDoubleContrastFloat(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float).reshape(x_shape) / 255.

    y_data = [
        -45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
        134.75, 409.25, -116.5
    ]
    y_np = np.array(y_data, dtype=np.float).reshape(x_shape) / 255.

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testDoubleContrastDouble(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float64).reshape(x_shape) / 255.

    y_data = [
        -45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
        134.75, 409.25, -116.5
    ]
    y_np = np.array(y_data, dtype=np.float64).reshape(x_shape) / 255.

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testHalfContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [22, 52, 65, 49, 118, 172, 41, 54, 176, 67, 178, 59]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=0.5)

  def testBatchDoubleContrast(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 81, 200, 255, 10, 0, 255, 116, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def _adjustContrastNp(self, x_np, contrast_factor):
    mean = np.mean(x_np, (1, 2), keepdims=True)
    y_np = mean + contrast_factor * (x_np - mean)
    return y_np

  def _adjustContrastTf(self, x_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
    return y_tf

  def testRandomContrast(self):
    x_shapes = [
        [1, 2, 2, 3],
        [2, 1, 2, 3],
        [1, 2, 2, 3],
        [2, 5, 5, 3],
        [2, 1, 1, 3],
    ]
    for x_shape in x_shapes:
      x_np = np.random.rand(*x_shape) * 255.
      contrast_factor = np.random.rand() * 2.0 + 0.1
      y_np = self._adjustContrastNp(x_np, contrast_factor)
      y_tf = self._adjustContrastTf(x_np, contrast_factor)
      self.assertAllClose(y_tf, y_np, rtol=1e-5, atol=1e-5)

  def testContrastFactorShape(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "contrast_factor must be scalar|"
                                "Shape must be rank 0 but is rank 1"):
      image_ops.adjust_contrast(x_np, [2.0])

class AdjustHueTest(test_util.TensorFlowTestCase):

  def testAdjustNegativeHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testAdjustPositiveHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBatchAdjustHue(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def _adjustHueNp(self, x_np, delta_h):
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in xrange(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      h += delta_h
      h = math.fmod(h + 10.0, 1.0)
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def _adjustHueTf(self, x_np, delta_h):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_hue(x, delta_h)
      y_tf = self.evaluate(y)
    return y_tf

  def testAdjustRandomHue(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    for x_shape in x_shapes:
      for test_style in test_styles:
        x_np = np.random.rand(*x_shape) * 255.
        delta_h = np.random.rand() * 2.0 - 1.0
        if test_style == "all_random":
          pass
        elif test_style == "rg_same":
          x_np[..., 1] = x_np[..., 0]
        elif test_style == "rb_same":
          x_np[..., 2] = x_np[..., 0]
        elif test_style == "gb_same":
          x_np[..., 2] = x_np[..., 1]
        elif test_style == "rgb_same":
          x_np[..., 1] = x_np[..., 0]
          x_np[..., 2] = x_np[..., 0]
        else:
          raise AssertionError("Invalid test style: %s" % (test_style))
        y_np = self._adjustHueNp(x_np, delta_h)
        y_tf = self._adjustHueTf(x_np, delta_h)
        self.assertAllClose(y_tf, y_np, rtol=2e-5, atol=1e-5)

  def testInvalidShapes(self):
    fused = False
    if not fused:
      # The tests are known to pass with the fused adjust_hue. We will enable
      # them when the fused implementation is the default.
      return
    x_np = np.random.rand(2, 3) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    fused = False
    with self.assertRaisesRegex(ValueError, "Shape must be at least rank 3"):
      self._adjustHueTf(x_np, delta_h)
    x_np = np.random.rand(4, 2, 4) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError("input must have 3 channels"):
      self._adjustHueTf(x_np, delta_h)



  def _testContrast(self, x_np, y_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
      self.assertAllClose(y_tf, y_np, 1e-6)

  def testDoubleContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 62, 169, 255, 28, 0, 255, 135, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testDoubleContrastFloat(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float).reshape(x_shape) / 255.

    y_data = [
        -45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
        134.75, 409.25, -116.5
    ]
    y_np = np.array(y_data, dtype=np.float).reshape(x_shape) / 255.

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testHalfContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [22, 52, 65, 49, 118, 172, 41, 54, 176, 67, 178, 59]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=0.5)

  def testBatchDoubleContrast(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def _adjustHueNp(self, x_np, delta_h):
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in xrange(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      h += delta_h
      h = math.fmod(h + 10.0, 1.0)
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def _adjustHueTf(self, x_np, delta_h):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_hue(x, delta_h)
      y_tf = self.evaluate(y)
    return y_tf

  def testAdjustRandomHue(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    for x_shape in x_shapes:
      for test_style in test_styles:
        x_np = np.random.rand(*x_shape) * 255.
        delta_h = np.random.rand() * 2.0 - 1.0
        if test_style == "all_random":
          pass
        elif test_style == "rg_same":
          x_np[..., 1] = x_np[..., 0]
        elif test_style == "rb_same":
          x_np[..., 2] = x_np[..., 0]
        elif test_style == "gb_same":
          x_np[..., 2] = x_np[..., 1]
        elif test_style == "rgb_same":
          x_np[..., 1] = x_np[..., 0]
          x_np[..., 2] = x_np[..., 0]
        else:
          raise AssertionError("Invalid test style: %s" % (test_style))
        y_np = self._adjustHueNp(x_np, delta_h)
        y_tf = self._adjustHueTf(x_np, delta_h)
        self.assertAllClose(y_tf, y_np, rtol=2e-5, atol=1e-5)

  def testInvalidShapes(self):
    fused = False
    if not fused:
      # The tests are known to pass with the fused adjust_hue. We will enable
      # them when the fused implementation is the default.
      return
    x_np = np.random.rand(2, 3) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    fused = False
    with self.assertRaisesRegex(ValueError, "Shape must be at least rank 3"):
      self._adjustHueTf(x_np, delta_h)
    x_np = np.random.rand(4, 2, 4) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError("input must have 3 channels"):
      self._adjustHueTf(x_np, delta_h)

    y_data = [0, 0, 0, 81, 200, 255, 10, 0, 255, 116, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)
    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def _adjustContrastNp(self, x_np, contrast_factor):
    mean = np.mean(x_np, (1, 2), keepdims=True)
    y_np = mean + contrast_factor * (x_np - mean)
    return y_np

  def _adjustContrastTf(self, x_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
    return y_tf

  def testRandomContrast(self):
    x_shapes = [
        [1, 2, 2, 3],
        [2, 1, 2, 3],
        [1, 2, 2, 3],
        [2, 5, 5, 3],
        [2, 1, 1, 3],
    ]
    for x_shape in x_shapes:
      x_np = np.random.rand(*x_shape) * 255.
      contrast_factor = np.random.rand() * 2.0 + 0.1
      y_np = self._adjustContrastNp(x_np, contrast_factor)
      y_tf = self._adjustContrastTf(x_np, contrast_factor)
      self.assertAllClose(y_tf, y_np, rtol=1e-5, atol=1e-5)

  def testContrastFactorShape(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "contrast_factor must be scalar|"
                                "Shape must be rank 0 but is rank 1"):
      image_ops.adjust_contrast(x_np, [2.0])

class CombinedNonMaxSuppressionTest(test_util.TensorFlowTestCase):

  # NOTE(b/142795960): parameterized tests do not work well with tf.tensor
  # inputs. Due to failures, creating another test `testInvalidTensorInput`
  # which is identical to this one except that the input here is a scalar as
  # opposed to a tensor.
  def testInvalidPyInput(self):
    boxes_np = [[[[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]]]
    scores_np = [[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]
    max_output_size_per_class = 5
    max_total_size = 2**31
    with self.assertRaisesRegex(
        (TypeError, ValueError),
        "type int64 that does not match expected type of int32|"
        "Tensor conversion requested dtype int32 for Tensor with dtype int64"):
      image_ops.combined_non_max_suppression(
          boxes=boxes_np,
          scores=scores_np,
          max_output_size_per_class=max_output_size_per_class,
          max_total_size=max_total_size)

  # NOTE(b/142795960): parameterized tests do not work well with tf.tensor
  # inputs. Due to failures, creating another this test which is identical to
  # `testInvalidPyInput` except that the input is a tensor here as opposed
  # to a scalar.
  def testInvalidTensorInput(self):
    boxes_np = [[[[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]]]
    scores_np = [[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]
    max_output_size_per_class = 5
    max_total_size = ops.convert_to_tensor(2**31)
    with self.assertRaisesRegex(
        (TypeError, ValueError),
        "type int64 that does not match expected type of int32|"
        "Tensor conversion requested dtype int32 for Tensor with dtype int64"):
      image_ops.combined_non_max_suppression(
          boxes=boxes_np,
          scores=scores_np,
          max_output_size_per_class=max_output_size_per_class,
          max_total_size=max_total_size)


class NonMaxSuppressionTest(test_util.TensorFlowTestCase):

  def testNonMaxSuppression(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 3
    iou_threshold_np = 0.5
    with self.cached_session():
      boxes = constant_op.constant(boxes_np)
      scores = constant_op.constant(scores_np)
      max_output_size = constant_op.constant(max_output_size_np)
      iou_threshold = constant_op.constant(iou_threshold_np)
      selected_indices = image_ops.non_max_suppression(
          boxes, scores, max_output_size, iou_threshold)
      self.assertAllClose(selected_indices, [3, 0, 5])
      
  def testNonMaxSuppression(self):
    import tensorflow as tf
    import os
    size = (200, 4)
    boxes_np = np.random.random(size=(200, 4)).astype(np.float32)
    scores_np = np.random.random(size=(200,)).astype(np.float32)    
    max_output_size_np = 200
    iou_threshold_np = 0.5
    for score_threshold_np in [0, 0.0001, 0.2]:
      with test_util.device(use_gpu=True):
        boxes = constant_op.constant(boxes_np)
        scores = constant_op.constant(scores_np)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np)
        selected_indices, count = gen_image_ops.non_max_suppression_v4(
            boxes, scores, max_output_size, iou_threshold, score_threshold_np)
        
      with test_util.device(use_gpu=False):
        boxes = constant_op.constant(boxes_np)
        scores = constant_op.constant(scores_np)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np)
        selected_indices_cpu, count_cpu = gen_image_ops.non_max_suppression_v4(
            boxes, scores, max_output_size, iou_threshold, score_threshold_np)      
        self.assertAllEqual(count.numpy(), count_cpu.numpy())
        self.assertAllEqual(selected_indices.numpy(), selected_indices_cpu.numpy())
      

  def testInvalidShape(self):

    def nms_func(box, score, iou_thres, score_thres):
      return image_ops.non_max_suppression(box, score, iou_thres, score_thres)

    iou_thres = 3
    score_thres = 0.5

    # The boxes should be 2D of shape [num_boxes, 4].
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Shape must be rank 2 but is rank 1"):
      boxes = constant_op.constant([0.0, 0.0, 1.0, 1.0])
      scores = constant_op.constant([0.9])
      nms_func(boxes, scores, iou_thres, score_thres)

    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Dimension must be 4 but is 3"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0]])
      scores = constant_op.constant([0.9])
      nms_func(boxes, scores, iou_thres, score_thres)

    # The boxes is of shape [num_boxes, 4], and the scores is
    # of shape [num_boxes]. So an error will be thrown.
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Dimensions must be equal, but are 1 and 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9, 0.75])
      nms_func(boxes, scores, iou_thres, score_thres)

    # The scores should be 1D of shape [num_boxes].
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Shape must be rank 1 but is rank 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([[0.9]])
      nms_func(boxes, scores, iou_thres, score_thres)

    # The max_output_size should be a scalar (0-D).
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Shape must be rank 0 but is rank 1"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9])
      nms_func(boxes, scores, [iou_thres], score_thres)

    # The iou_threshold should be a scalar (0-D).
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "Shape must be rank 0 but is rank 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9])
      nms_func(boxes, scores, iou_thres, [[score_thres]])

  @test_util.xla_allow_fallback(
      "non_max_suppression with dynamic output shape unsupported.")
  def testDataTypes(self):
    # Test case for GitHub issue 20199.
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 3
    iou_threshold_np = 0.5
    score_threshold_np = float("-inf")
    # Note: There are multiple versions of non_max_suppression v2, v3, v4.
    # gen_image_ops.non_max_suppression_v2:
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        selected_indices = gen_image_ops.non_max_suppression_v2(
            boxes, scores, max_output_size, iou_threshold)
        selected_indices = self.evaluate(selected_indices)
        self.assertAllClose(selected_indices, [3, 0, 5])
    # gen_image_ops.non_max_suppression_v3
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        selected_indices = gen_image_ops.non_max_suppression_v3(
            boxes, scores, max_output_size, iou_threshold, score_threshold)
        selected_indices = self.evaluate(selected_indices)
        self.assertAllClose(selected_indices, [3, 0, 5])
    # gen_image_ops.non_max_suppression_v4.
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        selected_indices, _ = gen_image_ops.non_max_suppression_v4(
            boxes, scores, max_output_size, iou_threshold, score_threshold)
        selected_indices = self.evaluate(selected_indices)
        self.assertAllClose(selected_indices, [3, 0, 5])
    # gen_image_ops.non_max_suppression_v5.
    soft_nms_sigma_np = float(0.0)
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        soft_nms_sigma = constant_op.constant(soft_nms_sigma_np, dtype=dtype)
        selected_indices, _, _ = gen_image_ops.non_max_suppression_v5(
            boxes, scores, max_output_size, iou_threshold, score_threshold,
            soft_nms_sigma)
        selected_indices = self.evaluate(selected_indices)
        self.assertAllClose(selected_indices, [3, 0, 5])

  def testZeroIOUThreshold(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [1., 1., 1., 1., 1., 1.]
    max_output_size_np = 3
    iou_threshold_np = 0.0
    with self.cached_session():
      boxes = constant_op.constant(boxes_np)
      scores = constant_op.constant(scores_np)
      max_output_size = constant_op.constant(max_output_size_np)
      iou_threshold = constant_op.constant(iou_threshold_np)
      selected_indices = image_ops.non_max_suppression(
          boxes, scores, max_output_size, iou_threshold)
      self.assertAllClose(selected_indices, [0, 3, 5])


class NonMaxSuppressionWithScoresTest(test_util.TensorFlowTestCase):

  @test_util.xla_allow_fallback(
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromThreeClustersWithSoftNMS(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 6
    iou_threshold_np = 1.0
    score_threshold_np = 0.0
    soft_nms_sigma_np = 0.5
    boxes = constant_op.constant(boxes_np)
    scores = constant_op.constant(scores_np)
    max_output_size = constant_op.constant(max_output_size_np)
    iou_threshold = constant_op.constant(iou_threshold_np)
    score_threshold = constant_op.constant(score_threshold_np)
    soft_nms_sigma = constant_op.constant(soft_nms_sigma_np)
    selected_indices, selected_scores = \
        image_ops.non_max_suppression_with_scores(
            boxes,
            scores,
            max_output_size,
            iou_threshold,
            score_threshold,
            soft_nms_sigma)
    selected_indices, selected_scores = self.evaluate(
        [selected_indices, selected_scores])
    self.assertAllClose(selected_indices, [3, 0, 1, 5, 4, 2])
    self.assertAllClose(selected_scores,
                        [0.95, 0.9, 0.384, 0.3, 0.256, 0.197],
                        rtol=1e-2, atol=1e-2)


class NonMaxSuppressionPaddedTest(test_util.TensorFlowTestCase,
                                  parameterized.TestCase):

  @test_util.disable_xla(
      "b/141236442: "
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromThreeClustersV1(self):
    with ops.Graph().as_default():
      boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
      scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
      max_output_size_np = 5
      iou_threshold_np = 0.5
      boxes = constant_op.constant(boxes_np)
      scores = constant_op.constant(scores_np)
      max_output_size = constant_op.constant(max_output_size_np)
      iou_threshold = constant_op.constant(iou_threshold_np)
      selected_indices_padded, num_valid_padded = \
          image_ops.non_max_suppression_padded(
              boxes,
              scores,
              max_output_size,
              iou_threshold,
              pad_to_max_output_size=True)
      selected_indices, num_valid = image_ops.non_max_suppression_padded(
          boxes,
          scores,
          max_output_size,
          iou_threshold,
          pad_to_max_output_size=False)
      # The output shape of the padded operation must be fully defined.
      self.assertEqual(selected_indices_padded.shape.is_fully_defined(), True)
      self.assertEqual(selected_indices.shape.is_fully_defined(), False)
      with self.cached_session():
        self.assertAllClose(selected_indices_padded, [3, 0, 5, 0, 0])
        self.assertEqual(num_valid_padded.eval(), 3)
        self.assertAllClose(selected_indices, [3, 0, 5])
        self.assertEqual(num_valid.eval(), 3)

  @parameterized.named_parameters([("_RunEagerly", True), ("_RunGraph", False)])
  @test_util.disable_xla(
      "b/141236442: "
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromThreeClustersV2(self, run_func_eagerly):
    if not context.executing_eagerly() and run_func_eagerly:
      # Skip running tf.function eagerly in V1 mode.
      self.skipTest("Skip test that runs tf.function eagerly in V1 mode.")
    else:

      @def_function.function
      def func(boxes, scores, max_output_size, iou_threshold):
        boxes = constant_op.constant(boxes_np)
        scores = constant_op.constant(scores_np)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np)

        yp, nvp = image_ops.non_max_suppression_padded(
            boxes,
            scores,
            max_output_size,
            iou_threshold,
            pad_to_max_output_size=True)

        y, n = image_ops.non_max_suppression_padded(
            boxes,
            scores,
            max_output_size,
            iou_threshold,
            pad_to_max_output_size=False)

        # The output shape of the padded operation must be fully defined.
        self.assertEqual(yp.shape.is_fully_defined(), True)
        self.assertEqual(y.shape.is_fully_defined(), False)

        return yp, nvp, y, n

      boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
      scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
      max_output_size_np = 5
      iou_threshold_np = 0.5

      selected_indices_padded, num_valid_padded, selected_indices, num_valid = \
          func(boxes_np, scores_np, max_output_size_np, iou_threshold_np)

      with self.cached_session():
        with test_util.run_functions_eagerly(run_func_eagerly):
          self.assertAllClose(selected_indices_padded, [3, 0, 5, 0, 0])
          self.assertEqual(self.evaluate(num_valid_padded), 3)
          self.assertAllClose(selected_indices, [3, 0, 5])
          self.assertEqual(self.evaluate(num_valid), 3)

  @test_util.xla_allow_fallback(
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromContinuousOverLapV1(self):
    with ops.Graph().as_default():
      boxes_np = [[0, 0, 1, 1], [0, 0.2, 1, 1.2], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]]
      scores_np = [0.9, 0.75, 0.6, 0.5, 0.4, 0.3]
      max_output_size_np = 3
      iou_threshold_np = 0.5
      score_threshold_np = 0.1
      boxes = constant_op.constant(boxes_np)
      scores = constant_op.constant(scores_np)
      max_output_size = constant_op.constant(max_output_size_np)
      iou_threshold = constant_op.constant(iou_threshold_np)
      score_threshold = constant_op.constant(score_threshold_np)
      selected_indices, num_valid = image_ops.non_max_suppression_padded(
          boxes,
          scores,
          max_output_size,
          iou_threshold,
          score_threshold)
      # The output shape of the padded operation must be fully defined.
      self.assertEqual(selected_indices.shape.is_fully_defined(), False)
      with self.cached_session():
        self.assertAllClose(selected_indices, [0, 2, 4])
        self.assertEqual(num_valid.eval(), 3)

  @parameterized.named_parameters([("_RunEagerly", True), ("_RunGraph", False)])
  @test_util.xla_allow_fallback(
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromContinuousOverLapV2(self, run_func_eagerly):
    if not context.executing_eagerly() and run_func_eagerly:
      # Skip running tf.function eagerly in V1 mode.
      self.skipTest("Skip test that runs tf.function eagerly in V1 mode.")
    else:

      @def_function.function
      def func(boxes, scores, max_output_size, iou_threshold, score_threshold):
        boxes = constant_op.constant(boxes)
        scores = constant_op.constant(scores)
        max_output_size = constant_op.constant(max_output_size)
        iou_threshold = constant_op.constant(iou_threshold)
        score_threshold = constant_op.constant(score_threshold)

        y, nv = image_ops.non_max_suppression_padded(
            boxes, scores, max_output_size, iou_threshold, score_threshold)

        # The output shape of the padded operation must be fully defined.
        self.assertEqual(y.shape.is_fully_defined(), False)

        return y, nv

      boxes_np = [[0, 0, 1, 1], [0, 0.2, 1, 1.2], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]]
      scores_np = [0.9, 0.75, 0.6, 0.5, 0.4, 0.3]
      max_output_size_np = 3
      iou_threshold_np = 0.5
      score_threshold_np = 0.1
      selected_indices, num_valid = func(boxes_np, scores_np,
                                         max_output_size_np, iou_threshold_np,
                                         score_threshold_np)
      with self.cached_session():
        with test_util.run_functions_eagerly(run_func_eagerly):
          self.assertAllClose(selected_indices, [0, 2, 4])
          self.assertEqual(self.evaluate(num_valid), 3)

class NonMaxSuppressionWithOverlapsTest(test_util.TensorFlowTestCase):

  def testSelectOneFromThree(self):
    overlaps_np = [
        [1.0, 0.7, 0.2],
        [0.7, 1.0, 0.0],
        [0.2, 0.0, 1.0],
    ]
    scores_np = [0.7, 0.9, 0.1]
    max_output_size_np = 3

    overlaps = constant_op.constant(overlaps_np)
    scores = constant_op.constant(scores_np)
    max_output_size = constant_op.constant(max_output_size_np)
    overlap_threshold = 0.6
    score_threshold = 0.4

    selected_indices = image_ops.non_max_suppression_with_overlaps(
        overlaps, scores, max_output_size, overlap_threshold, score_threshold)

    with self.cached_session():
      self.assertAllClose(selected_indices, [1])

class DecodeImageTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  _FORWARD_COMPATIBILITY_HORIZONS = [
      (2020, 1, 1),
      (2020, 7, 14),
      (2525, 1, 1),  # future behavior
  ]

  def testImageCropAndResize(self):
    if test_util.is_gpu_available():
      op = image_ops_impl.crop_and_resize_v2(
          image=array_ops.zeros((2, 1, 1, 1)),
          boxes=[[1.0e+40, 0, 0, 0]],
          box_indices=[1],
          crop_size=[1, 1])
      self.evaluate(op)
    else:
      message = "Boxes contains at least one element that is not finite"
      with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                  message):
        op = image_ops_impl.crop_and_resize_v2(
            image=array_ops.zeros((2, 1, 1, 1)),
            boxes=[[1.0e+40, 0, 0, 0]],
            box_indices=[1],
            crop_size=[1, 1])
        self.evaluate(op)

if __name__ == "__main__":
  googletest.main()
