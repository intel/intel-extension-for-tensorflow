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

from intel_extension_for_tensorflow.python.test_func import test_util

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
from tensorflow.python.ops import nn_ops
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
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import googletest

@test_util.run_all_in_native_and_block_format
class ResizeImagesV2Test(test_util.TensorFlowTestCase, parameterized.TestCase):

  METHODS = [
      image_ops.ResizeMethod.BILINEAR,
      # image_ops.ResizeMethod.NEAREST_NEIGHBOR,
      # image_ops.ResizeMethod.BICUBIC, image_ops.ResizeMethod.AREA,
      # image_ops.ResizeMethod.LANCZOS3, image_ops.ResizeMethod.LANCZOS5,
      # image_ops.ResizeMethod.GAUSSIAN, image_ops.ResizeMethod.MITCHELLCUBIC
  ]

  # Some resize methods, such as Gaussian, are non-interpolating in that they
  # change the image even if there is no scale change, for some test, we only
  # check the value on the value preserving methods.
  INTERPOLATING_METHODS = [
      image_ops.ResizeMethod.BILINEAR,
      # image_ops.ResizeMethod.NEAREST_NEIGHBOR,
      # image_ops.ResizeMethod.BICUBIC, image_ops.ResizeMethod.AREA,
      # image_ops.ResizeMethod.LANCZOS3, image_ops.ResizeMethod.LANCZOS5
  ]

  TYPES = [
      # np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.float16,
      # np.float32, np.float64
      np.float32,
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

  def shouldRunOnGPU(self, method, nptype):
    if (method == image_ops.ResizeMethod.NEAREST_NEIGHBOR and
        nptype in [np.float32, np.float64]):
      return True
    else:
      return False

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testNoOp(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    target_height = 6
    target_width = 4

    for nptype in self.TYPES:
      img_np = np.array(data, dtype=nptype).reshape(img_shape)

      for method in self.METHODS:
        with self.cached_session():
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images_v2(image, [target_height, target_width],
                                         method)
          yshape = array_ops.shape(y)
          resized, newshape = self.evaluate([y, yshape])
          self.assertAllEqual(img_shape, newshape)
          if method in self.INTERPOLATING_METHODS:
            self.assertAllClose(resized, img_np, atol=1e-5)

      # Resizing with a single image must leave the shape unchanged also.
      with self.cached_session():
        img_single = img_np.reshape(single_shape)
        image = constant_op.constant(img_single, shape=single_shape)
        y = image_ops.resize_images_v2(image, [target_height, target_width],
                                       self.METHODS[0])
        yshape = array_ops.shape(y)
        newshape = self.evaluate(yshape)
        self.assertAllEqual(single_shape, newshape)

  # half_pixel_centers unsupported in ResizeBilinear
  @test_util.disable_xla("b/127616992")
  def testTensorArguments(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    def resize_func(t, new_size, method):
      return image_ops.resize_images_v2(t, new_size, method)

    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    for method in self.METHODS:
      with self.cached_session():
        image = constant_op.constant(img_np, shape=img_shape)
        y = resize_func(image, [6, 4], method)
        yshape = array_ops.shape(y)
        resized, newshape = self.evaluate([y, yshape])
        self.assertAllEqual(img_shape, newshape)
        if method in self.INTERPOLATING_METHODS:
          self.assertAllClose(resized, img_np, atol=1e-5)

      # Resizing with a single image must leave the shape unchanged also.
      with self.cached_session():
        img_single = img_np.reshape(single_shape)
        image = constant_op.constant(img_single, shape=single_shape)
        y = resize_func(image, [6, 4], self.METHODS[0])
        yshape = array_ops.shape(y)
        resized, newshape = self.evaluate([y, yshape])
        self.assertAllEqual(single_shape, newshape)
        if method in self.INTERPOLATING_METHODS:
          self.assertAllClose(resized, img_single, atol=1e-5)

    # Incorrect shape.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant(4)
      _ = resize_func(image, new_size, image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([4])
      _ = resize_func(image, new_size, image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([1, 2, 3])
      _ = resize_func(image, new_size, image_ops.ResizeMethod.BILINEAR)

    # Incorrect dtypes.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([6.0, 4])
      _ = resize_func(image, new_size, image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = resize_func(image, [6, 4.0], image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = resize_func(image, [None, 4], image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = resize_func(image, [6, None], image_ops.ResizeMethod.BILINEAR)

  def testReturnDtypeV1(self):
    # Shape inference in V1.
    with ops.Graph().as_default():
      target_shapes = [[6, 4], [3, 2],
                       [
                           array_ops.placeholder(dtypes.int32),
                           array_ops.placeholder(dtypes.int32)
                       ]]
      for nptype in self.TYPES:
        image = array_ops.placeholder(nptype, shape=[1, 6, 4, 1])
        for method in self.METHODS:
          for target_shape in target_shapes:
            y = image_ops.resize_images_v2(image, target_shape, method)
            if method == image_ops.ResizeMethod.NEAREST_NEIGHBOR:
              expected_dtype = image.dtype
            else:
              expected_dtype = dtypes.float32
            self.assertEqual(y.dtype, expected_dtype)

  @parameterized.named_parameters([("_RunEagerly", True), ("_RunGraph", False)])
  def testReturnDtypeV2(self, run_func_eagerly):
    if not context.executing_eagerly() and run_func_eagerly:
      # Skip running tf.function eagerly in V1 mode.
      self.skipTest("Skip test that runs tf.function eagerly in V1 mode.")
    else:

      @def_function.function
      def test_dtype(image, target_shape, target_method):
        y = image_ops.resize_images_v2(image, target_shape, target_method)
        if method == image_ops.ResizeMethod.NEAREST_NEIGHBOR:
          expected_dtype = image.dtype
        else:
          expected_dtype = dtypes.float32

        self.assertEqual(y.dtype, expected_dtype)

      target_shapes = [[6, 4],
                       [3, 2],
                       [tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
                        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)]]

      for nptype in self.TYPES:
        image = tensor_spec.TensorSpec(shape=[1, 6, 4, 1], dtype=nptype)
        for method in self.METHODS:
          for target_shape in target_shapes:
            with test_util.run_functions_eagerly(run_func_eagerly):
              test_dtype.get_concrete_function(image, target_shape, method)

  # half_pixel_centers not supported by XLA
  @test_util.disable_xla("b/127616992")
  @test_util.run_deprecated_v1
  def testSumTensor(self):
    img_shape = [1, 6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    # Test size where width is specified as a tensor which is a sum
    # of two tensors.
    width_1 = constant_op.constant(1)
    width_2 = constant_op.constant(3)
    width = math_ops.add(width_1, width_2)
    height = constant_op.constant(6)

    img_np = np.array(data, dtype=np.float32).reshape(img_shape)

    for method in self.METHODS:
      with self.cached_session(use_gpu=True):
        image = constant_op.constant(img_np, shape=img_shape)
        y = array_ops.identity(image_ops.resize_images_v2(image, [height, width], method))
        yshape = array_ops.shape(y)
        resized, newshape = self.evaluate([y, yshape])
        self.assertAllEqual(img_shape, newshape)
        if method in self.INTERPOLATING_METHODS:
          self.assertAllClose(resized, img_np, atol=1e-5)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testResizeDown(self):
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    expected_data = [127, 64, 64, 127, 50, 100]
    target_height = 3
    target_width = 2

    # Test out 3-D and 4-D image shapes.
    img_shapes = [[1, 6, 4, 1], [6, 4, 1]]
    target_shapes = [[1, target_height, target_width, 1],
                     [target_height, target_width, 1]]

    for target_shape, img_shape in zip(target_shapes, img_shapes):

      for nptype in self.TYPES:
        img_np = np.array(data, dtype=nptype).reshape(img_shape)

        for method in self.METHODS:
          with self.cached_session(use_gpu=True):
            image = constant_op.constant(img_np, shape=img_shape)
            y = image_ops.resize_images_v2(
                image, [target_height, target_width], method)
            expected = np.array(expected_data).reshape(target_shape)
            resized = self.evaluate(y)
            self.assertAllClose(resized, expected, atol=1e-5)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testResizeUp(self):
    img_shape = [1, 3, 2, 1]
    data = [64, 32, 32, 64, 50, 100]
    target_height = 6
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethod.BILINEAR] = [
        64.0, 56.0, 40.0, 32.0, 56.0, 52.0, 44.0, 40.0, 40.0, 44.0, 52.0, 56.0,
        36.5, 45.625, 63.875, 73.0, 45.5, 56.875, 79.625, 91.0, 50.0, 62.5,
        87.5, 100.0
    ]
# for other methods
#     expected_data[image_ops.ResizeMethod.NEAREST_NEIGHBOR] = [
#         64.0, 64.0, 32.0, 32.0, 64.0, 64.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0,
#         32.0, 32.0, 64.0, 64.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 100.0,
#         100.0
#     ]
#     expected_data[image_ops.ResizeMethod.AREA] = [
#         64.0, 64.0, 32.0, 32.0, 64.0, 64.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0,
#         32.0, 32.0, 64.0, 64.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 100.0,
#         100.0
#     ]
#     expected_data[image_ops.ResizeMethod.LANCZOS3] = [
#         75.8294, 59.6281, 38.4313, 22.23, 60.6851, 52.0037, 40.6454, 31.964,
#         35.8344, 41.0779, 47.9383, 53.1818, 24.6968, 43.0769, 67.1244, 85.5045,
#         35.7939, 56.4713, 83.5243, 104.2017, 44.8138, 65.1949, 91.8603, 112.2413
#     ]
#     expected_data[image_ops.ResizeMethod.LANCZOS5] = [
#         77.5699, 60.0223, 40.6694, 23.1219, 61.8253, 51.2369, 39.5593, 28.9709,
#         35.7438, 40.8875, 46.5604, 51.7041, 21.5942, 43.5299, 67.7223, 89.658,
#         32.1213, 56.784, 83.984, 108.6467, 44.5802, 66.183, 90.0082, 111.6109
#     ]
#     expected_data[image_ops.ResizeMethod.GAUSSIAN] = [
#         61.1087, 54.6926, 41.3074, 34.8913, 54.6926, 51.4168, 44.5832, 41.3074,
#         41.696, 45.2456, 52.6508, 56.2004, 39.4273, 47.0526, 62.9602, 70.5855,
#         47.3008, 57.3042, 78.173, 88.1764, 51.4771, 62.3638, 85.0752, 95.9619
#     ]
#     expected_data[image_ops.ResizeMethod.BICUBIC] = [
#         70.1453, 59.0252, 36.9748, 25.8547, 59.3195, 53.3386, 41.4789, 35.4981,
#         36.383, 41.285, 51.0051, 55.9071, 30.2232, 42.151, 65.8032, 77.731,
#         41.6492, 55.823, 83.9288, 98.1026, 47.0363, 62.2744, 92.4903, 107.7284
#     ]
#     expected_data[image_ops.ResizeMethod.MITCHELLCUBIC] = [
#         66.0382, 56.6079, 39.3921, 29.9618, 56.7255, 51.9603, 43.2611, 38.4959,
#         39.1828, 43.4664, 51.2864, 55.57, 34.6287, 45.1812, 64.4458, 74.9983,
#         43.8523, 56.8078, 80.4594, 93.4149, 48.9943, 63.026, 88.6422, 102.6739
#     ]
    for nptype in self.TYPES:
      for method in expected_data:
        with self.cached_session(use_gpu=True):
          img_np = np.array(data, dtype=nptype).reshape(img_shape)
          image = constant_op.constant(img_np, shape=img_shape)
          y = array_ops.identity(image_ops.resize_images_v2(image, [target_height, target_width],
                                         method))
          resized = self.evaluate(y)
          expected = np.array(expected_data[method]).reshape(
              [1, target_height, target_width, 1])
          self.assertAllClose(resized, expected, atol=1e-04)
 
#   # XLA doesn't implement half_pixel_centers
#   @test_util.disable_xla("b/127616992")
#   def testLegacyBicubicMethodsMatchNewMethods(self):
#     img_shape = [1, 3, 2, 1]
#     data = [64, 32, 32, 64, 50, 100]
#     target_height = 6
#     target_width = 4
#     methods_to_test = ((gen_image_ops.resize_bilinear, "triangle"),
#                        (gen_image_ops.resize_bicubic, "keyscubic"))
#     for legacy_method, new_method in methods_to_test:
#       with self.cached_session():
#         img_np = np.array(data, dtype=np.float32).reshape(img_shape)
#         image = constant_op.constant(img_np, shape=img_shape)
#         legacy_result = legacy_method(
#             image,
#             constant_op.constant([target_height, target_width],
#                                  dtype=dtypes.int32),
#             half_pixel_centers=True)
#         scale = (
#             constant_op.constant([target_height, target_width],
#                                  dtype=dtypes.float32) /
#             math_ops.cast(array_ops.shape(image)[1:3], dtype=dtypes.float32))
#         new_result = gen_image_ops.scale_and_translate(
#             image,
#             constant_op.constant([target_height, target_width],
#                                  dtype=dtypes.int32),
#             scale,
#             array_ops.zeros([2]),
#             kernel_type=new_method,
#             antialias=False)
#         self.assertAllClose(
#             self.evaluate(legacy_result), self.evaluate(new_result), atol=1e-04)
 
#   def testResizeDownArea(self):
#     img_shape = [1, 6, 6, 1]
#     data = [
#         128, 64, 32, 16, 8, 4, 4, 8, 16, 32, 64, 128, 128, 64, 32, 16, 8, 4, 5,
#         10, 15, 20, 25, 30, 30, 25, 20, 15, 10, 5, 5, 10, 15, 20, 25, 30
#     ]
#     img_np = np.array(data, dtype=np.uint8).reshape(img_shape)
# 
#     target_height = 4
#     target_width = 4
#     expected_data = [
#         73, 33, 23, 39, 73, 33, 23, 39, 14, 16, 19, 21, 14, 16, 19, 21
#     ]
# 
#     with self.cached_session():
#       image = constant_op.constant(img_np, shape=img_shape)
#       y = image_ops.resize_images_v2(image, [target_height, target_width],
#                                      image_ops.ResizeMethod.AREA)
#       expected = np.array(expected_data).reshape(
#           [1, target_height, target_width, 1])
#       resized = self.evaluate(y)
#       self.assertAllClose(resized, expected, atol=1)
# 
#   @test_util.disable_xla("align_corners=False not supported by XLA")
#   def testCompareNearestNeighbor(self):
#     if test.is_gpu_available():
#       input_shape = [1, 5, 6, 3]
#       target_height = 8
#       target_width = 12
#       for nptype in [np.float32, np.float64]:
#         img_np = np.arange(
#             0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
#         with self.cached_session():
#           image = constant_op.constant(img_np, shape=input_shape)
#           new_size = constant_op.constant([target_height, target_width])
#           out_op = image_ops.resize_images_v2(
#               image, new_size, image_ops.ResizeMethod.NEAREST_NEIGHBOR)
#           gpu_val = self.evaluate(out_op)
#         with self.cached_session(use_gpu=False):
#           image = constant_op.constant(img_np, shape=input_shape)
#           new_size = constant_op.constant([target_height, target_width])
#           out_op = image_ops.resize_images_v2(
#               image, new_size, image_ops.ResizeMethod.NEAREST_NEIGHBOR)
#           cpu_val = self.evaluate(out_op)
#         self.assertAllClose(cpu_val, gpu_val, rtol=1e-5, atol=1e-5)
# 
#   @test_util.disable_xla("align_corners=False not supported by XLA")
#   def testBfloat16MultipleOps(self):
#     target_height = 8
#     target_width = 12
#     img = np.random.uniform(0, 100, size=(30, 10, 2)).astype(np.float32)
#     img_bf16 = ops.convert_to_tensor(img, dtype="bfloat16")
#     new_size = constant_op.constant([target_height, target_width])
#     img_methods = [
#         image_ops.ResizeMethod.BILINEAR,
#         image_ops.ResizeMethod.NEAREST_NEIGHBOR, image_ops.ResizeMethod.BICUBIC,
#         image_ops.ResizeMethod.AREA
#     ]
#     for method in img_methods:
#       out_op_bf16 = image_ops.resize_images_v2(img_bf16, new_size, method)
#       out_op_f32 = image_ops.resize_images_v2(img, new_size, method)
#       bf16_val = self.evaluate(out_op_bf16)
#       f32_val = self.evaluate(out_op_f32)
#       self.assertAllClose(bf16_val, f32_val, rtol=1e-2, atol=1e-2)
 
  @test_util.run_deprecated_v1
  def testCompareBilinear(self):
    if test.is_gpu_available():
      input_shape = [1, 5, 6, 3]
      target_height = 8
      target_width = 12
      for nptype in [np.float32]:
        img_np = np.arange(
            0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
        value = {}
        for use_gpu in [True, False]:
          with self.cached_session(use_gpu=use_gpu):
            image = constant_op.constant(img_np, shape=input_shape)
            new_size = constant_op.constant([target_height, target_width])
            out_op = array_ops.identity(image_ops.resize_images_v2(image, new_size,
                                             image_ops.ResizeMethod.BILINEAR))
            value[use_gpu] = self.evaluate(out_op)
        self.assertAllClose(value[True], value[False], rtol=1e-5, atol=1e-5)

  def testShapeInference(self):
    # Shape function requires placeholders and a graph.
    with ops.Graph().as_default():
      self._assertShapeInference([50, 60, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([55, 66, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([59, 69, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([50, 69, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([59, 60, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([None, 60, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([None, 66, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([None, 69, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([50, None, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([55, None, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([59, None, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([None, None, 3], [55, 66], [55, 66, 3])
      self._assertShapeInference([50, 60, None], [55, 66], [55, 66, None])
      self._assertShapeInference([55, 66, None], [55, 66], [55, 66, None])
      self._assertShapeInference([59, 69, None], [55, 66], [55, 66, None])
      self._assertShapeInference([50, 69, None], [55, 66], [55, 66, None])
      self._assertShapeInference([59, 60, None], [55, 66], [55, 66, None])
      self._assertShapeInference([None, None, None], [55, 66], [55, 66, None])

  def testNameScope(self):
    # Testing name scope requires placeholders and a graph.
    with ops.Graph().as_default():
      with self.cached_session():
        single_image = array_ops.placeholder(dtypes.float32, shape=[50, 60, 3])
        y = image_ops.resize_images(single_image, [55, 66])
        self.assertTrue(y.op.name.startswith("resize"))

  def _ResizeImageCall(self, x, max_h, max_w, preserve_aspect_ratio,
                       use_tensor_inputs):
    if use_tensor_inputs:
      target_max = ops.convert_to_tensor([max_h, max_w])
      x_tensor = ops.convert_to_tensor(x)
    else:
      target_max = (max_h, max_w)
      x_tensor = x

    def resize_func(t,
                    target_max=target_max,
                    preserve_aspect_ratio=preserve_aspect_ratio):
      return image_ops.resize_images(
          t, ops.convert_to_tensor(target_max),
          preserve_aspect_ratio=preserve_aspect_ratio)

    with self.cached_session():
      return self.evaluate(resize_func(x_tensor))

  def _assertResizeEqual(self,
                         x,
                         x_shape,
                         y,
                         y_shape,
                         preserve_aspect_ratio=True,
                         use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageCall(x, target_height, target_width,
                                   preserve_aspect_ratio, use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertResizeCheckShape(self,
                              x,
                              x_shape,
                              target_shape,
                              y_shape,
                              preserve_aspect_ratio=True,
                              use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width = target_shape
    x = np.array(x).reshape(x_shape)
    y = np.zeros(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageCall(x, target_height, target_width,
                                   preserve_aspect_ratio, use_tensor_inputs)
      self.assertShapeEqual(y, ops.convert_to_tensor(y_tf))

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioMultipleImages(self):
    x_shape = [10, 100, 80, 10]
    x = np.random.uniform(size=x_shape)
    for preserve_aspect_ratio in [True, False]:
      with self.subTest(preserve_aspect_ratio=preserve_aspect_ratio):
        expect_shape = [10, 250, 200, 10] if preserve_aspect_ratio \
            else [10, 250, 250, 10]
        self._assertResizeCheckShape(
            x,
            x_shape, [250, 250],
            expect_shape,
            preserve_aspect_ratio=preserve_aspect_ratio)
 
  @test_util.run_deprecated_v1
  def testPreserveAspectRatioNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape).astype('f')

    self._assertResizeEqual(x, x_shape, x, x_shape)
 
  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSmaller(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape).astype('f')

    self._assertResizeCheckShape(x, x_shape, [75, 50], [50, 50, 10])
 
  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSmallerMultipleImages(self):
    x_shape = [10, 100, 100, 10]
    x = np.random.uniform(size=x_shape).astype('f')

    self._assertResizeCheckShape(x, x_shape, [75, 50], [10, 50, 50, 10])
 
  @test_util.run_deprecated_v1
  def testPreserveAspectRatioLarger(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape).astype('f')

    self._assertResizeCheckShape(x, x_shape, [150, 200], [150, 150, 10])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSameRatio(self):
    x_shape = [1920, 1080, 3]
    x = np.random.uniform(size=x_shape).astype('f')

    self._assertResizeCheckShape(x, x_shape, [3840, 2160], [3840, 2160, 3])
 
  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSquare(self):
    x_shape = [299, 299, 3]
    x = np.random.uniform(size=x_shape).astype('f')

    self._assertResizeCheckShape(x, x_shape, [320, 320], [320, 320, 3])

  
class ResizeBilinearGradOpTest(test.TestCase, parameterized.TestCase):

  def _itGen(self, smaller_shape, larger_shape):
    up_sample = (smaller_shape, larger_shape)
    down_sample = (larger_shape, smaller_shape)
    pass_through = (larger_shape, larger_shape)
    shape_pairs = (up_sample, down_sample, pass_through)
    # Align corners is deprecated in TF2.0, but align_corners==False is not
    # supported by XLA.
    # options = [(True, False)]
    # if not test_util.is_xla_enabled():
    #   options += [(False, True), (False, False)]
    options = [(False, True)]
    for align_corners, half_pixel_centers in options:
      for in_shape, out_shape in shape_pairs:
        yield in_shape, out_shape, align_corners, half_pixel_centers

  def _getJacobians(self,
                    in_shape,
                    out_shape,
                    align_corners=False,
                    half_pixel_centers=False,
                    dtype=np.float32,
                    use_gpu=True,
                    force_gpu=True):
    if not test.is_gpu_available():
      force_gpu = False
    with self.cached_session(use_gpu=use_gpu, force_gpu=force_gpu) as sess:
      # Input values should not influence gradients
      x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(dtype)
      input_tensor = constant_op.constant(x, shape=in_shape)
      s1 = nn_ops.softmax(input_tensor)
      resized_tensor = image_ops.resize_bilinear(
          s1,
          out_shape[1:3],
          align_corners=align_corners,
          half_pixel_centers=half_pixel_centers)
      s2 = nn_ops.softmax(resized_tensor)
      # compute_gradient will use a random tensor as the init value
      return gradient_checker.compute_gradient(input_tensor, in_shape,
                                               s2, out_shape)

  @parameterized.parameters({
      'batch_size': 1,
      'channel_count': 1
  }, {
      'batch_size': 4,
      'channel_count': 3
  }, {
      'batch_size': 3,
      'channel_count': 2
  })
  @test_util.run_deprecated_v1
  def testGradients(self, batch_size, channel_count):
    smaller_shape = [batch_size, 2, 3, channel_count]
    larger_shape = [batch_size, 5, 6, channel_count]
    # smaller_shape = [batch_size, 2, 2, channel_count]
    # larger_shape = [batch_size, 5, 5, channel_count]
    for in_shape, out_shape, align_corners, half_pixel_centers in \
        self._itGen(smaller_shape, larger_shape):
      jacob_a, jacob_n = self._getJacobians(in_shape, out_shape, align_corners,
                                            half_pixel_centers)
      threshold = 1e-4
      self.assertAllClose(jacob_a, jacob_n, threshold, threshold)

if __name__ == "__main__":
  googletest.main()
