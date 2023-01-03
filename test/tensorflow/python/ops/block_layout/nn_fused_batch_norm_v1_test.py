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
"""Tests for fused_batch_norm_v1 related functionality in tensorflow.ops.nn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import gen_nn_ops

def fused_batch_norm_v1(
    x,
    scale,
    offset,  # pylint: disable=invalid-name
    mean=None,
    variance=None,
    epsilon=0.001,
    data_format="NHWC",
    is_training=True,
    name=None,
    exponential_avg_factor=1.0):

  if (not is_training or exponential_avg_factor != 1.0) and (
      (mean is None) or (variance is None)):
    raise ValueError("Both `mean` and `variance` must be a 1D tensor when "
                     "`is_training` is False or `exponential_avg_factor` != "
                     f"1.0. Received: `mean` {mean!r} and `variance` "
                     f"{variance!r}")
  x = ops.convert_to_tensor(x, name="input")
  scale = ops.convert_to_tensor(scale, name="scale")
  offset = ops.convert_to_tensor(offset, name="offset")
  if mean is None:
    mean = constant_op.constant([])
  if variance is None:
    variance = constant_op.constant([])

  # Set a minimum epsilon to 1.001e-5, which is a requirement by CUDNN to
  # prevent exception (see cudnn.h).
  min_epsilon = 1.001e-5
  epsilon = epsilon if epsilon > min_epsilon else min_epsilon

  y, running_mean, running_var, _, _ = gen_nn_ops._fused_batch_norm(
      x,
      scale,
      offset,
      mean,
      variance,
      epsilon=epsilon,
      exponential_avg_factor=exponential_avg_factor,
      data_format=data_format,
      is_training=is_training,
      name=name)
  return y, running_mean, running_var

class BatchNormalizationV1Test(test.TestCase):

  def _test_grad_grad(self,
                      x_shape,
                      x_dtype,
                      scale_shape,
                      scale_dtype,
                      use_gpu=True,
                      exponential_avg_factor=1.0,
                      data_format='NHWC',
                      is_training=True,
                      err_tolerance=1e-3):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    grad_y_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.cached_session(use_gpu=use_gpu) as sess:
      x = constant_op.constant(x_val, name='x')
      grad_y = constant_op.constant(grad_y_val, name='grad_y')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      if is_training and exponential_avg_factor == 1.0:
        pop_mean = None
        pop_var = None
      else:
        pop_mean = np.random.random_sample(scale_shape).astype(scale_dtype)
        pop_var = np.random.random_sample(scale_shape).astype(scale_dtype)

      y, _, _ = fused_batch_norm_v1(
          x,
          scale,
          offset,
          mean=pop_mean,
          variance=pop_var,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format,
          is_training=is_training)
      y_id = array_ops.identity(y)
      grad_x, grad_scale, grad_offset = array_ops.identity_n(gradients_impl.gradients(
          y_id, [x, scale, offset], grad_y))

      if is_training:
        epsilon = y.op.get_attr('epsilon')
        data_format = y.op.get_attr('data_format')
        grad_vals = self.evaluate([grad_x, grad_scale, grad_offset])
        grad_internal = nn_grad._BatchNormGrad(grad_y, x, scale, pop_mean,
                                               pop_var, epsilon, data_format)
        grad_internal_vals = self.evaluate(list(grad_internal))
        for grad_val, grad_internal_val in zip(grad_vals, grad_internal_vals):
          self.assertAllClose(grad_val, grad_internal_val, atol=err_tolerance)

      if x_dtype != np.float16:
        err_grad_grad_y_1 = gradient_checker.compute_gradient_error(
            grad_y, x_shape, grad_x, x_shape)
        err_grad_grad_y_2 = gradient_checker.compute_gradient_error(
            grad_y, x_shape, grad_scale, scale_shape)
        err_grad_grad_y_3 = gradient_checker.compute_gradient_error(
            grad_y, x_shape, grad_offset, scale_shape)
        # In freeze mode, grad_x is not a function of x.
        if is_training:
          err_grad_x_1 = gradient_checker.compute_gradient_error(
              x, x_shape, grad_x, x_shape)
        err_grad_x_2 = gradient_checker.compute_gradient_error(
            x, x_shape, grad_scale, scale_shape)

        err_grad_scale = gradient_checker.compute_gradient_error(
            scale, scale_shape, grad_x, x_shape)
      else:
        x32 = constant_op.constant(x_val, dtype=dtypes.float32, name='x32')
        grad_y32 = constant_op.constant(
            grad_y_val, dtype=dtypes.float32, name='grad_y32')
        y32, _, _ = array_ops.identity_n(fused_batch_norm_v1(
            x32,
            scale,
            offset,
            mean=pop_mean,
            variance=pop_var,
            exponential_avg_factor=exponential_avg_factor,
            data_format=data_format,
            is_training=is_training))
        grad_x32, grad_scale32, grad_offset32 = gradients_impl.gradients(
            y32, [x32, scale, offset], grad_y32)
        err_grad_grad_y_1 = self._compute_gradient_error_float16(
            grad_y, grad_y32, x_shape, grad_x, grad_x32, x_shape)
        err_grad_grad_y_2 = self._compute_gradient_error_float16(
            grad_y, grad_y32, x_shape, grad_scale, grad_scale32, scale_shape)
        err_grad_grad_y_3 = self._compute_gradient_error_float16(
            grad_y, grad_y32, x_shape, grad_offset, grad_offset32, scale_shape)
        # In freeze mode, grad_x is not a function of x.
        if is_training:
          err_grad_x_1 = self._compute_gradient_error_float16(
              x, x32, x_shape, grad_x, grad_x32, x_shape)
        err_grad_x_2 = self._compute_gradient_error_float16(
            x, x32, x_shape, grad_scale, grad_scale32, scale_shape)

        err_grad_scale = self._compute_gradient_error_float16(
            scale, scale, scale_shape, grad_x, grad_x32, x_shape)

    self.assertLess(err_grad_grad_y_1, err_tolerance)
    self.assertLess(err_grad_grad_y_2, err_tolerance)
    self.assertLess(err_grad_grad_y_3, err_tolerance)
    if is_training:
      self.assertLess(err_grad_x_1, err_tolerance)
    self.assertLess(err_grad_x_2, err_tolerance)
    self.assertLess(err_grad_scale, err_tolerance)

  def _testBatchNormGradGrad(self, config):
    shape = config['shape']
    err_tolerance = config['err_tolerance']
    dtype = config['dtype']
    rank = len(shape)
    if rank == 4:
      data_format_nhwc, features_nhwc = 'NHWC', shape[3]
      data_format_nchw, features_nchw = 'NCHW', shape[1]
    else:
      data_format_nhwc, features_nhwc = 'NDHWC', shape[4]
      data_format_nchw, features_nchw = 'NCDHW', shape[1]
    for is_training in [True,]:
      if test.is_gpu_available(cuda_only=True):
        self._test_grad_grad(
            shape,
            dtype, [features_nhwc],
            np.float32,
            use_gpu=True,
            data_format=data_format_nhwc,
            is_training=is_training,
            err_tolerance=err_tolerance)
        self._test_grad_grad(
            shape,
            dtype, [features_nchw],
            np.float32,
            use_gpu=True,
            data_format=data_format_nchw,
            is_training=is_training,
            err_tolerance=err_tolerance)
      self._test_grad_grad(
          shape,
          dtype, [features_nhwc],
          np.float32,
          use_gpu=False,
          data_format=data_format_nhwc,
          is_training=is_training,
          err_tolerance=err_tolerance)
      self._test_grad_grad(
          shape,
          dtype, [features_nchw],
          np.float32,
          use_gpu=False,
          data_format=data_format_nchw,
          is_training=is_training,
          err_tolerance=err_tolerance)

  @test_util.run_deprecated_v1
  def testBatchNormGradGradConfig1(self):
    config = {
        'shape': [2, 3, 4, 5],
        'err_tolerance': 1e-2,
        'dtype': np.float32,
    }
    self._testBatchNormGradGrad(config)

if __name__ == '__main__':
  test.main()
