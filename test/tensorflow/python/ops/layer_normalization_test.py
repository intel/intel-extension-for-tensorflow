# Copyright (c) 2022 Intel Corporation
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for normalization layers."""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import keras_testing_utils as testing_utils
# from keras import testing_utils
from intel_extension_for_tensorflow.python.test_func import keras_parameterized

import intel_extension_for_tensorflow.python.ops.layer_norm as layer_normalization
try:
  from tf_keras.src.layers.normalization import layer_normalization as tf_layer_normalization
except ImportError:
  from tf_keras.layers.normalization import layer_normalization as tf_layer_normalization

import numpy as np

import tf_keras as keras
from tensorflow.python.keras import combinations

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.training import gradient_descent

def _run_layernorm_correctness_test(layer, dtype='float32'):
  model = keras.models.Sequential()
  model.add(keras.layers.Lambda(lambda x: math_ops.cast(x, dtype='float16')))
  norm = layer(input_shape=(2, 2, 2), dtype=dtype)
  model.add(norm)
  model.compile(
      loss='mse',
      optimizer=gradient_descent.GradientDescentOptimizer(0.01),
      run_eagerly=testing_utils.should_run_eagerly())

  # centered on 5.0, variance 10.0
  x = (np.random.normal(loc=5.0, scale=10.0, size=(1000, 2, 2, 2))
       .astype(dtype))
  model.fit(x, x, epochs=4, verbose=0)
  out = model.predict(x)
  out -= keras.backend.eval(norm.beta)
  out /= keras.backend.eval(norm.gamma)

  np.testing.assert_allclose(out.mean(), 0.0, atol=1e-1)
  np.testing.assert_allclose(out.std(), 1.0, atol=1e-1)


class LayerNormalizationTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_basic_layernorm(self):
    testing_utils.layer_test(
        layer_normalization.LayerNormalization,
        kwargs={
            'gamma_regularizer': keras.regularizers.l2(0.01),
            'beta_regularizer': keras.regularizers.l2(0.01)
        },
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        layer_normalization.LayerNormalization,
        kwargs={
            'gamma_initializer': 'ones',
            'beta_initializer': 'ones',
        },
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        layer_normalization.LayerNormalization,
        kwargs={'scale': False,
                'center': False},
        input_shape=(3, 3))
    testing_utils.layer_test(
        layer_normalization.LayerNormalization,
        kwargs={'axis': (-3, -2, -1)},
        input_shape=(2, 8, 8, 3))

  @keras_parameterized.run_all_keras_modes
  def test_non_fused_layernorm(self):
    testing_utils.layer_test(
        layer_normalization.LayerNormalization,
        kwargs={'axis': -2},
        input_shape=(3, 4, 2))
    testing_utils.layer_test(
        layer_normalization.LayerNormalization,
        kwargs={'axis': (-3, -2)},
        input_shape=(2, 8, 8, 3))
    testing_utils.layer_test(
        layer_normalization.LayerNormalization,
        kwargs={'axis': (-3, -1)},
        input_shape=(2, 8, 8, 3))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_layernorm_weights(self):
    layer = layer_normalization.LayerNormalization(scale=False, center=False)
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.weights), 0)

    layer = layer_normalization.LayerNormalization()
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.trainable_weights), 2)
    self.assertEqual(len(layer.weights), 2)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_layernorm_regularization(self):
    layer = layer_normalization.LayerNormalization(
        gamma_regularizer='l1', beta_regularizer='l1')
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.losses), 2)
    max_norm = keras.constraints.max_norm
    layer = layer_normalization.LayerNormalization(
        gamma_constraint=max_norm, beta_constraint=max_norm)
    layer.build((None, 3, 4))
    self.assertEqual(layer.gamma.constraint, max_norm)
    self.assertEqual(layer.beta.constraint, max_norm)

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_convnet_channel_last(self):
    model = keras.models.Sequential()
    norm = layer_normalization.LayerNormalization(input_shape=(4, 4, 3))
    model.add(norm)
    model.compile(
        loss='mse',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01),
        run_eagerly=testing_utils.should_run_eagerly())

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 4, 4, 3))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm.beta), (1, 1, 1, 3))
    out /= np.reshape(keras.backend.eval(norm.gamma), (1, 1, 1, 3))

    np.testing.assert_allclose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
    np.testing.assert_allclose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_correctness(self):
    _run_layernorm_correctness_test(
        layer_normalization.LayerNormalization, dtype='float32')

#  Not support fp16 for training.
#  @keras_parameterized.run_all_keras_modes
#  def test_layernorm_mixed_precision(self):
#    _run_layernorm_correctness_test(
#        layer_normalization.LayerNormalization, dtype='float16')

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testIncorrectAxisType(self):
    with self.assertRaisesRegex(TypeError,
                                r'Expected an int or a list/tuple of ints'):
      _ = layer_normalization.LayerNormalization(axis={'axis': -1})

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInvalidAxis(self):
    with self.assertRaisesRegex(ValueError, r'Invalid axis: 3'):
      layer_norm = layer_normalization.LayerNormalization(axis=3)
      layer_norm.build(input_shape=(2, 2, 2))

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testDuplicateAxis(self):
    with self.assertRaisesRegex(ValueError, r'Duplicate axis:'):
      layer_norm = layer_normalization.LayerNormalization(axis=[-1, -1])
      layer_norm.build(input_shape=(2, 2, 2))

class LayerNormalizationNumericsTest(keras_parameterized.TestCase):
  """Tests LayerNormalization has correct and numerically stable outputs."""

  def _expected_layer_norm(self, x, beta, gamma, batch_input_shape, axis,
                           epsilon):
    """Returns the layer norm, which is computed using NumPy."""
    broadcast_shape = [batch_input_shape[i] if i in axis else 1
                       for i in range(len(batch_input_shape))]
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    expected = (x - mean) / np.sqrt(var + epsilon)
    expected *= np.reshape(gamma, broadcast_shape)
    expected += np.reshape(beta, broadcast_shape)
    return expected

  def _test_forward_pass(self, batch_input_shape, axis, fp32_tol=1e-4,
                         fp16_tol=1e-2):
    """Tests the forward pass of layer layer_normalization.

    Args:
      batch_input_shape: The input shape that will be used to test, including
        the batch dimension.
      axis: A list of axises to normalize. Will be passed to the `axis` argument
        of Layerlayer_normalization.
      fp32_tol: The relative and absolute tolerance for float32.
      fp16_tol: The relative and absolute tolerance for float16.
    """
    param_shape = [batch_input_shape[i] for i in axis]
    param_elems = 1
    for dim in param_shape:
      param_elems *= dim
    beta = np.arange(param_elems, dtype='float32').reshape(param_shape)
    gamma = np.arange(1, param_elems + 1, dtype='float32').reshape(param_shape)
    x = np.random.normal(size=batch_input_shape)

    for epsilon in 1e-12, 1e-3:
      expected = self._expected_layer_norm(x, beta, gamma, batch_input_shape,
                                           axis, epsilon)
      for dtype in 'float32', 'float16':
        '''wenjie: LayerNorm op does not register float16 on CPU
           test case for float16 needs to be enabled when registration is done
        '''
        if not test.is_gpu_available() and dtype == 'float16':
          continue
        norm = layer_normalization.LayerNormalization(
            axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
            epsilon=epsilon, beta_initializer=keras.initializers.constant(beta),
            gamma_initializer=keras.initializers.constant(gamma))
        y = norm(keras.backend.cast(x, dtype))
        actual = keras.backend.eval(y)

        if dtype == 'float32':
          tol = fp32_tol
        else:
          assert dtype == 'float16'
          tol = fp16_tol

        # We use absolute tolerances in addition to relative tolerances, because
        # some of the values are very close to zero.
        self.assertAllClose(expected, actual, rtol=tol, atol=tol)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_forward(self):
    # For numeric stability, we ensure the axis's dimension(s) have at least 4
    # elements.
    self._test_forward_pass((4, 3), (0,))
    self._test_forward_pass((3, 4), (1,))
    self._test_forward_pass((4, 3, 2), (0,))
    self._test_forward_pass((2, 4, 2), (1,))
    self._test_forward_pass((4, 5, 6), (2,))
    self._test_forward_pass((2, 3, 2), (0, 2))
    self._test_forward_pass((2, 2, 2, 2), (1, 3))
    self._test_forward_pass((2, 2, 2, 2), (2, 3))
    self._test_forward_pass((2, 3, 4, 5), (3,))

  def _test_backward_pass(self, batch_input_shape, axis, fp32_tol=1e-3):
    """Tests the backwards pass of layer normalization.

    Args:
      batch_input_shape: The input shape that will be used to test, including
        the batch dimension.
      axis: A list of axises to normalize. Will be passed to the `axis` argument
        of LayerNormalization.
      fp32_tol: The relative and absolute tolerance for float32.
    """
    param_shape = [batch_input_shape[i] for i in axis]
    param_elems = 1
    for dim in param_shape:
      param_elems *= dim
    beta = np.arange(param_elems, dtype='float32').reshape(param_shape)
    gamma = np.arange(1, param_elems + 1, dtype='float32').reshape(param_shape)
    x = np.random.normal(size=batch_input_shape)
    itex_training_fp16 = False

    for epsilon in 1e-12, 1e-3:
      # Now only support float32 training for onednn layernorm primitive.
      dtype = 'float32'

      norm = layer_normalization.LayerNormalization(
          axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
          epsilon=epsilon, beta_initializer=keras.initializers.constant(beta),
          gamma_initializer=keras.initializers.constant(gamma))
      norm.build(x.shape)

      tf_norm = tf_layer_normalization.LayerNormalization(
          axis=axis, dtype=dtype, batch_input_shape=batch_input_shape,
          epsilon=epsilon, beta_initializer=keras.initializers.constant(beta),
          gamma_initializer=keras.initializers.constant(gamma))
      tf_norm.build(x.shape)

      # pylint: disable=cell-var-from-loop
      def forward_fn(x, beta, gamma):
        # We must monkey-patch the attributes of `norm` with the function
        # arguments, so that the gradient checker will properly compute their
        # gradients. The gradient checker computes gradients with respect to
        # the input arguments of `f`.
        with test.mock.patch.object(norm, 'beta', beta):
          with test.mock.patch.object(norm, 'gamma', gamma):
            return norm(x)
      # pylint: enable=cell-var-from-loop
      results = gradient_checker_v2.compute_gradient(
          forward_fn, [keras.backend.cast(x, dtype), norm.beta, norm.gamma])
      ([x_grad_t, beta_grad_t, gamma_grad_t],
       [x_grad_n, beta_grad_n, gamma_grad_n]) = results

      def tf_forward_fn(x, beta, gamma):
        with test.mock.patch.object(tf_norm, 'beta', beta):
          with test.mock.patch.object(tf_norm, 'gamma', gamma):
            return tf_norm(x)
      # pylint: enable=cell-var-from-loop
      tf_results = gradient_checker_v2.compute_gradient(tf_forward_fn,
          [keras.backend.cast(x, dtype), tf_norm.beta, tf_norm.gamma])
      ([tf_x_grad_t, tf_beta_grad_t, tf_gamma_grad_t],
       [tf_x_grad_n, tf_beta_grad_n, tf_gamma_grad_n]) = tf_results

      tol = fp32_tol

      self.assertAllClose(x_grad_n, tf_x_grad_n, rtol=tol, atol=tol)
      self.assertAllClose(beta_grad_n, tf_beta_grad_n, rtol=tol, atol=tol)
      self.assertAllClose(gamma_grad_n, tf_gamma_grad_n, rtol=tol, atol=tol)

  # The gradient_checker_v2 does not work properly with LayerNorm in graph mode.
  @testing_utils.run_v2_only
  def test_backward(self):
    # For numeric stability, we ensure the axis's dimension(s) have at least 4
    # elements.
    self._test_backward_pass((4, 3), (0,))
    self._test_backward_pass((3, 4), (1,))
    self._test_backward_pass((2, 4, 2), (1,))
    self._test_backward_pass((2, 3, 4), (2,))
    self._test_backward_pass((2, 3, 2), (0, 2))
    self._test_backward_pass((2, 2, 2, 2), (1, 3))
    self._test_backward_pass((2, 2, 2, 2), (2, 3))
    self._test_backward_pass((1, 2, 3, 4), (3,))

  # _MklLayerNorm defines scale/shift/mean/variance datatype as bf16 when input is
  # is bf16, while OneDnn only supports those tensors with fp32 datatype. Here we
  # have a test here to ensure there is not accuracy issue.
  @test_util.run_deprecated_v1
  def test_bf16mkllayernorm(self):
    with ops.name_scope("test"):
      x_shape = [16, 16, 16]
      scale_shape = [16]

      x_np = np.random.normal(size=x_shape)
      scale_np = np.random.normal(size=scale_shape)
      shift_np = np.random.normal(size=scale_shape)

      x_bf16 =  constant_op.constant(x_np, dtype=dtypes.bfloat16)
      scale_bf16 = constant_op.constant(scale_np, dtype=dtypes.bfloat16)
      shift_bf16 = constant_op.constant(shift_np, dtype=dtypes.bfloat16)

      ln_bf16 = load_ops_library._MklLayerNorm(x=x_bf16, scale=scale_bf16, offset=shift_bf16)
      ln_bf16 = math_ops.cast(ln_bf16, dtypes.float32)
      ln_bf16 = array_ops.identity(ln_bf16)

      ln_bf16_res = self.evaluate(ln_bf16)

      x_fp32 = constant_op.constant(x_np, dtype=dtypes.float32)
      scale_fp32 = constant_op.constant(scale_np, dtype=dtypes.float32)
      shift_fp32 = constant_op.constant(shift_np, dtype=dtypes.float32)

      ln_fp32 = load_ops_library._MklLayerNorm(x=x_fp32, scale=scale_fp32, offset=shift_fp32)
      ln_fp32 = array_ops.identity(ln_fp32)

      ln_fp32_res = self.evaluate(ln_fp32)

      self.assertAllClose(ln_bf16_res, ln_fp32_res, rtol=1e-2, atol=1e-2)




if __name__ == '__main__':
  test.main()
