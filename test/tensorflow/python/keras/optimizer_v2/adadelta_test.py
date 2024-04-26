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
"""Tests for Adadelta Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_USE_LEGACY_KERAS']='1'

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tf_keras.src.optimizers.legacy import adadelta
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables

class AdadeltaOptimizerTest(test.TestCase):

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    num_updates = 4  # number of ADADELTA steps to perform
    for dtype in [dtypes.half, dtypes.float32]:
      for grad in [0.2, 0.1, 0.01]:
        for lr in [1.0, 0.5, 0.1]:
          var0_init = [1.0, 2.0]
          var1_init = [3.0, 4.0]
          if use_resource:
            var0 = resource_variable_ops.ResourceVariable(
                var0_init, dtype=dtype)
            var1 = resource_variable_ops.ResourceVariable(
                var1_init, dtype=dtype)
          else:
            var0 = variables.Variable(var0_init, dtype=dtype)
            var1 = variables.Variable(var1_init, dtype=dtype)

          grads = constant_op.constant([grad, grad], dtype=dtype)

          accum = 0.0
          accum_update = 0.0
          from tensorflow.python.training import adadelta
          # ADADELTA gradient optimizer
          rho = 0.95
          epsilon = 1e-8
          if use_callable_params:
            adadelta_opt = adadelta.AdadeltaOptimizer(
                learning_rate=lambda: lr,  # pylint: disable=cell-var-from-loop
                rho=lambda: rho,  # pylint: disable=cell-var-from-loop
                epsilon=epsilon)  # pylint: disable=cell-var-from-loop
          else:
            adadelta_opt = adadelta.AdadeltaOptimizer(
                learning_rate=lr, rho=rho, epsilon=epsilon)
          if not context.executing_eagerly():
            adadelta_update = adadelta_opt.apply_gradients(
                zip([grads, grads], [var0, var1]))
            self.evaluate(variables.global_variables_initializer())

            # TODO(lxuechen): This is hard to test in eager mode,
            # since the optimizer is not fully initialized until the first
            # call to `apply_gradients`
            opt_vars = adadelta_opt.variables()
            self.assertStartsWith(opt_vars[0].name, var0._shared_name)
            self.assertStartsWith(opt_vars[1].name, var0._shared_name)
            self.assertStartsWith(opt_vars[2].name, var1._shared_name)
            self.assertStartsWith(opt_vars[3].name, var1._shared_name)
            self.assertEqual(4, len(opt_vars))
            # Assign slots
            slot = [None] * 2
            slot_update = [None] * 2
            self.assertEqual(["accum", "accum_update"],
                             adadelta_opt.get_slot_names())
            slot[0] = adadelta_opt.get_slot(var0, "accum")
            self.assertEqual(slot[0].get_shape(), var0.get_shape())
            self.assertFalse(slot[0] in variables.trainable_variables())

            slot_update[0] = adadelta_opt.get_slot(var0, "accum_update")
            self.assertEqual(slot_update[0].get_shape(), var0.get_shape())
            self.assertFalse(slot_update[0] in variables.trainable_variables())

            slot[1] = adadelta_opt.get_slot(var1, "accum")
            self.assertEqual(slot[1].get_shape(), var1.get_shape())
            self.assertFalse(slot[1] in variables.trainable_variables())

            slot_update[1] = adadelta_opt.get_slot(var1, "accum_update")
            self.assertEqual(slot_update[1].get_shape(), var1.get_shape())
            self.assertFalse(slot_update[1] in variables.trainable_variables())

          # Fetch params to validate initial values
          self.assertAllClose(var0_init, self.evaluate(var0))
          self.assertAllClose(var1_init, self.evaluate(var1))

          update = [None] * num_updates
          tot_update = 0
          for step in range(num_updates):
            # Run adadelta update for comparison
            if not context.executing_eagerly():
              self.evaluate(adadelta_update)
            else:
              adadelta_opt.apply_gradients(zip([grads, grads], [var0, var1]))

            # Perform initial update without previous accum values
            accum = accum * rho + (grad**2) * (1 - rho)
            update[step] = (
                np.sqrt(accum_update + epsilon) *
                (1. / np.sqrt(accum + epsilon)) * grad)
            accum_update = (
                accum_update * rho + (update[step]**2) * (1.0 - rho))
            tot_update += update[step] * lr

            if not context.executing_eagerly():
              # Check that the accumulators have been updated
              # TODO(lxuechen): This is hard to test in eager mode
              for slot_idx in range(2):
                self.assertAllCloseAccordingToType(
                    np.array([accum, accum], dtype=dtype.as_numpy_dtype()),
                    self.evaluate(slot[slot_idx]),
                    rtol=1e-5)

                self.assertAllCloseAccordingToType(
                    np.array(
                        [accum_update, accum_update],
                        dtype=dtype.as_numpy_dtype()),
                    self.evaluate(slot_update[slot_idx]),
                    rtol=1e-5)

              # Check that the parameters have been updated
              self.assertAllCloseAccordingToType(
                  np.array(
                      [var0_init[0] - tot_update, var0_init[1] - tot_update],
                      dtype=dtype.as_numpy_dtype()),
                  self.evaluate(var0),
                  rtol=1e-5)

              self.assertAllCloseAccordingToType(
                  np.array(
                      [var1_init[0] - tot_update, var1_init[1] - tot_update],
                      dtype=dtype.as_numpy_dtype()),
                  self.evaluate(var1),
                  rtol=1e-5)

  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes
  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testBasicCallableParams(self):
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  def doTestAdadelta(self, dtype):
    SHAPE1 = [8192, 8192]
    SHAPE2 = [8192, 8192]

    np.random.seed(1)
    input_1 = np.reshape(np.random.normal(size=np.prod(SHAPE1)), newshape=SHAPE1)
    input_2 = np.reshape(np.random.normal(size=np.prod(SHAPE2)), newshape=SHAPE2)
    var0 = resource_variable_ops.ResourceVariable(input_1, dtype=dtype)
    x = constant_op.constant(input_2, dtype=dtype)
    pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
    loss = pred * pred
    from tensorflow.python.training import adadelta
    sgd_op = adadelta.AdadeltaOptimizer(1.0, 1.0, 1.0).minimize(loss)

    self.evaluate(variables.global_variables_initializer())

    # Fetch params to validate initial values
    self.assertAllCloseAccordingToType(input_1, self.evaluate(var0))

    # Run 1 step of sgd
    sgd_op.run()
    return var0

  @test_util.run_deprecated_v1
  def testMinimizeSparseResourceVariable(self):
    for dtype in [dtypes.float32]:
        
        with self.session(use_gpu=False):
            res = self.doTestAdadelta(dtype)
            ans_cpu = self.evaluate(res)

        with self.session(use_gpu=True):
            res = self.doTestAdadelta(dtype)
            y_gpu = self.evaluate(res)

        # Validate updated params
        self.assertAllClose(tf.cast(ans_cpu, dtype), y_gpu, rtol=1e-2, atol=1e-2)

  def testConstructAdadeltaWithLR(self):
    opt = adadelta.Adadelta(lr=1.0, rho=0.9, epsilon=1.)
    opt_2 = adadelta.Adadelta(learning_rate=0.1, rho=0.9, epsilon=1., lr=1.0)
    opt_3 = adadelta.Adadelta(learning_rate=0.1, rho=0.9, epsilon=1.)
    self.assertIsInstance(opt.lr, variables.Variable)
    self.assertIsInstance(opt_2.lr, variables.Variable)
    self.assertIsInstance(opt_3.lr, variables.Variable)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(opt.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_2.lr), (1.0))
    self.assertAllClose(self.evaluate(opt_3.lr), (0.1))

  def testConstructAdadeltaWithEpsilonValues(self):
    opt = adadelta.Adadelta(epsilon=None)
    self.assertEqual(opt.epsilon, 1e-7)

    opt = adadelta.Adadelta(epsilon=1e-8)
    self.assertEqual(opt.epsilon, 1e-8)


if __name__ == "__main__":
  test.main()
