

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stateful_random_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
import tensorflow as tf
from tensorflow.python.ops import stateful_random_ops as random
import re

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import values as dist_values
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.kernel_tests.random import util as random_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import stateful_random_ops as random
from tensorflow.python.ops import variables
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test

g_seeded = None
g_unseeded = None


GPU_FLOATS = [dtypes.float16, dtypes.float32, dtypes.bfloat16]
CPU_FLOATS = GPU_FLOATS + [dtypes.float64]
FLOATS = GPU_FLOATS
INTS = [dtypes.int32, dtypes.int64]

RNG_ALG_PHILOX = 1
RNG_ALG_THREEFRY = 2
PHILOX_STATE_SIZE = 3
THREEFRY_STATE_SIZE = 2

def _get_state_size(alg):
  if alg == RNG_ALG_PHILOX:
    return PHILOX_STATE_SIZE
  elif alg == RNG_ALG_THREEFRY:
    return THREEFRY_STATE_SIZE
  else:
    raise ValueError("Unsupported algorithm id: %s" % alg)

def _key_to_state(alg, key):
    # Padding with zeros on the left. The zeros will be the counter.
    return [0] * (_get_state_size(alg) - 1) + [key]


class StatefulRandomOpsTest(test.TestCase, parameterized.TestCase):

  def split_helper(self, gen, count):
      keys_tmp = gen._make_int64_keys(shape=[count])
      alg = RNG_ALG_PHILOX
      gens = [random.Generator(state=_key_to_state(alg, key), alg=alg)
              for key in self.evaluate(keys_tmp)]  
      return gens

  def assertAllDifferent(self, tensors):
    tensors = [array_ops.reshape(t, shape=[-1]) for t in tensors]
    ls = self.evaluate(array_ops.concat(tensors, axis=0)).tolist()
    self.assertAllEqual(len(ls), len(set(ls)))

  def testBatchSeeds(self):
    """Test for batch seeds.
    """
    shape = [2, 3]
    count = 6
    with self.cached_session(use_gpu=True) as sess:
      gen = random.Generator.from_seed(1234)
      sess.run(gen._state_var.initializer)
      keys1 = gen._make_int64_keys(shape=shape)
      keys2 = gen._make_int64_keys(shape=shape)
      self.assertAllDifferent([keys1, keys2])
      seeds1 = gen.make_seeds(count=count)
      seeds2 = gen.make_seeds(count=count)
      self.assertAllDifferent([seeds1[0, :], seeds2[0, :]])

      gens = self.split_helper(gen, count=count)
      
      sess.run([g._state_var.initializer for g in gens])
      self.assertAllEqual(count, len(gens))
      randoms = [g.uniform_full_int(shape=shape, dtype=dtypes.int32)
                for g in gens]
      self.assertAllDifferent(randoms)
      # Tests graph mode.
      @def_function.function
      def f():
        return gen.make_seeds(count=count)
      for _ in range(3):
        f()

  def assertRegex(self, pattern, text):
    self.assertTrue(
        re.search(pattern, text),
        "Can't find pattern '%s' in text '%s'" % (pattern, text))

  @test_util.run_v1_only(
      ("This test is specifically for checking TF1 compatibility. "
       "It cannot run under TF2."))
  def testTF1(self):
    seed = 1234
    shape = [2, 3]
    expected_normal1 = constant_op.constant(
        [[0.9356609, 1.0854305, -0.93788373],
         [-0.50615472, 1.31697023, 0.71375787]], dtype=dtypes.float32)
    expected_normal2 = constant_op.constant(
        [[-0.3964749, 0.8369565, -0.30946946],
         [1.1206646, 1.00852597, -0.10185789]], dtype=dtypes.float32)
    with self.cached_session(use_gpu=True) as sess:
      gen1 = random.Generator.from_seed(seed)
      gen2 = random.Generator.from_non_deterministic_state()
      sess.run((gen1._state_var.initializer, gen2._state_var.initializer))
      r1 = gen1.normal(shape, dtype=dtypes.float32)
      r2 = gen2.normal(shape, dtype=dtypes.float32)
      def f():
        return sess.run((r1, r2))
      def check_results(expected_normal, v1, v2):
        self.assertAllClose(expected_normal, v1, rtol=1e-5, atol=1e-5)
        self.assertAllEqual(shape, v2.shape)
      check_results(expected_normal1, *f())
      check_results(expected_normal2, *f())

  @parameterized.parameters(INTS + [dtypes.uint32, dtypes.uint64])
  @test_util.run_cuda_only
  def testGPUEqualsCPU(self, dtype):
    """Tests that GPU and CPU generate the same integer outputs."""
    seed = 1234
    shape = [315, 49]
    with ops.device("/device:CPU:0"):
      rng1 = random.Generator.from_seed(seed)
      cpu = rng1.uniform_full_int(
          shape=shape, dtype=dtype)
    with ops.device(test_util.gpu_device_name()):
      rng2 = random.Generator.from_seed(seed)
      gpu = rng2.uniform_full_int(
          shape=shape, dtype=dtype)
    with self.cached_session(use_gpu=True) as sess:
        sess.run((rng1._state_var.initializer, rng2._state_var.initializer))
        self.assertAllEqual(self.evaluate(cpu), self.evaluate(gpu))

  @parameterized.parameters(FLOATS + INTS)
  def testUniformIsInRange(self, dtype):
    minval = 2
    maxval = 33
    size = 1000
    with self.cached_session(use_gpu=True) as sess:
        gen = random.Generator.from_seed(1234)
        x = gen.uniform(
            shape=[size], dtype=dtype, minval=minval, maxval=maxval)
        sess.run(gen._state_var.initializer)
        x = self.evaluate(x)
        self.assertTrue(np.all(x >= minval))
        self.assertTrue(np.all(x < maxval))

  @parameterized.parameters(FLOATS)
  def testNormalIsFinite(self, dtype):
    with self.cached_session(use_gpu=True) as sess:
        gen = random.Generator.from_seed(1234)
        x = gen.normal(shape=[10000], dtype=dtype)
        sess.run(gen._state_var.initializer)
        x = self.evaluate(x)
        self.assertTrue(np.all(np.isfinite(x)))

  @parameterized.parameters(FLOATS + INTS)
  def testDistributionOfUniform(self, dtype):
    """Use Pearson's Chi-squared test to test for uniformity."""
    n = 1000
    seed = 12
    with self.cached_session(use_gpu=True) as sess:
        gen = random.Generator.from_seed(seed)
        maxval = 1
        if dtype.is_integer:
            maxval = 100
        x = gen.uniform(shape=[n], maxval=maxval, dtype=dtype)
        sess.run(gen._state_var.initializer)
        x = self.evaluate(x).astype(float)        
    if maxval > 1:
      # Normalize y to range [0, 1).
      x = x.astype(float) / maxval
    # Tests that the values are distributed amongst 10 bins with equal
    # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
    # p=0.05. This test is probabilistic and would be flaky if the random
    # seed were not fixed.
    val = random_test_util.chi_squared(x, 10)
    self.assertLess(val, 16.92)

  @parameterized.parameters(FLOATS)
  def testDistributionOfNormal(self, dtype):
    """Use Anderson-Darling test to test distribution appears normal."""
    n = 1000
    with self.cached_session(use_gpu=True) as sess:
        gen = random.Generator.from_seed(1234)
        x = gen.normal(shape=[n], dtype=dtype)
        sess.run(gen._state_var.initializer)
        x = self.evaluate(x)        
    # The constant 2.492 is the 5% critical value for the Anderson-Darling
    # test where the mean and variance are known. This test is probabilistic
    # so to avoid flakiness the seed is fixed.
    self.assertLess(
        random_test_util.anderson_darling(x.astype(float)), 2.492)

  @test_util.run_v2_only
  def testRngSkip(self):
      '''
      This test verifies the correct variation of random state when skip.
      '''
      key = 1234
      counter = 5678
      delta = 432
      state = math_ops.cast([counter, 0, key], 'int64')
      state = variables.Variable(state, dtype="int64", trainable=False)
      resource_variable_ops.variable_accessed(state)
      gen_stateful_random_ops.rng_skip(
              state.handle,
              algorithm=math_ops.cast(tf.random.Algorithm.PHILOX.value, dtypes.int64),
              delta=math_ops.cast(delta, dtypes.int64))
      new_counter = state[0]
      self.assertAllEqual(counter + delta * 256, new_counter)

  @parameterized.parameters([dtypes.float16, dtypes.float32, dtypes.float64])
  @test_util.run_v2_only
  def testStatefulStandardNormalV2(self, dtype):
      """Use Anderson-Darling test to test distribution appears normal."""
      n = 10000
      shape = ops.convert_to_tensor([n], dtype=dtypes.int64)

      gen = random.Generator.from_seed(1234)
      state = gen.state
      resource_variable_ops.variable_accessed(state)
      algorithm = math_ops.cast(tf.random.Algorithm.PHILOX.value, dtypes.int64)
      x = gen_stateful_random_ops.stateful_standard_normal_v2(
      state.handle, algorithm, shape, dtype).numpy()

      # The constant 2.492 is the 5% critical value for the Anderson-Darling
      # test where the mean and variance are known. This test is probabilistic
      # so to avoid flakiness the seed is fixed.
      self.assertLess(
          random_test_util.anderson_darling(x.astype(float)), 2.492)

  @test_util.run_v2_only
  def testInputErrors(self):
      """Tests that proper errors are raised.
      """
      shape = [2, 3]
      gen = random.Generator.from_seed(1234)

      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          r"must have shape \[\], not"):
          gen_stateful_random_ops.rng_skip(
          gen.state.handle, gen.algorithm, [0, 0])

      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          r"must have shape \[\], not"):
          gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, [0, 0], shape)

      with self.assertRaisesWithPredicateMatch(
          TypeError, "EagerTensor of dtype int64"):
          gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 1.1, shape)

      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError, "Unsupported algorithm id"):
          gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 123, shape)

      var = variables.Variable([0, 0], dtype=dtypes.int32)
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "dtype of RNG state variable must be int64, not"):
          gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)

      var = variables.Variable([[0]], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "RNG state must have one and only one dimension, not"):
          gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)

      var = variables.Variable([0], dtype=dtypes.int64)
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "For the Philox algorithm, the size of state must be at least 3; got 1"):
          gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)

if __name__ == "__main__":
  config.set_soft_device_placement(True)
  test.main()
