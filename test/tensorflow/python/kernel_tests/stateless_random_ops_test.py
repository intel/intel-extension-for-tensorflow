# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stateless random ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import functools

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.ops import gen_stateless_random_ops

def stateless_random_normal(shape,
                            seed,
                            mean=0.0,
                            stddev=1.0,
                            dtype=dtypes.float32,
                            name=None):
  """
  stateless_random_ops now only call v2 version.
  Here is a py implementation calling v1 version for testing.
  """
  shape = tensor_util.shape_tensor(shape)
  mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
  stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
  rnd = gen_stateless_random_ops.stateless_random_normal(
      shape, seed, dtype=dtype)
  result = math_ops.add(rnd * stddev, mean, name=name)
  tensor_util.maybe_set_static_shape(result, shape)
  return result

def stateless_truncated_normal(shape,
                               seed,
                               mean=0.0,
                               stddev=1.0,
                               dtype=dtypes.float32,
                               name=None):
  """
  stateless_random_ops now only call v2 version.
  Here is a py implementation calling v1 version for testing.
  """
  shape = tensor_util.shape_tensor(shape)
  mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
  stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
  rnd = gen_stateless_random_ops.stateless_truncated_normal(
      shape, seed, dtype=dtype)
  result = math_ops.add(rnd * stddev, mean, name=name)
  tensor_util.maybe_set_static_shape(result, shape)
  return result

def stateless_random_uniform(shape,
                             seed,
                             minval=0,
                             maxval=None,
                             dtype=dtypes.float32,
                             name=None):
  """
  stateless_random_ops now only call v2 version.
  Here is a py implementation calling v1 version for testing.
  """
  dtype = dtypes.as_dtype(dtype)
  accepted_dtypes = (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                     dtypes.float64, dtypes.int32, dtypes.int64, dtypes.uint32,
                     dtypes.uint64)
  if dtype not in accepted_dtypes:
    raise ValueError(
        f"Argument `dtype` got invalid value {dtype}. Accepted dtypes are "
        f"{accepted_dtypes}.")
  if dtype.is_integer:
    if (minval is None) != (maxval is None):
      raise ValueError(
          f"For integer `dtype` argument {dtype}, argument `minval` and "
          f"`maxval` must be both None or not None. Got `minval`={minval} and "
          f"`maxval`={maxval}.")
    if minval is not None and dtype in (dtypes.uint32, dtypes.uint64):
      raise ValueError(
          f"Argument `dtype` got invalid value {dtype} when argument `minval` "
          f"is not None. Please don't use unsigned integers in this case.")
  elif maxval is None:
    maxval = 1
  shape = tensor_util.shape_tensor(shape)
  if dtype.is_integer and minval is None:
    result = (
        gen_stateless_random_ops.stateless_random_uniform_full_int(
            shape, seed, dtype=dtype, name=name))
  else:
    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    if dtype.is_integer:
      result = gen_stateless_random_ops.stateless_random_uniform_int(
          shape,
          seed,
          minval=minval,
          maxval=maxval,
          name=name)
    else:
      rnd = gen_stateless_random_ops.stateless_random_uniform(
          shape, seed, dtype=dtype)
      result = math_ops.add(rnd * (maxval - minval), minval, name=name)
  tensor_util.maybe_set_static_shape(result, shape)
  return result

def invert_philox(key, value):
  """Invert the Philox bijection."""
  key = np.array(key, dtype=np.uint32)
  value = np.array(value, dtype=np.uint32)
  step = np.array([0x9E3779B9, 0xBB67AE85], dtype=np.uint32)
  for n in range(10)[::-1]:
    key0, key1 = key + n * step
    v0 = value[3] * 0x991a7cdb & 0xffffffff
    v2 = value[1] * 0x6d7cae67 & 0xffffffff
    hi0 = v0 * 0xD2511F53 >> 32
    hi1 = v2 * 0xCD9E8D57 >> 32
    v1 = hi1 ^ value[0] ^ key0
    v3 = hi0 ^ value[2] ^ key1
    value = v0, v1, v2, v3
  return np.array(value)


class StatelessOpsTest(test.TestCase):

  def _test_match(self, cases):
    # Stateless ops should be the same as stateful ops on the first call
    # after seed scrambling.
    cases = tuple(cases)
    key = 0x3ec8f720, 0x02461e29
    for seed in (7, 17), (11, 5), (2, 3):
      preseed = invert_philox(key, (seed[0], 0, seed[1], 0)).astype(np.uint64)
      preseed = preseed[::2] | preseed[1::2] << 32
      random_seed.set_random_seed(seed[0])
      with test_util.use_gpu():
        for stateless_op, stateful_op in cases:
          stateful = stateful_op(seed=seed[1])
          pure = stateless_op(seed=preseed)
          self.assertAllClose(self.evaluate(stateful), self.evaluate(pure))

  def _test_determinism(self, cases):
    # Stateless values should be equal iff the seeds are equal (roughly)
    cases = tuple(cases)
    with self.test_session(use_gpu=True):
      for seed_type in [dtypes.int32, dtypes.int64]:
        seed_t = array_ops.placeholder(seed_type, shape=[2])
        seeds = [(x, y) for x in range(5) for y in range(5)] * 3
        for stateless_op, _ in cases:
          pure = stateless_op(seed=seed_t)
          values = [
              (seed, pure.eval(feed_dict={seed_t: seed})) for seed in seeds
          ]
          for s0, v0 in values:
            for s1, v1 in values:
              self.assertEqual(s0 == s1, np.all(v0 == v1))

  def _float_cases(self, shape_dtypes=(None,)):
    float_cases = (
        # Uniform distribution, with and without range
        (stateless_random_uniform, random_ops.random_uniform, {}),
        (stateless_random_uniform, random_ops.random_uniform,
         dict(minval=2.2, maxval=7.1)),
        # Normal distribution, with and without mean+stddev
        (stateless_random_normal, random_ops.random_normal, {}),
        (stateless_random_normal, random_ops.random_normal,
         dict(mean=2, stddev=3)),
        # Truncated normal distribution, with and without mean+stddev
        (stateless_truncated_normal, random_ops.truncated_normal, {}),
        (stateless_truncated_normal, random_ops.truncated_normal,
         dict(mean=3, stddev=4)),
        # run Uniform distribution v2 implictly, with and without range
        (stateless.stateless_random_uniform, random_ops.random_uniform, {}),
        (stateless.stateless_random_uniform, random_ops.random_uniform,
         dict(minval=2.2, maxval=7.1)),
        # run Normal distribution v2 implictly, with and without mean+stddev
        (stateless.stateless_random_normal, random_ops.random_normal, {}),
        (stateless.stateless_random_normal, random_ops.random_normal,
         dict(mean=2, stddev=3)),
        # run Truncated normal distribution v2 implictly, with and without mean+stddev
        (stateless.stateless_truncated_normal, random_ops.truncated_normal, {}),
        (stateless.stateless_truncated_normal, random_ops.truncated_normal,
         dict(mean=3, stddev=4)),
    )
    for dtype in dtypes.float16, dtypes.float32, dtypes.float64:
      for shape_dtype in shape_dtypes:
        for shape in (), (3,), (2, 5):
          if shape_dtype is not None:
            shape = constant_op.constant(shape, dtype=shape_dtype)
          for stateless_op, stateful_op, kwds in float_cases:
            kwds = dict(shape=shape, dtype=dtype, **kwds)
            yield (functools.partial(stateless_op, **kwds),
                   functools.partial(stateful_op, **kwds))

  def _int_cases(self, shape_dtypes=(None,)):
    for shape_dtype in shape_dtypes:
      for shape in (), (3,), (2, 5):
        if shape_dtype is not None:
          shape = constant_op.constant(shape, dtype=shape_dtype)
        for dtype in dtypes.int32, dtypes.int64:
          kwds = dict(minval=2, maxval=11111, dtype=dtype, shape=shape)
          yield (functools.partial(stateless.stateless_random_uniform, **kwds),
                 functools.partial(random_ops.random_uniform, **kwds))

  def _multinomial_cases(self):
    num_samples = 10
    for logits_dtype in np.float16, np.float32, np.float64:
      for output_dtype in dtypes.int32, dtypes.int64:
        for logits in ([[0.1, 0.25, 0.5, 0.15]], [[0.5, 0.5], [0.8, 0.2],
                                                  [0.25, 0.75]]):
          kwds = dict(
              logits=constant_op.constant(logits, dtype=logits_dtype),
              num_samples=num_samples,
              output_dtype=output_dtype)
          yield (functools.partial(stateless.stateless_multinomial, **kwds),
                 functools.partial(random_ops.multinomial, **kwds))

  @test_util.run_deprecated_v1
  def testMatchFloat(self):
    self._test_match(self._float_cases())

  @test_util.run_deprecated_v1
  def testMatchInt(self):
    self._test_match(self._int_cases())

  @test_util.run_deprecated_v1
  def testMatchMultinomial(self):
    self._test_match(self._multinomial_cases())

  @test_util.run_deprecated_v1
  def testDeterminismFloat(self):
    self._test_determinism(
        self._float_cases(shape_dtypes=(dtypes.int32, dtypes.int64)))

  @test_util.run_deprecated_v1
  def testDeterminismInt(self):
    self._test_determinism(
        self._int_cases(shape_dtypes=(dtypes.int32, dtypes.int64)))

  @test_util.run_deprecated_v1
  def testDeterminismMultinomial(self):
    self._test_determinism(self._multinomial_cases())


if __name__ == '__main__':
  test.main()
