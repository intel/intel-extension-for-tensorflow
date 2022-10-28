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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import multinomial


class MultinomialTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  @test_util.run_v1_only("b/120545219")
  def testSimpleShapes(self):
    with self.cached_session():
      for dtype in [np.float32, np.float64]:
        p = np.array([.1, .3, .6]).astype(dtype)
        dist = multinomial.Multinomial(total_count=1., probs=p)
        self.assertEqual(3, dist.event_shape_tensor().eval())
        self.assertAllEqual([], dist.batch_shape_tensor())
        self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
        self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

  @test_util.run_v1_only("b/120545219")
  def testComplexShapes(self):
    with self.cached_session():
      for dtype in [np.float32, np.float64]:
        p = 0.5 * np.ones([3, 2, 2], dtype=dtype)
        n = [[3., 2], [4, 5], [6, 7]]
        dist = multinomial.Multinomial(total_count=n, probs=p)
        self.assertEqual(2, dist.event_shape_tensor().eval())
        self.assertAllEqual([3, 2], dist.batch_shape_tensor())
        self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
        self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

  @test_util.run_v1_only("b/120545219")
  def testN(self):
    for dtype in [np.float32, np.float64]:
      p = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]).astype(dtype)
      n = [[3.], [4]]
      with self.cached_session():
        dist = multinomial.Multinomial(total_count=n, probs=p)
        self.assertEqual((2, 1), dist.total_count.get_shape())
        self.assertAllClose(n, dist.total_count)

  @test_util.run_v1_only("b/120545219")
  def testP(self):
    for dtype in [np.float32, np.float64]:
      p = np.array([[0.1, 0.2, 0.7]]).astype(dtype)
      with self.cached_session():
        dist = multinomial.Multinomial(total_count=3., probs=p)
        self.assertEqual((1, 3), dist.probs.get_shape())
        self.assertEqual((1, 3), dist.logits.get_shape())
        self.assertAllClose(p, dist.probs)

  @test_util.run_v1_only("b/120545219")
  def testLogits(self):
    for dtype in [np.float32, np.float64]:
      p = np.array([[0.1, 0.2, 0.7]], dtype=dtype)
      logits = np.log(p) - 50.
      with self.cached_session():
        multinom = multinomial.Multinomial(total_count=3., logits=logits)
        self.assertEqual((1, 3), multinom.probs.get_shape())
        self.assertEqual((1, 3), multinom.logits.get_shape())
        self.assertAllClose(p, multinom.probs)
        self.assertAllClose(logits, multinom.logits)

  @test_util.run_v1_only("b/120545219")
  def testPmfUnderflow(self):
    for dtype in [np.float32, np.float64]:
      logits = np.array([[-200, 0]], dtype=dtype)
      with self.cached_session():
        dist = multinomial.Multinomial(total_count=1., logits=logits)
        lp = dist.log_prob([1., 0.]).eval()[0]
        self.assertAllClose(-200, lp, atol=0, rtol=1e-6)

  @test_util.run_v1_only("b/120545219")
  def testPmfandCountsAgree(self):
    for dtype in [np.float32, np.float64]:
      p = np.array([[0.1, 0.2, 0.7]]).astype(dtype)
      n = [[5.]]
      with self.cached_session():
        dist = multinomial.Multinomial(total_count=n, probs=p, validate_args=True)
        dist.prob([2., 3, 0]).eval()
        dist.prob([3., 0, 2]).eval()
        with self.assertRaisesOpError("must be non-negative"):
          dist.prob([-1., 4, 2]).eval()
        with self.assertRaisesOpError("counts must sum to `self.total_count`"):
          dist.prob([3., 3, 0]).eval()

  @test_util.run_v1_only("b/120545219")
  def testPmfNonIntegerCounts(self):
    for dtype in [np.float32, np.float64]:
      p = np.array([[0.1, 0.2, 0.7]]).astype(dtype)
      n = [[5.]]
      with self.cached_session():
        # No errors with integer n.
        multinom = multinomial.Multinomial(
            total_count=n, probs=p, validate_args=True)
        multinom.prob([2., 1, 2]).eval()
        multinom.prob([3., 0, 2]).eval()
        # Counts don't sum to n.
        with self.assertRaisesOpError("counts must sum to `self.total_count`"):
          multinom.prob([2., 3, 2]).eval()
        # Counts are non-integers.
        x = array_ops.placeholder(dtypes.float32)
        with self.assertRaisesOpError(
            "cannot contain fractional components."):
          multinom.prob(x).eval(feed_dict={x: [1.0, 2.5, 1.5]})

        multinom = multinomial.Multinomial(
            total_count=n, probs=p, validate_args=False)
        multinom.prob([1., 2., 2.]).eval()
        # Non-integer arguments work.
        multinom.prob([1.0, 2.5, 1.5]).eval()

  def testPmfBothZeroBatches(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        # Both zero-batches.  No broadcast
        p = np.array([0.5, 0.5]).astype(dtype)
        counts = [1., 0]
        total_count = constant_op.constant(1.0, dtype=dtype)
        pmf = multinomial.Multinomial(total_count=total_count, probs=p).prob(counts)
        self.assertAllClose(0.5, self.evaluate(pmf))
        self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesNontrivialN(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        # Both zero-batches.  No broadcast
        p = np.array([0.1, 0.9]).astype(dtype)
        counts = [3., 2]
        total_count = constant_op.constant(5.0, dtype=dtype)
        dist = multinomial.Multinomial(total_count=total_count, probs=p)
        pmf = dist.prob(counts)
        # 5 choose 3 = 5 choose 2 = 10. 10 * (.9)^2 * (.1)^3 = 81/10000.
        self.assertAllClose(81. / 10000, self.evaluate(pmf))
        self.assertEqual((), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenSameRank(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        p = np.array([[0.1, 0.9]]).astype(dtype)
        counts = [[1., 0], [0, 1]]
        total_count = constant_op.constant(1.0, dtype=dtype)
        pmf = multinomial.Multinomial(total_count=total_count, probs=p).prob(counts)
        self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
        self.assertEqual((2), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenLowerRank(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        p = np.array([0.1, 0.9]).astype(dtype)
        counts = [[1., 0], [0, 1]]
        total_count = constant_op.constant(1.0, dtype=dtype)
        pmf = multinomial.Multinomial(total_count=total_count, probs=p).prob(counts)
        self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
        self.assertEqual((2), pmf.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        p = np.array([[0.1, 0.9], [0.7, 0.3]]).astype(dtype)
        counts = [[1., 0]]
        pmf = multinomial.Multinomial(total_count=1., probs=p).prob(counts)
        self.assertAllClose(pmf, [0.1, 0.7])
        self.assertEqual((2), pmf.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        p = np.array([[0.1, 0.9], [0.7, 0.3]]).astype(dtype)
        counts = [1., 0]
        pmf = multinomial.Multinomial(total_count=1., probs=p).prob(counts)
        self.assertAllClose(pmf, [0.1, 0.7])
        self.assertEqual(pmf.get_shape(), (2))

  def testPmfShapeCountsStretchedN(self):
    for dtype in [np.float32]:
      with self.cached_session():
        # [2, 2, 2]
        p = np.array([[[0.1, 0.9], [0.1, 0.9]], [[0.7, 0.3], [0.7, 0.3]]]).astype(dtype)
        # [2, 2]
        n = [[3., 3], [3, 3]]
        # [2]
        counts = [2., 1]
        pmf = multinomial.Multinomial(total_count=n, probs=p).prob(counts)
        self.evaluate(pmf)
        self.assertEqual(pmf.get_shape(), (2, 2))

  def testPmfShapeCountsPStretchedN(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        p = np.array([0.1, 0.9]).astype(dtype)
        counts = [3., 2]
        n = np.full([4, 3], 5., dtype=dtype)
        pmf = multinomial.Multinomial(total_count=n, probs=p).prob(counts)
        self.evaluate(pmf)
        self.assertEqual((4, 3), pmf.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testMultinomialMean(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        n = 5.
        p = np.array([0.1, 0.2, 0.7]).astype(dtype)
        dist = multinomial.Multinomial(total_count=n, probs=p)
        expected_means = 5 * np.array(p, dtype=dtype)
        self.assertEqual((3,), dist.mean().get_shape())
        self.assertAllClose(expected_means, dist.mean())

  @test_util.run_v1_only("b/120545219")
  def testMultinomialCovariance(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        n = 5.
        p = np.array([0.1, 0.2, 0.7]).astype(dtype)
        dist = multinomial.Multinomial(total_count=n, probs=p)
        expected_covariances = [[9. / 20, -1 / 10, -7 / 20],
                                [-1 / 10, 4 / 5, -7 / 10],
                                [-7 / 20, -7 / 10, 21 / 20]]
        self.assertEqual((3, 3), dist.covariance().get_shape())
        self.assertAllClose(expected_covariances, dist.covariance())

  @test_util.run_v1_only("b/120545219")
  def testMultinomialCovarianceBatch(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session():
        # Shape [2]
        n = [5.] * 2
        # Shape [4, 1, 2]
        p = np.array([[[0.1, 0.9]], [[0.1, 0.9]]] * 2).astype(dtype)
        dist = multinomial.Multinomial(total_count=n, probs=p)
        # Shape [2, 2]
        inner_var = [[9. / 20, -9 / 20], [-9 / 20, 9 / 20]]
        # Shape [4, 2, 2, 2]
        expected_covariances = [[inner_var, inner_var]] * 4
        self.assertEqual((4, 2, 2, 2), dist.covariance().get_shape())
        self.assertAllClose(expected_covariances, dist.covariance())

  def testCovarianceMultidimensional(self):
    for dtype in [np.float32, np.float64]:
      # Shape [3, 5, 4]
      p = np.random.dirichlet([.25, .25, .25, .25], [3, 5]).astype(dtype)
      # Shape [6, 3, 3]
      p2 = np.random.dirichlet([.3, .3, .4], [6, 3]).astype(dtype)

      ns = np.random.randint(low=1, high=11, size=[3, 5]).astype(dtype)
      ns2 = np.random.randint(low=1, high=11, size=[6, 1]).astype(dtype)

      with self.cached_session():
        dist = multinomial.Multinomial(ns, p)
        dist2 = multinomial.Multinomial(ns2, p2)

        covariance = dist.covariance()
        covariance2 = dist2.covariance()
        self.assertEqual((3, 5, 4, 4), covariance.get_shape())
        self.assertEqual((6, 3, 3, 3), covariance2.get_shape())

  @test_util.run_v1_only("b/120545219")
  def testCovarianceFromSampling(self):
    for dtype in [np.float32, np.float64]:
      # We will test mean, cov, var, stddev on a DirichletMultinomial constructed
      # via broadcast between alpha, n.
      theta = np.array([[1., 2, 3],
                        [2.5, 4, 0.01]], dtype=dtype)
      theta /= np.sum(theta, 1)[..., array_ops.newaxis]
      n = np.array([[10., 9.], [8., 7.], [6., 5.]], dtype=dtype)
      with self.cached_session() as sess:
        # batch_shape=[3, 2], event_shape=[3]
        dist = multinomial.Multinomial(n, theta)
        x = dist.sample(int(1000e3), seed=1)
        sample_mean = math_ops.reduce_mean(x, 0)
        x_centered = x - sample_mean[array_ops.newaxis, ...]
        sample_cov = math_ops.reduce_mean(math_ops.matmul(
            x_centered[..., array_ops.newaxis],
            x_centered[..., array_ops.newaxis, :]), 0)
        sample_var = array_ops.matrix_diag_part(sample_cov)
        sample_stddev = math_ops.sqrt(sample_var)
        [
            sample_mean_,
            sample_cov_,
            sample_var_,
            sample_stddev_,
            analytic_mean,
            analytic_cov,
            analytic_var,
            analytic_stddev,
        ] = sess.run([
            sample_mean,
            sample_cov,
            sample_var,
            sample_stddev,
            dist.mean(),
            dist.covariance(),
            dist.variance(),
            dist.stddev(),
        ])
        self.assertAllClose(sample_mean_, analytic_mean, atol=0.01, rtol=0.01)
        self.assertAllClose(sample_cov_, analytic_cov, atol=0.01, rtol=0.01)
        self.assertAllClose(sample_var_, analytic_var, atol=0.01, rtol=0.01)
        self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.01, rtol=0.01)

  @test_util.run_v1_only("b/120545219")
  def testSampleUnbiasedNonScalarBatch(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session() as sess:
        dist = multinomial.Multinomial(
            total_count=[7., 6., 5.],
            logits=math_ops.log(2. * self._rng.rand(4, 3, 2).astype(dtype)))
        n = int(3e4)
        x = dist.sample(n, seed=0)
        sample_mean = math_ops.reduce_mean(x, 0)
        # Cyclically rotate event dims left.
        x_centered = array_ops.transpose(x - sample_mean, [1, 2, 3, 0])
        sample_covariance = math_ops.matmul(
            x_centered, x_centered, adjoint_b=True) / n
        [
            sample_mean_,
            sample_covariance_,
            actual_mean_,
            actual_covariance_,
        ] = sess.run([
            sample_mean,
            sample_covariance,
            dist.mean(),
            dist.covariance(),
        ])
        self.assertAllEqual([4, 3, 2], sample_mean.get_shape())
        self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
        self.assertAllEqual([4, 3, 2, 2], sample_covariance.get_shape())
        self.assertAllClose(
            actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  @test_util.run_v1_only("b/120545219")
  def testSampleUnbiasedScalarBatch(self):
    for dtype in [np.float32, np.float64]:
      with self.cached_session() as sess:
        dist = multinomial.Multinomial(
            total_count=5.,
            logits=math_ops.log(2. * self._rng.rand(4).astype(dtype)))
        n = int(5e3)
        x = dist.sample(n, seed=0)
        sample_mean = math_ops.reduce_mean(x, 0)
        x_centered = x - sample_mean  # Already transposed to [n, 2].
        sample_covariance = math_ops.matmul(
            x_centered, x_centered, adjoint_a=True) / n
        [
            sample_mean_,
            sample_covariance_,
            actual_mean_,
            actual_covariance_,
        ] = sess.run([
            sample_mean,
            sample_covariance,
            dist.mean(),
            dist.covariance(),
        ])
        self.assertAllEqual([4], sample_mean.get_shape())
        self.assertAllClose(actual_mean_, sample_mean_, atol=0., rtol=0.10)
        self.assertAllEqual([4, 4], sample_covariance.get_shape())
        self.assertAllClose(
            actual_covariance_, sample_covariance_, atol=0., rtol=0.20)

  def testNotReparameterized(self):
    for dtype in [dtypes.float32]:
      total_count = constant_op.constant(5.0, dtype=dtype)
      p = constant_op.constant([0.2, 0.6], dtype=dtype)
      with backprop.GradientTape() as tape:
        tape.watch(total_count)
        tape.watch(p)
        dist = multinomial.Multinomial(
            total_count=total_count,
            probs=p)
        samples = dist.sample(100)
      grad_total_count, grad_p = tape.gradient(samples, [total_count, p])
      self.assertIsNone(grad_total_count)
      self.assertIsNone(grad_p)


if __name__ == "__main__":
  test.main()
