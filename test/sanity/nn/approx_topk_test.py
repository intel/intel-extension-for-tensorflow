# Copyright (c) 2021-2023 Intel Corporation
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for approx_topk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import itertools
from absl.testing import parameterized

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager.def_function import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


@test_util.run_all_in_graph_and_eager_modes
class ApproxTopkTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def setUp(self):
    test_util.TensorFlowTestCase.setUp(self)
    self._rng = np.random.default_rng(42)

  def compute_recall(self, result_neighbors, ground_truth_neighbors):
    """Computes the recall of an approximate nearest neighbor search.

    Args:
      result_neighbors: int32 numpy array of the shape [num_queries,
        neighbors_per_query] where the values are the indices of the dataset.
      ground_truth_neighbors: int32 numpy array of with shape [num_queries,
        ground_truth_neighbors_per_query] where the values are the indices of
        the dataset.

    Returns:
      The recall.
    """
    self.assertLen(result_neighbors.shape, 2)
    self.assertLen(ground_truth_neighbors.shape, 2)
    self.assertEqual(result_neighbors.shape[0], ground_truth_neighbors.shape[0])
    gt_sets = [set(np.asarray(x)) for x in ground_truth_neighbors]

    def hits_per_q(q, nn_per_q):
      return len(list(x for x in nn_per_q if x.item() in gt_sets[q]))

    hits = sum(
        hits_per_q(q, nn_per_q) for q, nn_per_q in enumerate(result_neighbors))
    return hits / ground_truth_neighbors.size

  @parameterized.parameters(
      itertools.product(
          [dtypes.bfloat16, dtypes.float16, dtypes.float32],
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
          [True, False],  # aggregate_to_topk
      ))
  def test_non_fused_max_k(self, dtype, k, row_size, num_rows,
                           aggregate_to_topk):
    if not test.is_gpu_available():
      self.skipTest('CPU do not support')

    row = np.arange(row_size, dtype=np.float32)
    db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))
    db_op = constant_op.constant(db, dtype=dtype)

    def ann(db, k):
      return nn_ops.approx_max_k(db, k, aggregate_to_topk=aggregate_to_topk)

    with test_util.force_gpu():
      _, idx = self.evaluate(ann(db_op, k))
      gt = np.argsort(-db)[:, :k]
      ann_recall = self.compute_recall(idx, gt)
      self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [dtypes.float32],  # Use float32 for numerical stability.
          [1, 10],  # k
          [100, 500],  # db_size
          [1, 10, 128],  # qy_size
          [2, 32],  # feature dim
      ))
  # MIPS = Maximal Inner Product Search
  def test_mips(self, dtype, k, db_size, qy_size, feature_dim):
    if not test.is_gpu_available():
      self.skipTest('CPU do not support')

    qy = self._rng.random([qy_size, feature_dim])
    db = self._rng.random([db_size, feature_dim])
    qy_op = constant_op.constant(qy, dtype=dtype)
    db_op = constant_op.constant(db, dtype=dtype)
    
    def ann(qy, db, k):
      scores = math_ops.matmul(qy, db, transpose_b=True)
      return nn_ops.approx_max_k(scores, k)

    with test_util.force_gpu():
      _, idx = self.evaluate(ann(qy_op, db_op, k))
      scores = self.evaluate(-math_ops.matmul(qy_op, db_op, transpose_b=True))
      gt = np.argsort(scores)[:, :k]
      ann_recall = self.compute_recall(idx, gt)
      self.assertGreaterEqual(ann_recall, 0.95)

  @parameterized.parameters(
      itertools.product(
          [dtypes.float16, dtypes.bfloat16, dtypes.float32],
          [1, 10],  # k
          [100, 500],  # row_size
          [1, 10, 128],  # num_rows
      )
  )
  def test_nonjit(self, dtype, k, row_size, num_rows):
    if not test.is_gpu_available():
      self.skipTest('CPU do not support')

    # Support regular topk semantics.
    with test_util.force_gpu():
      row = np.arange(row_size, dtype=np.float32)
      db = np.stack(list(self._rng.permutation(row) for _ in range(num_rows)))
      db_tensor = constant_op.constant(db, dtype=dtype)
      _, idx = self.evaluate(nn_ops.approx_max_k(db_tensor, k))
      sorted_idx = np.sort(idx)
      expected = np.sort(np.argsort(-db)[:, :k])
      self.assertAllEqual(sorted_idx, expected)


if __name__ == '__main__':
  test.main()
