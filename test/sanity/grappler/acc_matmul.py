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
"""Tests for miscellaneous functionality in tensorflow.ops.nn."""

import functools
import math

from absl.testing import parameterized
import numpy as np
import os
import tensorflow as tf

from intel_extension_for_tensorflow.python.test_func import test as test_lib

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import gen_nn_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session

tf.compat.v1.disable_eager_execution()
os.environ["ITEX_LAYOUT_OPT"] = "0"
@test_util.run_deprecated_v1
class AccMatmulTest(test_lib.TestCase):

  def _npElu(self, np_features):
    return np.where(np_features < 0, np.exp(np_features) - 1, np_features)

  def testWithCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.rand(4,3).astype(dtypes.bfloat16.as_numpy_dtype)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)

    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = array_ops.identity(array_ops.identity(c1))

    with self.session() as sess:
      for i in range(2):
        sess.run(variables.global_variables_initializer())
        output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np).astype(np.float32)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXAccMatMul':
          if ((not node.attr['is_bf16_math_mode'].b) and
                node.attr['T'].type == dtypes.bfloat16._type_enum and
                node.attr['Tout'].type == dtypes.float32._type_enum and
                node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAdd(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.rand(4,3).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = np.random.rand(3).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)


    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = array_ops.identity(nn.bias_add(c1, badd_value))

    with self.session() as sess:
      for i in range(2):
        sess.run(variables.global_variables_initializer())
        output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np).astype(np.float32) + badd_value_np

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if ((not node.attr['is_bf16_math_mode'].b) and
              node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddActivation(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = (np.random.rand(4,3) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = (np.random.rand(3) - 0.5).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)


    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    c1 = nn.bias_add(c1, badd_value)
    d1 = array_ops.identity(nn_ops.elu(c1))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np).astype(np.float32) + badd_value_np
      expected = self._npElu(expected)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if ((not node.attr['is_bf16_math_mode'].b) and
              node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddAndAdd(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.rand(4,3).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = np.random.rand(3).astype(np.float32)
    d2_np = np.random.rand(3,3).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)
    d2 = constant_op.constant(d2_np)

    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1, d2])
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np).astype(np.float32) + badd_value_np \
        + d2_np

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if ((not node.attr['is_bf16_math_mode'].b) and
              node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddAndAdd2(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,3).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.rand(3,3).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = np.random.rand(3).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)

    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1, math_ops.cast(a, tf.float32)])
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np).astype(np.float32) + badd_value_np\
        + a_np.astype(np.float32)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if ((not node.attr['is_bf16_math_mode'].b) and node.attr['inplace_sum'].b and
              node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddAndAddActivation(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = (np.random.rand(4,3) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = (np.random.rand(3) - 0.5).astype(np.float32)
    d2_np = (np.random.rand(3,3) - 0.5).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)
    d2 = constant_op.constant(d2_np)

    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = nn.bias_add(c1, badd_value)
    e = nn_ops.elu(math_ops.add_n([d1, d2]))
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np).astype(np.float32) + badd_value_np \
        + d2_np
      expected = self._npElu(expected)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if ((not node.attr['is_bf16_math_mode'].b) and
              node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithMatmulCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(np.float32)
    b_np = np.random.rand(4,3).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)
    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = array_ops.identity(array_ops.identity(c1))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXAccMatMul':
          if (node.attr['is_bf16_math_mode'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithMatmulCastBiasAdd(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(np.float32)
    b_np = np.random.rand(4,3).astype(np.float32)
    badd_value_np = np.random.rand(3).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)
    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = array_ops.identity(nn.bias_add(c1, badd_value))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if (node.attr['is_bf16_math_mode'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithMatmulCastBiasAddActivation(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(np.float32)
    b_np = (np.random.rand(4,3) - 0.5).astype(np.float32)
    badd_value_np = (np.random.rand(3) - 0.5).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)
    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    c1 = nn.bias_add(c1, badd_value)
    d1 = array_ops.identity(nn_ops.elu(c1))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np
      expected = self._npElu(expected)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if (node.attr['is_bf16_math_mode'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithMatmulCastBiasAddAndAdd(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(np.float32)
    b_np = np.random.rand(4,3).astype(np.float32)
    badd_value_np = np.random.rand(3).astype(np.float32)
    d2_np = np.random.rand(3,3).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)
    d2 = constant_op.constant(d2_np)

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)

    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1, d2])
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np + d2_np

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if (node.attr['is_bf16_math_mode'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithMatmulCastBiasAddAndAdd2(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,3).astype(np.float32)
    b_np = np.random.rand(3,3).astype(np.float32)
    badd_value_np = np.random.rand(3).astype(np.float32)

    a = array_ops.identity(constant_op.constant(a_np))
    b = array_ops.identity(constant_op.constant(b_np))
    badd_value = constant_op.constant(badd_value_np)

    a16 = tf.cast(a, tf.bfloat16)
    b16 = tf.cast(b, tf.bfloat16)

    c1 = math_ops.matmul(a16, b16)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1, a])
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np + a_np

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if (node.attr['is_bf16_math_mode'].b and (not node.attr['inplace_sum'].b) and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithMatmulCastBiasAddAndAddActivation(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(np.float32)
    b_np = (np.random.rand(4,3) - 0.5).astype(np.float32)
    badd_value_np = (np.random.rand(3) - 0.5).astype(np.float32)
    d2_np = (np.random.rand(3,3) - 0.5).astype(np.float32)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)
    d2 = constant_op.constant(d2_np)

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)

    c1 = math_ops.matmul(a, b)
    c1 = math_ops.cast(c1, tf.float32)
    d1 = nn.bias_add(c1, badd_value)
    e = nn_ops.elu(math_ops.add_n([d1, d2]))
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np + d2_np
      expected = self._npElu(expected)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if (node.attr['is_bf16_math_mode'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.rand(4,3).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = np.random.rand(3).astype(dtypes.bfloat16.as_numpy_dtype)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)


    c1 = math_ops.matmul(a, b)
    c1 = nn.bias_add(c1, badd_value)
    d1 = array_ops.identity(math_ops.cast(c1, tf.float32))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = (np.matmul(a_np, b_np) + badd_value_np).astype(np.float32)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if ((not node.attr['is_bf16_math_mode'].b) and
              node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.bfloat16._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddActivationCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = (np.random.rand(4,3) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = \
      (np.random.rand(3) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)


    c1 = math_ops.matmul(a, b)
    c1 = nn_ops.elu(nn.bias_add(c1, badd_value))
    d1 = array_ops.identity(math_ops.cast(c1, tf.float32))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np
      expected = self._npElu(expected).astype(np.float32)

      # self.assertAllCloseAccordingToType(output_val, expected)
      # TODO(ITEX): uncomment after OneDNN support BF32
      self.assertAllClose(output_val, expected, rtol=1e-2, atol=1e-2)


      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if ((not node.attr['is_bf16_math_mode'].b) and
              node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.bfloat16._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddAndAddCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.rand(4,3).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = np.random.rand(3).astype(dtypes.bfloat16.as_numpy_dtype)
    d2_np = np.random.rand(3,3).astype(dtypes.bfloat16.as_numpy_dtype)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)
    d2 = constant_op.constant(d2_np)

    c1 = math_ops.matmul(a, b)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1, d2])
    e = math_ops.cast(e, tf.float32)
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = (np.matmul(a_np, b_np) +
        badd_value_np + d2_np).astype(np.float32)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if ((not node.attr['is_bf16_math_mode'].b) and  (not node.attr['inplace_sum'].b)
              and node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.bfloat16._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddAndAddCast2(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,3).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.rand(3,3).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = np.random.rand(3).astype(dtypes.bfloat16.as_numpy_dtype)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)

    c1 = math_ops.matmul(a, b)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1,a])
    e = array_ops.identity(math_ops.cast(e, tf.float32))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = (np.matmul(a_np, b_np) + badd_value_np
        + a_np).astype(np.float32)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if ((not node.attr['is_bf16_math_mode'].b) and (not node.attr['inplace_sum'].b)
              and node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.bfloat16._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testWithBiasAddAndAddActivationCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = (np.random.rand(4,3) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    badd_value_np = (np.random.rand(3) -
      0.5).astype(dtypes.bfloat16.as_numpy_dtype)
    d2_np = (np.random.rand(3,3) - 0.5).astype(dtypes.bfloat16.as_numpy_dtype)

    a = constant_op.constant(a_np)
    b = constant_op.constant(b_np)
    badd_value = constant_op.constant(badd_value_np)
    d2 = constant_op.constant(d2_np)

    c1 = math_ops.matmul(a, b)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1,d2])
    e = math_ops.cast(nn_ops.elu(e), tf.float32)
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np + d2_np
      expected = self._npElu(expected).astype(np.float32)

      self.assertAllCloseAccordingToType(output_val, expected)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if ((not node.attr['is_bf16_math_mode'].b) and (not node.attr['inplace_sum'].b)
              and node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.bfloat16._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithBiasAddCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(np.float32)
    b_np = np.random.rand(4,3).astype(np.float32)
    badd_value_np = np.random.rand(3).astype(np.float32)

    a = constant_op.constant(a_np) + 0.
    b = constant_op.constant(b_np) + 0.
    badd_value = constant_op.constant(badd_value_np) + 0.

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)
    badd_value = tf.cast(array_ops.identity(badd_value),tf.bfloat16)

    c1 = math_ops.matmul(a, b)
    c1 = nn.bias_add(c1, badd_value)
    d1 = array_ops.identity(math_ops.cast(c1, tf.float32))


    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np

      self.assertAllClose(output_val, expected, rtol=1e-2, atol=1e-2)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if (node.attr['is_bf16_math_mode'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithBiasAddActivationCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(np.float32)
    b_np = (np.random.rand(4,3) - 0.5).astype(np.float32)
    badd_value_np = (np.random.rand(3) - 0.5).astype(np.float32)

    a = constant_op.constant(a_np) + 0.
    b = constant_op.constant(b_np) + 0.
    badd_value = constant_op.constant(badd_value_np) + 0.

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)
    badd_value = tf.cast(array_ops.identity(badd_value),tf.bfloat16)

    c1 = math_ops.matmul(a, b)
    c1 = nn.bias_add(c1, badd_value)
    d1 = array_ops.identity(math_ops.cast(nn_ops.elu(c1), tf.float32))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(d1, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np
      expected = self._npElu(expected)

      self.assertAllClose(output_val, expected, rtol=1e-2, atol=1e-2)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMul':
          if (node.attr['is_bf16_math_mode'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithBiasAddAndAddCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,4).astype(np.float32)
    b_np = np.random.rand(4,3).astype(np.float32)
    badd_value_np = np.random.rand(3).astype(np.float32)
    d2_np = np.random.rand(3,3).astype(np.float32)

    a = constant_op.constant(a_np) + 0.
    b = constant_op.constant(b_np) + 0.
    badd_value = constant_op.constant(badd_value_np) + 0.
    d2 = constant_op.constant(d2_np) + 0.

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)
    badd_value = tf.cast(array_ops.identity(badd_value),tf.bfloat16)
    d2 = tf.cast(array_ops.identity(d2),tf.bfloat16)

    c1 = math_ops.matmul(a, b)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1,d2])
    e = math_ops.cast(e, tf.float32)
    e = array_ops.identity(e)

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np + d2_np

      self.assertAllClose(output_val, expected, rtol=1e-2, atol=1e-2)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if (node.attr['is_bf16_math_mode'].b and node.attr['inplace_sum'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithBiasAddAndAddCast2(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = np.random.rand(3,3).astype(np.float32)
    b_np = np.random.rand(3,3).astype(np.float32)
    badd_value_np = np.random.rand(3).astype(np.float32)

    a = constant_op.constant(a_np) + 0.
    b = constant_op.constant(b_np) + 0.
    badd_value = constant_op.constant(badd_value_np) + 0.

    a = tf.cast(array_ops.identity(a), tf.bfloat16)
    b = tf.cast(array_ops.identity(b), tf.bfloat16)
    badd_value = tf.cast(array_ops.identity(badd_value), tf.bfloat16)

    c1 = math_ops.matmul(a, b)
    d1 = nn.bias_add(c1, badd_value)
    e = math_ops.add_n([d1, a])
    e = array_ops.identity(math_ops.cast(e, tf.float32))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np + a_np

      self.assertAllClose(output_val, expected, rtol=1e-2, atol=1e-2)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if (node.attr['is_bf16_math_mode'].b and (not node.attr['inplace_sum'].b) and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

  def testCastWithBiasAddAndAddActivationCast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    a_np = (np.random.rand(3,4) - 0.5).astype(np.float32)
    b_np = (np.random.rand(4,3) - 0.5).astype(np.float32)
    badd_value_np = (np.random.rand(3) - 0.5).astype(np.float32)
    d2_np = (np.random.rand(3,3) - 0.5).astype(np.float32)

    a = constant_op.constant(a_np) + 0.
    b = constant_op.constant(b_np) + 0.
    badd_value = constant_op.constant(badd_value_np) + 0.
    d2 = constant_op.constant(d2_np) + 0.

    a = tf.cast(array_ops.identity(a),tf.bfloat16)
    b = tf.cast(array_ops.identity(b),tf.bfloat16)
    badd_value = tf.cast(array_ops.identity(badd_value),tf.bfloat16)
    d2 = tf.cast(array_ops.identity(d2),tf.bfloat16)

    c1 = math_ops.matmul(a, b)
    d1 = nn.bias_add(c1, badd_value)
    e = nn_ops.elu(math_ops.add_n([d1, d2]))
    e = array_ops.identity(math_ops.cast(e, tf.float32))

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(e, options=run_options, run_metadata=metadata)
      graph = metadata.partition_graphs[0]
      expected = np.matmul(a_np, b_np) + badd_value_np + d2_np
      expected = self._npElu(expected)

      self.assertAllClose(output_val, expected, rtol=1e-2, atol=1e-2)

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulWithSum':
          if (node.attr['is_bf16_math_mode'].b and node.attr['inplace_sum'].b and
              node.attr['T'].type == dtypes.float32._type_enum and
              node.attr['Tout'].type == dtypes.float32._type_enum and
              node.attr['Tpost'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      self.assertTrue(existing_pattern)

@test_util.run_deprecated_v1
class AccMatmulGradTest(test_lib.TestCase):
  def testFusedMatMulGradCastBias(self):
    # TODO(itex): Remove this restriction when FusedMatMulGrad is fixed
    if not test_lib.is_gpu_available():
      self.skipTest("No GPU available")
    np.random.seed(0)
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()
    a_shape = [2, 5]
    b_shape = [5, 3]
    dz_shape = [2, 3]

    a_np = np.random.normal(size=a_shape).astype(dtypes.bfloat16.as_numpy_dtype)
    b_np = np.random.normal(size=b_shape).astype(dtypes.bfloat16.as_numpy_dtype)
    dz_np = np.random.normal(size=dz_shape).astype(dtypes.bfloat16.as_numpy_dtype)

    a = tf.Variable(a_np)
    b = tf.Variable(b_np)
    dz = array_ops.identity(tf.Variable(dz_np))

    ga = math_ops.matmul(dz, b, transpose_b=True)
    gb = math_ops.matmul(a, dz, transpose_a=True)
    gbias = gen_nn_ops.bias_add_grad(dz)

    ga = array_ops.identity(tf.cast(ga, tf.float32))
    gb = array_ops.identity(tf.cast(gb, tf.float32))
    gbias = array_ops.identity(tf.cast(gbias, tf.float32))
    result = array_ops.identity(nn_ops.bias_add(math_ops.matmul(ga, gb), gbias))

    # np.finfo doesn't support bfloat16. So, we manually compute the eps which
    # defines the difference between 1.0 and the next smallest representable
    # float larger than 1.0. For bfloat16, the difference is 1/128.
    if a_np.dtype == dtypes.bfloat16.as_numpy_dtype:
      epsilon = 0.0078125
    else:
      epsilon = np.finfo(a_np.dtype).eps
    delta = epsilon**(1.0 / 3.0)
    tol = 20 * delta

    with self.session() as sess:
      sess.run(variables.global_variables_initializer())
      output_val = sess.run(result, options=run_options, run_metadata=metadata)
      expected = [[4.893589, 0.762014, 0.607488],
                  [3.082745, -0.287658, 0.405976]]
      self.assertAllClose(output_val, expected, rtol=tol, atol=tol)
      graph = metadata.partition_graphs[0]

      existing_pattern = False
      for node in graph.node:
        if node.op == '_ITEXFusedAccMatMulGrad':
          if (node.attr['T'].type == dtypes.bfloat16._type_enum and
              node.attr['Tgrad'].type == dtypes.float32._type_enum):
            existing_pattern = True
          break
      if test_lib.is_gpu_available():
        self.skipTest("This fusion on GPU side is canceled.") 
      self.assertTrue(existing_pattern)


if __name__ == "__main__":
  test_lib.main()
