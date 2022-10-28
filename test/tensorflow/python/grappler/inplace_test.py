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


from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

import numpy as np
import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

# TODO(yifeng): Support both plain and block format tests in the future.
# Test plain format.
os.environ['ITEX_LAYOUT_OPT']="0"

def GetRandomNormalInput(shape, dtype):
  # float16 has limited range so we reduce the variance of the scalars.
  scale = 10.0 if dtype != np.float16 else 0.1
  loc = -10.0 if dtype != np.float16 else 0.1
  vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
  if dtype in (np.complex64, np.complex128):
    imag = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    vals += 1j * imag
  return vals.reshape(shape)

def GetInplaceFlagByName(graph, tgt_node_name, attr_name):
  for node in graph.node:
    if (node.name == tgt_node_name):
      return node.attr[attr_name].b
  return False

class InplaceTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testRefCountEqualsToOne(self):
    # TODO(yifeng): Remove this limitation in the future.
    if test.is_gpu_available():
      self.skipTest("Softmax in-place is temporarily unavailable on GPU")

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    phl_in = GetRandomNormalInput([3, 4], np.float32)
    phr_in = GetRandomNormalInput([4, 3], np.float32)

    tgt_name = "arbitrary"

    with self.cached_session() as sess:
      phl = array_ops.placeholder(np.float32)
      phr = array_ops.placeholder(np.float32)
      mm = math_ops.matmul(phl, phr)
      tf_softmax = array_ops.identity(nn_ops.softmax(mm, name=tgt_name))

      out = sess.run([tf_softmax], feed_dict={phl: phl_in, phr: phr_in},
                     options=run_options, run_metadata=metadata)

    graph = metadata.partition_graphs[0]
    inplace_flag = GetInplaceFlagByName(graph, tgt_name, "is_inplace")

    self.assertTrue(inplace_flag)

  @test_util.run_deprecated_v1
  def testRefCountGreaterThanOne(self):
    # TODO(yifeng): Remove this limitation in the future.
    if test.is_gpu_available():
      self.skipTest("Softmax in-place is temporarily unavailable on GPU")

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    phl_in = GetRandomNormalInput([3, 4], np.float32)
    phr_in = GetRandomNormalInput([4, 5], np.float32)
    bias_in = GetRandomNormalInput([5], np.float32)

    tgt_name = "arbitrary"

    with self.cached_session() as sess:
      phl = array_ops.placeholder(np.float32)
      phr = array_ops.placeholder(np.float32)
      bias = array_ops.placeholder(np.float32)

      mm = math_ops.matmul(phl, phr)
      bias_add = math_ops.add(mm, bias)
      tf_softmax = array_ops.identity(nn_ops.softmax(mm, name=tgt_name))

      sess.run([tf_softmax, bias_add],
               feed_dict={phl: phl_in, phr: phr_in, bias: bias_in},
               options=run_options, run_metadata=metadata)

    graph = metadata.partition_graphs[0]
    inplace_flag = GetInplaceFlagByName(graph, tgt_name, "is_inplace")

    self.assertFalse(inplace_flag)


class InplaceSumTest(test.TestCase):

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add has refcount 1.
  @test_util.run_deprecated_v1
  def testAddWithRefCountOne(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    phl_in = GetRandomNormalInput([3, 3], np.float32)
    phr_in = GetRandomNormalInput([3, 3], np.float32)
    bias_in = GetRandomNormalInput([3], np.float32)
    pha_in = GetRandomNormalInput([3, 3], np.float32)

    tgt_name = "arbitrary"

    with self.cached_session() as sess:
      # Input tensor shape should be specified explicitly, otherwise
      # `math_ops.matmul` may return BatchMatMulV2
      phl = array_ops.placeholder(np.float32, [3, 3])
      phr = array_ops.placeholder(np.float32, [3, 3])
      bias = array_ops.placeholder(np.float32, [3])
      # AddInput placeholder
      pha = array_ops.placeholder(np.float32, [3, 3])

      relu = nn_ops.relu(pha)

      mm = math_ops.matmul(phl, phr)
      bias_add = nn_ops.bias_add(mm, bias)
      add = array_ops.identity(math_ops.add_n([bias_add, relu], name=tgt_name))

      sess.run([add],
          feed_dict={phl: phl_in, phr: phr_in, bias: bias_in, pha: pha_in},
          options=run_options, run_metadata=metadata)

    graph = metadata.partition_graphs[0]
    inplace_flag = GetInplaceFlagByName(graph, tgt_name, "inplace_sum")

    self.assertTrue(inplace_flag)

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add has a total refcount of 2, and Add is its last consumer.
  @test_util.run_deprecated_v1
  def testAddWithRefCountTwoAndRunAddLast(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    phl_in = GetRandomNormalInput([3, 3], np.float32)
    phr_in = GetRandomNormalInput([3, 3], np.float32)
    bias_in = GetRandomNormalInput([3], np.float32)

    tgt_name = "arbitrary"

    with self.cached_session() as sess:
      phl = array_ops.placeholder(np.float32, [3, 3])
      phr = array_ops.placeholder(np.float32, [3, 3])
      bias = array_ops.placeholder(np.float32, [3])
      # AddInput placeholder

      elu = nn_ops.elu(phl)
      relu = nn_ops.relu(phl)

      mm = math_ops.matmul(relu, phr)
      bias_add = nn_ops.bias_add(mm, bias)
      add = array_ops.identity(math_ops.add_n([bias_add, elu], name=tgt_name))

      sess.run([add], feed_dict={phl: phl_in, phr: phr_in, bias: bias_in},
               options=run_options, run_metadata=metadata)

    graph = metadata.partition_graphs[0]
    inplace_flag = GetInplaceFlagByName(graph, tgt_name, "inplace_sum")

    self.assertTrue(inplace_flag)

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add has refcount 2, and there is no dependency between its two consumers.
  @test_util.run_deprecated_v1
  def testAddWithRefCountTwoAndNoDependence(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    phl_in = GetRandomNormalInput([3, 3], np.float32)
    phr_in = GetRandomNormalInput([3, 3], np.float32)
    bias_in = GetRandomNormalInput([3], np.float32)
    pha_in = GetRandomNormalInput([3, 3], np.float32)

    tgt_name = "arbitrary"

    with self.cached_session() as sess:
      phl = array_ops.placeholder(np.float32, [3, 3])
      phr = array_ops.placeholder(np.float32, [3, 3])
      bias = array_ops.placeholder(np.float32, [3])
      # AddInput placeholder
      pha = array_ops.placeholder(np.float32, [3, 3])

      mm = math_ops.matmul(phl, phr)
      bias_add = nn_ops.bias_add(mm, bias)
      add = array_ops.identity(math_ops.add_n([bias_add, pha], name=tgt_name))

      relu = nn_ops.relu(pha)

      sess.run([add, relu],
          feed_dict={phl: phl_in, phr: phr_in, bias: bias_in, pha: pha_in},
          options=run_options, run_metadata=metadata)

    graph = metadata.partition_graphs[0]
    inplace_flag = GetInplaceFlagByName(graph, tgt_name, "inplace_sum")

    self.assertFalse(inplace_flag)

  # Tests tensor forwarding of a fused Conv2D+BiasAdd+Add op when the input to
  # Add is the same as the input to the fused Conv2D op and needs a tensor
  # buffer.
  @test_util.run_deprecated_v1
  def testAddWithSameSrcAndAddTensorBuffer(self):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    metadata = config_pb2.RunMetadata()

    phl_in = GetRandomNormalInput([3, 3], np.float32)
    phr_in = GetRandomNormalInput([3, 3], np.float32)
    bias_in = GetRandomNormalInput([3], np.float32)

    tgt_name = "arbitrary"

    with self.cached_session() as sess:
      phl = array_ops.placeholder(np.float32, [3, 3])
      phr = array_ops.placeholder(np.float32, [3, 3])
      bias = array_ops.placeholder(np.float32, [3])
      # AddInput placeholder

      relu = nn_ops.relu(phl)

      mm = math_ops.matmul(relu, phr)
      bias_add = nn_ops.bias_add(mm, bias)
      add = array_ops.identity(math_ops.add_n([bias_add, relu], name=tgt_name))

      sess.run([add], feed_dict={phl: phl_in, phr: phr_in, bias: bias_in},
               options=run_options, run_metadata=metadata)

    graph = metadata.partition_graphs[0]
    inplace_flag = GetInplaceFlagByName(graph, tgt_name, "inplace_sum")

    self.assertFalse(inplace_flag)

if __name__ == "__main__":
  test.main()
