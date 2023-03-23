# Copyright (c) 2021 Intel Corporation
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

# pylint: disable=missing-module-docstring
from tensorflow.python import keras
from tensorflow.python.framework import ops
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

@keras.utils.generic_utils.register_keras_serializable(package="Itex")
def gelu(features, approximate=False, name=None):
  """Applies the Gaussian error linear unit (GELU) activation function.

  Gaussian error linear unit (GELU) computes
  `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
  The (GELU) nonlinearity weights inputs by their value, rather than gates
  inputs by their sign as in ReLU.

  This python api itex.ops.gelu(x, approximate=approximate) replaces
  tf.nn.gelu(x, approximate=approximate)

  For example:

  >>> import intel_extension_for_tensorflow as itex
  >>> x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
  >>> y = itex.ops.gelu(x)
  >>> y.numpy()
  array([-0.00404969, -0.15865526,  0.        ,  0.8413447 ,  2.9959502 ],
        dtype=float32)
  >>> y = itex.ops.gelu(x, approximate=True)
  >>> y.numpy()
  array([-0.00363725, -0.158808  ,  0.        ,  0.841192  ,  2.9963627 ],
        dtype=float32)

  Args:
      x: Input tensor.
      approximate: A `bool`, whether to enable approximation.

  Returns:
      The gaussian error linear activation:
      `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
      if `approximate` is `True` or
      `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
      where `P(X) ~ N(0, 1)`,
      if `approximate` is `False`.

  Reference:
    - [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
  """
  with ops.name_scope(name, "Gelu", [features]):
    features = ops.convert_to_tensor(features, name="features")
    return load_ops_library.itex_gelu(features, approximate)
