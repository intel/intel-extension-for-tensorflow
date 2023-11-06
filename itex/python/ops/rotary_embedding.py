# Copyright (c) 2023 Intel Corporation
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
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.framework import ops
from typing import List, Optional, Union
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.framework import config

def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def rotate_every_two(x: tf.Tensor) -> tf.Tensor:
    rotate_half_tensor = tf.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), axis=-1)
    new_shape = shape_list(rotate_half_tensor)[:-2] + [tf.math.reduce_prod(shape_list(rotate_half_tensor)[-2:])]
    rotate_half_tensor = tf.reshape(rotate_half_tensor, new_shape)
    return rotate_half_tensor

def apply_rotary_pos_emb(tensor,sin,cos):
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

@keras.utils.generic_utils.register_keras_serializable(package="Itex")
def qk_rotary_positional_embedding(q,k,sin,cos, rotary_dim=64,num_attention_heads=16,head_dim=256, name=None):
  if config.list_logical_devices('XPU'):
    with ops.name_scope(name, "qk_rotary_positional_embedding", [q,k,sin,cos]):
      q = ops.convert_to_tensor(q, name="query")
      k = ops.convert_to_tensor(k, name="key")
      sin = ops.convert_to_tensor(sin, name="sin")
      cos = ops.convert_to_tensor(cos, name="cos")
      return load_ops_library.qk_rotary_positional_embedding(q,k,sin,cos,rotary_dim=rotary_dim,num_attention_heads=num_attention_heads,head_dim=head_dim)
  else:
    k_rot = k[:, :, :, : rotary_dim]
    k_pass = k[:, :, :, rotary_dim :]

    q_rot = q[:, :, :, : rotary_dim]
    q_pass = q[:, :, :, rotary_dim :]

    k_rot = apply_rotary_pos_emb(k_rot, sin,cos)
    q_rot = apply_rotary_pos_emb(q_rot, sin,cos)

    result_k = tf.concat((k_rot, k_pass), axis=-1)
    result_q = tf.concat((q_rot, q_pass), axis=-1)
    return (result_q,result_k)
