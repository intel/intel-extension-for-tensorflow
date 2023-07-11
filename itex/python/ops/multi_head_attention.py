#Copyright 2023 The TensorFlow Authors.All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
"""Keras-based multi-head attention layer."""


import collections
import math
import string
import functools
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from intel_extension_for_tensorflow.python.device import is_xehpc
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

_CHR_IDX = string.ascii_lowercase

def _stateless_dropout(input_tensor, dropout_prob, seed):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.experimental.stateless_dropout(input_tensor, rate=dropout_prob, seed=seed)
  return output


def scaled_dot_product_attention(query, key, value, atten_mask=None, dropout_p=0.0, seed=(2,3), is_causal=False, use_fast_attention=True):
    """Applies Dot-product attention with query, key, value tensors.

        Args:
            query: Projected query `Tensor` of shape `(B, N, F, head_size)`.
            key: Projected key `Tensor` of shape `(B, N, T, head_size)`.
            value: Projected value `Tensor` of shape `(B, N, T, head_size)`.
            atten_mask (optinal Tensor): a boolean mask of shape `(B, F, T)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            dropout_p (float): dropout probability, if greater than 0.0, dropout is applied
            seed ([int, int]): seed for dropout
            is_causal (bool): If true, assumes causal attention masking and errors if both atten_mask and is_causal are set.
            use_fast_attention (bool): if true, use core op, otherwise use naive small ops implementation

        Returns:
          atten_output: Multi-headed outputs of attention computation.
    """ 
#TODO : remove is_causal limitation once flash attention backward is supported  
    q_seq_len = query.shape[2]
    head_size = query.shape[3]
    can_use_fast_sdp = is_xehpc() and (query.dtype == tf.bfloat16 or query.dtype == tf.float16) and (is_causal == False) and  \
        (q_seq_len <= 512) and (head_size <= 128)
    def sdp():
        i_dtype = query.dtype
        
        atten_scores = tf.matmul(query, key, transpose_b=True)
        head_size = query.shape[3]
        head_scale = 1.0 / tf.sqrt(float(head_size))
        atten_scores = tf.multiply(atten_scores, tf.cast(head_scale, i_dtype))

#TODO(Itex), handle when atten_mask is(B, N, F, T) or other shapes
#atten_mask : (B, 1, F, T)
        if atten_mask is not None:
            atten_scores += atten_mask

#Normalize the attention scores to probabilities.
# `atten_probs` =[B, N, F, T]
        atten_probs = tf.nn.softmax(atten_scores, -1)

        if dropout_p != 0.0:
            atten_probs = _stateless_dropout(atten_probs, dropout_p, seed)

# `atten_output` =[B, N, F, H]
        atten_output = tf.matmul(atten_probs, value)

# `output` =[B, F, N, H]
        output  = tf.transpose(a=atten_output, perm=[0, 2, 1, 3])  
        return output
    
    def fast_sdp():
        batch_size = query.shape[0] 
        num_heads = query.shape[1]
        from_seq_len = query.shape[2]
        head_size = query.shape[3]
        to_seq_len = key.shape[2]  
        
        i_dtype = query.dtype
        use_dropout = (dropout_p != 0.0)
        use_mask = (atten_mask is not None)
        if use_dropout:
            uniform_sampler = functools.partial(stateless_random_ops.stateless_random_uniform, seed=seed)            
            random_tensor = uniform_sampler(shape=[batch_size, num_heads, from_seq_len, to_seq_len], dtype=i_dtype)
            dropout_mask = math_ops.greater_equal(random_tensor, dropout_p)
        else:
            dropout_mask = 0
        output, atten, atten_dp = load_ops_library.scaled_dot_product_attention(
            query=query, 
            key=key, 
            value=value, 
            atten_mask=atten_mask, 
            dropout_mask=dropout_mask,
            dropout_prob=dropout_p,
            use_mask=use_mask,
            use_dropout=use_dropout)
        return output
       
    if use_fast_attention and can_use_fast_sdp:
        output = fast_sdp()
    else:
        output = sdp()      
    return output