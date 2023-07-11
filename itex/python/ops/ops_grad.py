# Copyright (c) 2021-2022 Intel Corporation
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
from tensorflow.python.framework import ops
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops

@ops.RegisterGradient("Gelu")
def _gelu_grad(op, grad):
  return load_ops_library.gelu_grad(
      grad, op.inputs[0], op.get_attr("approximate")
  )

@ops.RegisterGradient("ITEXGelu")
def _itex_gelu_grad(op, grad):
  return load_ops_library.itex_gelu_grad(
      grad, op.inputs[0], op.get_attr("approximate")
  )

@ops.RegisterGradient("LayerNorm")
def _layer_norm_grad(op, *grad):
  """A dummy docstring."""
  x = op.inputs[0]
  grad_y = grad[0]
  scale = op.inputs[1]
  epsilon = op.get_attr("epsilon")
  is_training = op.get_attr("is_training")
  data_format = op.get_attr("data_format")
  grad_fun = load_ops_library.layer_norm_grad
  reserve_space_1 = op.outputs[1]
  reserve_space_2 = op.outputs[2]
  dx, dscale, doffset, _, _ = grad_fun(
      y_backprop=grad_y, x=x, scale=scale, reserve_space_1=reserve_space_1,
      reserve_space_2=reserve_space_2, epsilon=epsilon, is_training=is_training,
      data_format=data_format)
  return dx, dscale, doffset

@ops.RegisterGradient("ITEXLayerNorm")
def _itex_layer_norm_grad(op, *grad):
  """A dummy docstring."""
  x = op.inputs[0]
  grad_y = grad[0]
  scale = op.inputs[1]
  epsilon = op.get_attr("epsilon")
  is_training = op.get_attr("is_training")
  data_format = op.get_attr("data_format")
  grad_fun = load_ops_library.itex_layer_norm_grad
  reserve_space_1 = op.outputs[1]
  reserve_space_2 = op.outputs[2]
  dx, dscale, doffset, _, _ = grad_fun(
      y_backprop=grad_y, x=x, scale=scale, reserve_space_1=reserve_space_1,
      reserve_space_2=reserve_space_2, epsilon=epsilon, is_training=is_training,
      data_format=data_format)
  return dx, dscale, doffset

@ops.RegisterGradient("ItexRnn")
def _itex_rnn_grad(op, *grad):
  if not op.get_attr("is_training"):
    raise ValueError("To use RNN in gradients, is_training must be True.")
  return load_ops_library.itex_rnn_grad(
      input=op.inputs[0],
      input_h=op.inputs[1],
      input_c=op.inputs[2],
      params=op.inputs[3],
      dropout_mask=op.inputs[4],
      recurrent_dropout_mask=op.inputs[5],
      sequence_lengths=op.inputs[6],
      output=op.outputs[0],
      output_h=op.outputs[1],
      output_c=op.outputs[2],
      workspace=op.outputs[3],
      output_backprop=grad[0],
      output_h_backprop=grad[1],
      output_c_backprop=grad[2],
      rnn_mode=op.get_attr("rnn_mode"),
      dropout=op.get_attr("dropout"),
      recurrent_dropout=op.get_attr("recurrent_dropout"),
      # seed=op.get_attr("seed"),
      # seed2=op.get_attr("seed2"),
      num_proj=op.get_attr("num_proj"),
      var_seq_length=op.get_attr("var_seq_length")) + (None, None, None,)

@ops.RegisterGradient("ScaledDotProductAttention")
def _scaled_dot_product_attention_grad(op, *grad):
  dq, dk, dv = load_ops_library.scaled_dot_product_attention_grad(
      query=op.inputs[0],
      key=op.inputs[1],
      value=op.inputs[2],
      dropout_mask=op.inputs[4],
      atten=op.outputs[1],
      atten_dp=op.outputs[2],
      output_backprop=grad[0],
      dropout_prob=op.get_attr("dropout_prob"))
  return (dq, dk, dv, None, None)

      
@ops.RegisterGradient("FusedDenseBiasAddGelu")
def _itex_fused_dense_bias_add_gelu_grad(op, *grad):
  feature = op.inputs[0]
  weights = op.inputs[1]
  workspace = op.outputs[1]
  output_backprop = grad[0]
  dgelu = math_ops.mul(workspace, output_backprop)
  # TODO(zw): fuse dbias and dweights ? replace reduce_sum with biasAddgrad
  # dbias = math_ops.reduce_sum(dgelu, axis=0)  
  dbias = gen_nn_ops.bias_add_grad(dgelu)
  dweights = math_ops.matmul(feature, dgelu, transpose_a=True)
  dfeature = math_ops.matmul(dgelu, weights, transpose_b=True)
  return dfeature, dweights, dbias
