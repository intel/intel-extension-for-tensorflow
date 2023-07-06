# Copyright (c) 2021 Intel Corporation
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
"""Adam for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

class AdamWithWeightDecayOptimizer(optimizer.Optimizer): # pylint: disable=missing-class-docstring
  def __init__(self, # pylint: disable=dangerous-default-value
               weight_decay_rate=0.001,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-8,
               use_locking=False,
               name="Adam",
               exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]):
    r"""Construct a new Adam optimizer with weight decay
    """
    super(AdamWithWeightDecayOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.weight_decay_rate = weight_decay_rate

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta_1_t = None
    self._beta_2_t = None
    self._epsilon_t = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta_1_power", graph=graph),
              self._get_non_slot_variable("beta_2_power", graph=graph))

  def _create_slots(self, var_list):
    """A dummy docstring."""
    # Create the beta_1 and beta_2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta_1,
        name="beta_1_power",
        colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta_2,
        name="beta_2_power",
        colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta_1 = self._call_if_callable(self._beta_1)
    beta_2 = self._call_if_callable(self._beta_2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta_1_t = ops.convert_to_tensor(beta_1, name="beta_1")
    self._beta_2_t = ops.convert_to_tensor(beta_2, name="beta_2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _apply_dense(self, grad, var):
    """A dummy docstring."""
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta_1_power, beta_2_power = self._get_beta_accumulators()
    param_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(param_name): # pylint: disable=no-else-return
      return load_ops_library.itex_apply_adam_with_weight_decay(
          var,
          m,
          v,
          math_ops.cast(beta_1_power, var.dtype.base_dtype),
          math_ops.cast(beta_2_power, var.dtype.base_dtype),
          math_ops.cast(self._lr_t, var.dtype.base_dtype),
          math_ops.cast(self._beta_1_t, var.dtype.base_dtype),
          math_ops.cast(self._beta_2_t, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
          math_ops.cast(self.weight_decay_rate, var.dtype.base_dtype),
          grad,
          self._use_locking).op
    else:
      return training_ops.apply_adam(
          var,
          m,
          v,
          math_ops.cast(beta_1_power, var.dtype.base_dtype),
          math_ops.cast(beta_2_power, var.dtype.base_dtype),
          math_ops.cast(self._lr_t, var.dtype.base_dtype),
          math_ops.cast(self._beta_1_t, var.dtype.base_dtype),
          math_ops.cast(self._beta_2_t, var.dtype.base_dtype),
          math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var): # pylint: disable=arguments-differ
    """A dummy docstring."""
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta_1_power, beta_2_power = self._get_beta_accumulators()
    param_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(param_name): # pylint: disable=no-else-return
      return load_ops_library.itex_resource_apply_adam_with_weight_decay(
          var.handle,
          m.handle,
          v.handle,
          math_ops.cast(beta_1_power, grad.dtype.base_dtype),
          math_ops.cast(beta_2_power, grad.dtype.base_dtype),
          math_ops.cast(self._lr_t, grad.dtype.base_dtype),
          math_ops.cast(self._beta_1_t, grad.dtype.base_dtype),
          math_ops.cast(self._beta_2_t, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
          math_ops.cast(self.weight_decay_rate, var.dtype.base_dtype),
          grad,
          self._use_locking)
    else:
      return training_ops.resource_apply_adam(
          var.handle,
          m.handle,
          v.handle,
          math_ops.cast(beta_1_power, grad.dtype.base_dtype),
          math_ops.cast(beta_2_power, grad.dtype.base_dtype),
          math_ops.cast(self._lr_t, grad.dtype.base_dtype),
          math_ops.cast(self._beta_1_t, grad.dtype.base_dtype),
          math_ops.cast(self._beta_2_t, grad.dtype.base_dtype),
          math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
          grad,
          use_locking=self._use_locking)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add): # pylint: disable=arguments-differ,missing-function-docstring
    beta_1_power, beta_2_power = self._get_beta_accumulators()
    beta_1_power = math_ops.cast(beta_1_power, var.dtype.base_dtype)
    beta_2_power = math_ops.cast(beta_2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta_1_t = math_ops.cast(self._beta_1_t, var.dtype.base_dtype)
    beta_2_t = math_ops.cast(self._beta_2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
    # m_t = beta_1 * m + (1 - beta_1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta_1_t)
    m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta_2 * v + (1 - beta_2) * (g_t * g_t)
    v = self.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
    v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(
        var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values,
        var,
        grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x,
            i,
            v,
            use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices): # pylint: disable=arguments-differ
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta_1_power, beta_2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta_1_power):
        update_beta_1 = beta_1_power.assign(
            beta_1_power * self._beta_1_t, use_locking=self._use_locking)
        update_beta_2 = beta_2_power.assign(
            beta_2_power * self._beta_2_t, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta_1, update_beta_2], name=name_scope)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
    if not grads_and_vars:
      raise ValueError("No variables provided.")
    converted_grads_and_vars = []
    for g, v in grads_and_vars:
      if g is not None:
        g = ops.convert_to_tensor(g)
        converted_grads_and_vars.append((g, v))
    return super(AdamWithWeightDecayOptimizer, self).apply_gradients(
        converted_grads_and_vars, global_step, name)
