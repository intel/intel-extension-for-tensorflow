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

import os
import re
import warnings
import tensorflow as tf
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
if os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"):
  from tf_keras.src.optimizers import optimizer as kerasoptimizer
  from tf_keras.src.optimizers import utils as optimizer_utils
else:
  from keras.src.optimizers import optimizer as kerasoptimizer
  from keras.src import utils as optimizer_utils

class AdamWithWeightDecayLegacyOptimizer(optimizer.Optimizer): # pylint: disable=missing-class-docstring
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
    super(AdamWithWeightDecayLegacyOptimizer, self).__init__(use_locking, name)
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
          v,
          grad,
          use_locking=self._use_locking,
          use_amsgrad=False).op
    else:
      return gen_training_ops.apply_adam(
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
          v.handle,
          grad,
          use_locking=self._use_locking,
          use_amsgrad=False)
    else:
      return gen_training_ops.resource_apply_adam(
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
    return super(AdamWithWeightDecayLegacyOptimizer, self).apply_gradients(
        converted_grads_and_vars, global_step, name)

class AdamOptimizer(kerasoptimizer.Optimizer):
    r"""Optimizer that implements the AdamW algorithm.

    AdamW optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments with an added
    method to decay weights per the techniques discussed in the paper,
    'Decoupled Weight Decay Regularization' by
    [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the underying Adam method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.001.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to 0.9.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates.
            Defaults to 0.999.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to 1e-7.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond".
            Defaults to `False`.
        name: Optional name for the operations created when applying
            gradients. Defaults to "AdamW".
        {{base_optimizer_keyword_args}}

    Reference:
      - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980) for `adam`
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        adaptive_epsilon=False,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Adam",
        **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.adaptive_epsilon = adaptive_epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        """Initialize optimizer variables.

        AdamW optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build AdamW variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        if self.adaptive_epsilon:
            epsilon_hat = self.epsilon * tf.sqrt(1 - beta_2_power)
        else:
            epsilon_hat = self.epsilon

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + epsilon_hat))
        else:
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
            else:
                v_hat = v

            return load_ops_library.itex_resource_apply_adam_with_weight_decay(
                variable.handle,
                m.handle,
                v.handle,
                beta_1_power,
                beta_2_power,
                lr,
                math_ops.cast(self.beta_1, variable.dtype),
                math_ops.cast(self.beta_2, variable.dtype),
                math_ops.cast(self.epsilon, variable.dtype),
                math_ops.cast(0.0, variable.dtype),
                v_hat.handle,
                gradient,
                use_locking=False,
                use_amsgrad=self.amsgrad)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config

class AdamWithWeightDecayOptimizer(kerasoptimizer.Optimizer):
    r"""Optimizer that implements the AdamW algorithm.

    AdamW optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments with an added
    method to decay weights per the techniques discussed in the paper,
    'Decoupled Weight Decay Regularization' by
    [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the underying Adam method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.001.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to 0.9.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates.
            Defaults to 0.999.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to 1e-7.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond".
            Defaults to `False`.
        name: Optional name for the operations created when applying
            gradients. Defaults to "AdamW".
        {{base_optimizer_keyword_args}}

    Reference:
      - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980) for `adam`
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="AdamW",
        **kwargs
    ):
        if "weight_decay_rate" in kwargs:
            warnings.warn(
                "weight_decay_rate has been renamed to weight_decay,",
                DeprecationWarning,
            )
            weight_decay = kwargs["weight_decay_rate"]
            del kwargs["weight_decay_rate"]

        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        AdamW optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build AdamW variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]
        
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self._use_weight_decay(variable):
                wd = tf.cast(self.weight_decay, variable.dtype)
                variable.assign_sub(variable * wd * lr)
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            if not self._use_weight_decay(variable):
                self.weight_decay = 0
            if self.amsgrad:
               v_hat = self._velocity_hats[self._index_dict[var_key]]
            else:
               v_hat = v  # just a placeholder
            
            return load_ops_library.itex_resource_apply_adam_with_weight_decay(
                variable.handle,
                m.handle,
                v.handle,
                beta_1_power,
                beta_2_power,
                lr,
                math_ops.cast(self.beta_1, variable.dtype),
                math_ops.cast(self.beta_2, variable.dtype),
                math_ops.cast(self.epsilon, variable.dtype),
                math_ops.cast(self.weight_decay, variable.dtype),
                v_hat.handle,
                gradient,
                use_locking=False,
                use_amsgrad=self.amsgrad)

    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients to variables.

        Args:
          grads_and_vars: List of `(gradient, variable)` pairs.
          name: string, defaults to None. The name of the namescope to
            use when creating variables. If None, `self.name` will be used.

        Returns:
          A `tf.Variable`, representing the current iteration.

        Raises:
          TypeError: If `grads_and_vars` is malformed.
        """
        self._compute_current_learning_rate()
        grads_and_vars = list(grads_and_vars)
        if len(grads_and_vars) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return self._iterations
        grads, trainable_variables = zip(*grads_and_vars)
        scope_name = name or self.name or "optimizer"
        with tf.name_scope(scope_name):
            with tf.init_scope():
                # Lift variable creation to init scope to avoid environment
                # issues.
                self.build(trainable_variables)
            grads_and_vars = optimizer_utils.filter_empty_gradients(
                grads_and_vars
            )
            if len(list(grads_and_vars)) == 0:
                # Check again after filtering gradients.
                return self._iterations

            grads, trainable_variables = zip(*grads_and_vars)

            grads = self._clip_gradients(grads)
            grads = self._deduplicate_sparse_grad(grads)
            # self._apply_weight_decay(trainable_variables) # when dense, calculate in adamw kernel
            grads_and_vars = list(zip(grads, trainable_variables))
            iteration = self._internal_apply_gradients(grads_and_vars)

            # Apply variable constraints after applying gradients.
            for variable in trainable_variables:
                if variable.constraint is not None:
                    variable.assign(variable.constraint(variable))
            return iteration

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config

class LAMBOptimizer(kerasoptimizer.Optimizer):
    r"""Optimizer that implements the Layer-wise Adaptive Moments (LAMB) algorithm.

    See paper [Large Batch Optimization for Deep Learning: Training BERT
    in 76 minutes](https://arxiv.org/abs/1904.00962).

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.001.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates.
            Defaults to 0.9.
        beta_2: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 2nd moment estimates.
            Defaults to 0.999.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
            Defaults to 1e-7.
        amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm
            from the paper "On the Convergence of Adam and beyond".
            Defaults to `False`.
        name: Optional name for the operations created when applying
              gradients. Defaults to "LAMB".
        {{base_optimizer_keyword_args}}

    Reference:
      - [You et al., 2019](https://arxiv.org/abs/1904.00962) for `LAMB`.
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.004,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="LAMB",
        **kwargs
    ):
        if "weight_decay_rate" in kwargs:
            warnings.warn(
                "weight_decay_rate has been renamed to weight_decay,",
                DeprecationWarning,
            )
            weight_decay = kwargs["weight_decay_rate"]
            del kwargs["weight_decay_rate"]

        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

        if self.weight_decay is None:
            raise ValueError(
                "Missing value of `weight_decay` which is required and"
                " must be a float value."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        AdamW optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
          var_list: list of model variables to build AdamW variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            m_t_hat = m / (1 - beta_1_power)
            v_t_hat = v / (1 - beta_2_power)
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v_t_hat))
                v_t_hat = v_hat
            update = m_t_hat / (tf.sqrt(v_t_hat) + self.epsilon)
            if self._use_weight_decay(variable):
                wd = tf.cast(self.weight_decay, variable.dtype)
                update += variable * wd
            ratio=1.0
            if self._use_layer_adaptation(variable):
                w_norm = tf.norm(variable, ord=2)
                g_norm = tf.norm(update, ord=2)
                ratio = tf.where(
                    tf.greater(w_norm, 0),
                    tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                    1.0,
                )
            variable.assign_sub(ratio * lr * update)
        else:
            if not self._use_weight_decay(variable):
                self.weight_decay = 0
            if self.amsgrad:
               v_hat = self._velocity_hats[self._index_dict[var_key]]
            else:
               v_hat = v  # just a placeholder
            
            return load_ops_library.itex_resource_apply_lamb(
                variable.handle,
                m.handle,
                v.handle,
                beta_1_power,
                beta_2_power,
                lr,
                math_ops.cast(self.beta_1, variable.dtype),
                math_ops.cast(self.beta_2, variable.dtype),
                math_ops.cast(self.epsilon, variable.dtype),
                math_ops.cast(self.weight_decay, variable.dtype),
                v_hat.handle,
                gradient,
                use_locking=False,
                use_amsgrad=self.amsgrad,
                use_lamb=self._use_layer_adaptation(variable))

    def exclude_from_layer_adaptation(self, var_list=None, var_names=None):
        """Exclude variables from layer adaptation.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `tf.Variable`s to exclude from layer adaptation.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from layer adaptation. For example, `var_names=['bias']`
                excludes all bias variables from layer adaptation.
        """
        if hasattr(self, "_built") and self._built:
            raise ValueError(
                "`exclude_from_layer_adaptation()` can only be configued before "
                "the optimizer is built."
            )

        if var_list:
            self._exclude_from_layer_adaptation = [
                self._var_key(variable) for variable in var_list
            ]
        else:
            self._exclude_from_layer_adaptation = []
        self._exclude_from_layer_adaptation_names = var_names or []

    def _use_layer_adaptation(self, variable):
        exclude_from_layer_adaptation = getattr(
            self, "_exclude_from_layer_adaptation", []
        )
        exclude_from_layer_adaptation_names = getattr(
            self, "_exclude_from_layer_adaptation_names", []
        )
        variable_id = self._var_key(variable)
        for exclude_id in exclude_from_layer_adaptation:
            if variable_id == exclude_id:
                return False
        for name in exclude_from_layer_adaptation_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients to variables.

        Args:
          grads_and_vars: List of `(gradient, variable)` pairs.
          name: string, defaults to None. The name of the namescope to
            use when creating variables. If None, `self.name` will be used.

        Returns:
          A `tf.Variable`, representing the current iteration.

        Raises:
          TypeError: If `grads_and_vars` is malformed.
        """
        self._compute_current_learning_rate()
        grads_and_vars = list(grads_and_vars)
        if len(grads_and_vars) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return self._iterations
        grads, trainable_variables = zip(*grads_and_vars)
        scope_name = name or self.name or "optimizer"
        with tf.name_scope(scope_name):
            with tf.init_scope():
                # Lift variable creation to init scope to avoid environment
                # issues.
                self.build(trainable_variables)
            grads_and_vars = optimizer_utils.filter_empty_gradients(
                grads_and_vars
            )
            if len(list(grads_and_vars)) == 0:
                # Check again after filtering gradients.
                return self._iterations

            grads, trainable_variables = zip(*grads_and_vars)

            grads = self._clip_gradients(grads)
            grads = self._deduplicate_sparse_grad(grads)
            # self._apply_weight_decay(trainable_variables) # when dense, calculate in adamw kernel
            grads_and_vars = list(zip(grads, trainable_variables))
            iteration = self._internal_apply_gradients(grads_and_vars)

            # Apply variable constraints after applying gradients.
            for variable in trainable_variables:
                if variable.constraint is not None:
                    variable.assign(variable.constraint(variable))
            return iteration

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config

# Copyright (c) 2021 Intel Corporation
#
