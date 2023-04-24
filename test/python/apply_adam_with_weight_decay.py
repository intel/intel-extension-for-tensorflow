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
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.ops import variables
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library


def apply_adam_with_weight_decay_numpy(var, m, v, beta1_power, beta2_power, lr, beta_1, beta_2, epsilon, wd, grad):
    """Numpy update"""
    lr_t = lr * np.sqrt(1 - beta2_power) / (1 - beta1_power)
    m = beta_1 * m + (1 - beta_1) * grad
    v = beta_2 * v + (1 - beta_2) * grad ** 2
    var = var * (1 - wd * lr) - lr_t * m / (np.sqrt(v) + epsilon)
    return var

tf.compat.v1.disable_eager_execution()
@test_util.run_all_in_native_and_block_format
class ApplyAdamWithWeightDecay(test.TestCase):
    def test_apply_adam_with_weight_decay(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")
        dtype = dtypes.float32
        
        var = np.arange(100, dtype=dtype.as_numpy_dtype)
        m = np.arange(1, 101, dtype=dtype.as_numpy_dtype)
        v = np.arange(101, 201, dtype=dtype.as_numpy_dtype)
        grad = np.arange(100, dtype=dtype.as_numpy_dtype)

        t=2
        var_t = variables.RefVariable(var)
        m_t = variables.RefVariable(m)
        v_t = variables.RefVariable(v)
        beta1 = np.array(0.9, dtype=var.dtype)
        beta2 = np.array(0.999, dtype=var.dtype)
        beta1_power = beta1**t
        beta2_power = beta2**t
        
        lr = np.array(0.001, dtype=var.dtype)
        epsilon = np.array(1e-8, dtype=var.dtype)
        weight_decay_rate = np.array(0.02, dtype=var.dtype)

        
        apply_adam_with_weight_decay = load_ops_library.itex_apply_adam_with_weight_decay(
                var_t,
                m_t,
                v_t,
                beta1_power,
                beta2_power,
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay_rate,
                grad)
        numpy_out = apply_adam_with_weight_decay_numpy(var,
                m,
                v,
                beta1_power,
                beta2_power,
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay_rate,
                grad)

        with self.session(use_gpu=True) as sess:   
            sess.run(variables.global_variables_initializer())
            out = sess.run(apply_adam_with_weight_decay)

        self.assertAllCloseAccordingToType(out, numpy_out)
        
if __name__ == '__main__':
    test.main()