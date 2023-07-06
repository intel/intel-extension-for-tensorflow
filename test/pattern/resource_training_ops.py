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
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.ops import variables
from tensorflow.python.ops import resource_variable_ops  
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library
import os

os.environ['ITEX_LAYOUT_OPT'] = '0'  # only can fused when is_layout_opt = OFF
tf.compat.v1.disable_eager_execution()
class resourceTraingingOpsTest(test.TestCase):
    def _toType(self, dtype):
        if dtype == np.float16:
            return dtypes.float16
        elif dtype == np.float32:
            return dtypes.float32
        elif dtype == np.float64:
            return dtypes.float64
        elif dtype == np.int32:
            return dtypes.int32
        elif dtype == np.int64:
            return dtypes.int64
        else:
            assert False, (dtype)

    """test _FusedResourceApplyAdam"""
    def test_resource_apply_adam(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")
        dtype = np.float32
        use_gpu = True  # Only GPU supports this fusion.
        
        var = np.arange(100).astype(dtype)
        m = np.arange(1, 101).astype(dtype)
        v = np.arange(101, 201).astype(dtype)
        grad = np.arange(100).astype(dtype)
        t = 2
        
        var_t = resource_variable_ops.ResourceVariable(var)
        m_t = resource_variable_ops.ResourceVariable(m)
        v_t = resource_variable_ops.ResourceVariable(v)
        beta1 = np.array(0.9, dtype=var.dtype)
        beta2 = np.array(0.999, dtype=var.dtype)
        beta1_power = beta1**t
        beta2_power = beta2**t
        
        lr = np.array(0.001, dtype=var.dtype)
        epsilon = np.array(1e-8, dtype=var.dtype)
        beta1_t = constant_op.constant(beta1, self._toType(var.dtype), [])
        beta2_t = constant_op.constant(beta2, self._toType(var.dtype), [])
        lr_t = constant_op.constant(lr, self._toType(var.dtype), [])
        epsilon_t = constant_op.constant(epsilon, self._toType(var.dtype), [])
        grad_t = tf.multiply(grad, 3)
        
        resource_apply_adam = tf.raw_ops.ResourceApplyAdam(var=var_t.handle, m=m_t.handle, v=v_t.handle, beta1_power=beta1_power,
                                                beta2_power=beta2_power, lr=lr_t, beta1=beta1_t,
                                                beta2=beta2_t, epsilon=epsilon_t, grad=grad_t)
        
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        with self.session(use_gpu=use_gpu) as sess:   
            sess.run(variables.global_variables_initializer())
            out = sess.run(resource_apply_adam, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedResourceApplyAdam'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'Mul'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")

    """ test _FusedResourceApplyMomentum """
    def test_resource_apply_momentum(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")
        dtype = np.float32
        use_gpu = True  # Only GPU supports this fusion.
        
        var = np.arange(100).astype(dtype)
        accum = np.arange(1, 101).astype(dtype)
        lr = np.array(0.001, dtype=var.dtype)
        momentum = np.array(1.5, dtype=var.dtype)
        grad = np.arange(100).astype(dtype)
        tmp = np.arange(101, 201).astype(dtype)

        
        var_t = resource_variable_ops.ResourceVariable(var)
        accum_t = resource_variable_ops.ResourceVariable(accum)
        grad = tf.multiply(grad, 2)
        grad_t = tf.add_n([grad, tmp])
     
        apply_momentum = tf.raw_ops.ResourceApplyMomentum(var=var_t.handle, accum=accum_t.handle, lr=lr, grad=grad_t, momentum=momentum)
    
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        
        with self.session(use_gpu=use_gpu) as sess:   
            sess.run(variables.global_variables_initializer())
            out = sess.run(apply_momentum, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedResourceApplyMomentum'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 2 and fused_ops[0] == b'Mul' and fused_ops[1] == b'AddN'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
        
        
    """ test _FusedResourceApplyAdamWithWeightDecay """
    def test_resource_apply_adam_with_weight_decay(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")
        dtype = np.float32
        use_gpu = True  # Only GPU supports this fusion.
        
        var = np.arange(100).astype(dtype)
        m = np.arange(1, 101).astype(dtype)
        v = np.arange(101, 201).astype(dtype)
        grad = np.arange(100).astype(dtype)

        t=2
        var_t = resource_variable_ops.ResourceVariable(var)
        m_t = resource_variable_ops.ResourceVariable(m)
        v_t = resource_variable_ops.ResourceVariable(v)
        beta1 = np.array(0.9, dtype=var.dtype)
        beta2 = np.array(0.999, dtype=var.dtype)
        beta1_power = beta1**t
        beta2_power = beta2**t
        
        lr = np.array(0.001, dtype=var.dtype)
        epsilon = np.array(1e-8, dtype=var.dtype)
        weight_decay_rate = np.array(0.02, dtype=var.dtype)
        
        grad = tf.multiply(grad, 2)
        apply_adam_with_weight_decay = load_ops_library.itex_resource_apply_adam_with_weight_decay(
                var_t.handle,
                m_t.handle,
                v_t.handle,
                beta1_power,
                beta2_power,
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay_rate,
                grad)
            
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        with self.session(use_gpu=use_gpu) as sess:   
            sess.run(variables.global_variables_initializer())
            out = sess.run(apply_adam_with_weight_decay, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedResourceApplyAdamWithWeightDecay'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'Mul'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")

if __name__ == '__main__':
    test.main()

