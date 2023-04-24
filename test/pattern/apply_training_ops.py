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
from tensorflow.python.ops import array_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.python.ops import variables
from intel_extension_for_tensorflow.python.ops.load_ops_library import load_ops_library

tf.compat.v1.disable_eager_execution()
@test_util.run_all_in_native_and_block_format
class trainingOpsTest(test.TestCase):
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
      
      
    """test _FusedApplyAdam"""
    def test_apply_adam(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")
        dtype = np.float32
        use_gpu = True  # Only GPU supports this fusion.
        
        var = np.arange(100).astype(dtype)
        m = np.arange(1, 101).astype(dtype)
        v = np.arange(101, 201).astype(dtype)
        grad = np.arange(100).astype(dtype)
        t = 1
        
        var_t = variables.RefVariable(var)
        m_t = variables.RefVariable(m)
        v_t = variables.RefVariable(v)
        beta1 = np.array(0.9, dtype=var.dtype)
        beta2 = np.array(0.999, dtype=var.dtype)
        beta1_power = beta1**t
        beta2_power = beta2**t
        
        lr = np.array(0.001, dtype=var.dtype)
        epsilon = np.array(1e-8, dtype=var.dtype)
        grad = tf.multiply(grad, 3)
        
        apply_adam = tf.raw_ops.ApplyAdam(var=var_t, m=m_t, v=v_t, beta1_power=beta1_power,
                                                beta2_power=beta2_power, lr=lr, beta1=beta1,
                                                beta2=beta2, epsilon=epsilon, grad=grad)
        output = array_ops.identity(apply_adam)
    
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        with self.session(use_gpu=use_gpu) as sess:   
            sess.run(variables.global_variables_initializer())
            out = sess.run(output, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedApplyAdam'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'Mul'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")

    
    """ test _FusedApplyMomentum """
    def test_apply_momentum(self):
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

        
        var_t = variables.RefVariable(var)
        accum_t = variables.RefVariable(accum)
        grad = tf.multiply(grad, 2)
        grad_t = tf.add_n([grad, tmp])
     
        apply_momentum = tf.raw_ops.ApplyMomentum(var=var_t, accum=accum_t, lr=lr, grad=grad_t, momentum=momentum)
        output = array_ops.identity(apply_momentum)
    
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        
        with self.session(use_gpu=use_gpu) as sess:   
            sess.run(variables.global_variables_initializer())
            out = sess.run(output, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedApplyMomentum'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 2 and fused_ops[0] == b'Mul' and fused_ops[1] == b'AddN'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
        
    """ test _FusedApplyAdamWithWeightDecay """
    def test_apply_adam_with_weight_decay(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")
        dtype = np.float32
        use_gpu = True  # Only GPU supports this fusion.
        
        var = np.arange(100).astype(dtype)
        m = np.arange(1, 101).astype(dtype)
        v = np.arange(101, 201).astype(dtype)
        grad = np.arange(100).astype(dtype)

        t=1
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
        
        grad = tf.multiply(grad, 2)
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
        
        output = array_ops.identity(apply_adam_with_weight_decay)
    
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        with self.session(use_gpu=use_gpu) as sess:   
            sess.run(variables.global_variables_initializer())
            out = sess.run(output, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedApplyAdamWithWeightDecay'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'Mul'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
        
        

if __name__ == '__main__':
    test.main()

