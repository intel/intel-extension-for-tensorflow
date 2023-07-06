# Copyright (c) 2023 Intel Corporation
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


import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops  
from tensorflow.python.ops import variables
from tensorflow.python.ops import resources
from tensorflow.core.protobuf import rewriter_config_pb2
import numpy as np
import time
import os
import subprocess
import sys

os.environ["ITEX_LAYOUT_OPT"] = "0"

config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=True,
                                  inter_op_parallelism_threads=1)
config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.ON

def _Conv3DGrad(op_inputs, grad, data_format = "NDHWC", dilations=[1,1,1,1,1], strides=[1,1,1,1,1], padding="VALID"):
  shape_0, shape_1 = array_ops.shape_n([op_inputs[0], op_inputs[1]])
  return [
      nn_ops.conv3d_backprop_input_v2(
          shape_0,
          op_inputs[1],
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format),
      nn_ops.conv3d_backprop_filter_v2(
          op_inputs[0],
          shape_1,
          grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format)
  ]

def _BiasAddGrad(received_grad, data_format = "NDHWC"):
  return (received_grad,
          gen_nn_ops.bias_add_grad(
              out_backprop=received_grad, data_format=data_format))

def _ReluGrad(op_outputs, grad):
  return gen_nn_ops.relu_grad(grad, op_outputs[0])


tf.compat.v1.disable_eager_execution()
class PadFusedConv3DTest(test_util.TensorFlowTestCase):

    def _basePadWithConv3D(self, need_grad = True, need_check_accuracy = True, add_control_in=False):
        tf.compat.v1.disable_eager_execution()
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()

        x_arr = np.random.rand(1, 5, 8, 7, 1).astype(np.float32)
        x = tf.compat.v1.placeholder(tf.float32, shape=(1, 5, 8, 7, 1))
        if need_grad:
            loss = tf.compat.v1.placeholder(tf.float32, shape=(1, 7, 9, 7, 1))
            loss_arr = np.random.rand(1, 7, 9, 7, 1).astype(np.float32)

        w_arr = np.random.rand(1, 2, 3, 1, 1).astype(np.float32)
        w = resource_variable_ops.ResourceVariable(w_arr)

        b_arr = np.random.rand(1).astype(np.float32)
        b = resource_variable_ops.ResourceVariable(b_arr)

        pad_value = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
        p = constant_op.constant(pad_value, dtype=dtypes.int32)

        if add_control_in:
            with tf.control_dependencies([w,]):
                x_pad = array_ops.pad(x, p)
        else:
            x_pad = array_ops.pad(x, p)

        conv3d = nn_ops.Conv3D(input=x_pad, filter=w, strides=[1, 1, 1, 1, 1], padding='VALID', data_format='NDHWC')
        conv_bias = tf.nn.bias_add(conv3d, b)
        relu = nn_ops.relu(conv_bias)
        fused = array_ops.identity(relu)

        if need_grad:
            grad_relu = _ReluGrad([relu,], loss)
            (grad_biasadd_input, grad_biasadd_bias) = _BiasAddGrad(grad_relu, data_format = "NHWC")
            (grad_conv_input, grad_conv_filter) = _Conv3DGrad([x_pad, w], grad_biasadd_input, data_format = "NDHWC", dilations=[1,1,1,1,1], strides=[1,1,1,1,1], padding="VALID")

            grad_biasadd_bias = array_ops.identity(grad_biasadd_bias)
            grad_conv_filter = array_ops.identity(grad_conv_filter)
        
        # fused pattern output value from gpu side
        os.environ["ITEX_ENABLE_REMAPPER"] = "1" 
        with test_util.device(use_gpu=True), tf.compat.v1.Session(config=config) as sess:
            resources.initialize_resources([b,w]).run()

            if need_grad:
                ret_gpu = sess.run((fused, grad_biasadd_bias, grad_conv_filter), feed_dict={x: x_arr, loss: loss_arr}, options=run_options, run_metadata=metadata)
            else:
                ret_gpu = sess.run((fused, ), feed_dict={x: x_arr, }, options=run_options, run_metadata=metadata)

            graph = metadata.partition_graphs[0]
            found_fused_op = False
            found_fused_bwd_op = False
            for node in graph.node:
                if "PadWithFusedConv3D" in node.op:
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 2 and fused_ops[0] == b'BiasAdd' and fused_ops[1] == b'Relu'
                if "PadWithConv3DBackpropFilter" in node.op:
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_bwd_op = len(fused_ops) == 1 and fused_ops[0] == b'BiasAddGrad' 
                
            if add_control_in:
                if need_grad:
                    self.assertTrue((not found_fused_op) or (not found_fused_bwd_op), "this pattern has fusion issue!!")
                else:
                    self.assertTrue((not found_fused_op), "this pattern has fusion issue!!")
            else:
                if need_grad:
                    self.assertTrue(found_fused_op and found_fused_bwd_op, "this pattern has fusion issue!!")
                else:
                    self.assertTrue(found_fused_op, "this pattern has fusion issue!!")

        if need_check_accuracy:
            # reference value which is no fusion
            os.environ["ITEX_ENABLE_REMAPPER"] = "0" 
            with self.session(use_gpu=False) as sess:
                resources.initialize_resources([b,w]).run()
                if need_grad:
                    ret_ref = sess.run((fused, grad_biasadd_bias, grad_conv_filter), feed_dict={x: x_arr, loss: loss_arr}, options=run_options, run_metadata=metadata)
                else:
                    ret_ref = sess.run((fused, ), feed_dict={x: x_arr, }, options=run_options, run_metadata=metadata)
            self.assertAllClose(ret_ref, ret_gpu)

    def testPadWithFusedConv3D(self, need_grad = True, need_check_accuracy = True):
        self._basePadWithConv3D(need_grad = False, need_check_accuracy = True)

    def testPadWithConv3DBackpropFilterWithBias(self, need_grad = True, need_check_accuracy = True):
        self._basePadWithConv3D(need_grad = True, need_check_accuracy = True)

    def testNotPadWithFusedConv3D(self):
        self._basePadWithConv3D(need_grad = False, need_check_accuracy = False, add_control_in=True)

    def testNotPadWithConv3DBackpropFilterWithBias(self):
        self._basePadWithConv3D(need_grad = True, need_check_accuracy = False, add_control_in=True)

if __name__ == '__main__':
    test.main()
