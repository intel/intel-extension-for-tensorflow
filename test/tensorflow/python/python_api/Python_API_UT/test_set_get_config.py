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
import intel_extension_for_tensorflow as itex
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops

tf.compat.v1.disable_eager_execution()
class SetGetConfigTest(test_util.TensorFlowTestCase):
    """test set_config and get_config itex python api"""

    @test_util.run_deprecated_v1
    def testSetGetConfig_gpu(self):
        graph_options = itex.GraphOptions()
        graph_options.auto_mixed_precision = itex.ON
        cfg = itex.ConfigProto(graph_options=graph_options)

        itex.set_config(cfg)
        self.assertProtoEquals("""
          graph_options { auto_mixed_precision: ON }
        """, itex.get_config())

    def testSetConfig(self):
        x = tf.compat.v1.placeholder(tf.float32, shape=(3, 4))
        y = tf.compat.v1.placeholder(tf.float32, shape=(4, 5))
        b = np.random.rand(5).astype(np.float32)

        x_arr = np.random.rand(3, 4)
        y_arr = np.random.rand(4, 5)
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        fused = tf.nn.bias_add(tf.matmul(x, y), b)
        fused = array_ops.identity(fused)

        ## Disable remapper
        graph_options = itex.GraphOptions(remapper=itex.OFF)
        cfg = itex.ConfigProto(graph_options=graph_options)
        itex.set_config(cfg)

        with self.session(use_gpu=True) as sess:
            sess.run(fused, feed_dict={x: x_arr, y: y_arr}, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedMatMul'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'BiasAdd'
                    break
            self.assertFalse(found_fused_op, "itex.set_config does not work!")

        ## Enable remapper
        graph_options = itex.GraphOptions(remapper=itex.ON)
        cfg = itex.ConfigProto(graph_options=graph_options)
        itex.set_config(cfg)

        with self.session(use_gpu=True) as sess:
            sess.run(fused, feed_dict={x: x_arr, y: y_arr}, options=run_options, run_metadata=metadata)
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedMatMul'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'BiasAdd'
                    break
            self.assertTrue(found_fused_op, "itex.set_config does not work!")

if __name__ == "__main__":
    test.main()
