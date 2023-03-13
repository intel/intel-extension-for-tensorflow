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


import numpy as np
import intel_extension_for_tensorflow as itex
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.ops import array_ops
import time


tf.compat.v1.disable_eager_execution()

class FusedMatMulTest(test_util.TensorFlowTestCase):
    """test fused matmul"""
    def testMatMulReshapeBiasAddRelu(self):
        tf.random.set_seed(0)
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()
        tf_result = [[[0.,1.0434492,1.3701142,0.24190968]], [[0.,4.2524147,1.6927314,0.32540703]]]
        itex.experimental_ops_override()
        with self.session(use_gpu=True) as sess:
            tf.compat.v1.keras.backend.set_session(sess)
            inputs = tf.keras.Input(shape=(1,3,))
            x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=array_ops.identity(x))
            itex_result = model.predict(np.array([[[1.,2.,3.]],[[4.,5.,6.]]]).astype(np.float32))
            start_time = time.time()
            ret_gpu = sess.run("Identity", feed_dict={"input_1:0": np.array([[[1.,2.,3.]],[[1.,2.,3.]]]).astype(np.float32)},options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if ('FusedMatMul' in node.op):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 1 and fused_ops[0] == b'BiasAdd'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")
            self.assertAllClose(tf_result, itex_result)

if __name__ == '__main__':
    test.main()
