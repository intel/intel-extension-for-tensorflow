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

from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.core.protobuf import config_pb2
import time
import os



tf.compat.v1.disable_eager_execution()
@test_util.run_all_in_native_and_block_format
class FusedAddNTest(test.TestCase):
    """test _ITEXFusedAddN"""
    def test_addn_l2loss(self):
        if not test.is_gpu_available():
            self.skipTest("No GPU available")

        x = np.array([1, 2, 3, 4], dtype=np.float32)
        y = np.array([2, 1, 5, 6], dtype=np.float32)

        l2loss1 = tf.raw_ops.L2Loss(t=x)
        l2loss2 = tf.raw_ops.L2Loss(t=y)
        addN = math_ops.add_n([l2loss1, l2loss2])
        output = array_ops.identity(addN)
        
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        metadata = config_pb2.RunMetadata()

        # CPU does not support the fusion of l2loss + AddN
        with self.session(use_gpu=True) as sess: 
            os.environ['ITEX_ENABLE_REMAPPER'] = '1'           
            start_time = time.time()
            ret_gpu = sess.run(output, options=run_options, run_metadata=metadata)
            duration = time.time() - start_time
            print("end to end duration is : {}".format(duration))
            # Graph should contain fused op.
            graph = metadata.partition_graphs[0]
            found_fused_op = False
            for node in graph.node:
                if node.op in ('_ITEXFusedAddN'):
                    fused_ops = node.attr['fused_ops'].list.s
                    found_fused_op = len(fused_ops) == 2 and fused_ops[0] == b'AddN' and fused_ops[1] == b'l2loss'
                    break
            self.assertTrue(found_fused_op, "this pattern has fusion issue!!")            

        
        
if __name__ == '__main__':
    test.main()
