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


import tensorflow as tf
import intel_extension_for_tensorflow as itex
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.core.utils.protobuf import config_pb2

class SetGetConfigTest(test_util.TensorFlowTestCase):
    """test set_config and get_config itex python api"""

    @test_util.run_deprecated_v1
    def testSetGetConfig_gpu(self):
        graph_options = config_pb2.GraphOptions()
        graph_options.auto_mixed_precision = config_pb2.ON
        cfg = config_pb2.ConfigProto(graph_options=graph_options)

        itex.set_config(cfg)
        self.assertProtoEquals("""
          graph_options { auto_mixed_precision: ON }
        """, itex.get_config())

if __name__ == "__main__":
    test.main()
