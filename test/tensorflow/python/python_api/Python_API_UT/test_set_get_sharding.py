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
import intel_extension_for_tensorflow as itex
from intel_extension_for_tensorflow.python.test_func import test_util
from intel_extension_for_tensorflow.python.test_func import test

class SetGetSharding(test_util.TensorFlowTestCase):
    """test sharding itex python api"""

    @test_util.run_deprecated_v1
    def testSetGetSharding(self):
        config = itex.ShardingConfig()
        config.auto_mode = False
        device_gpu = config.devices.add()
        device_gpu.device_type = "gpu"
        device_gpu.device_num = 2
        device_gpu.batch_size = 128
        device_gpu.stage_num = 16
        device_cpu = config.devices.add()
        device_cpu.device_type = "cpu"
        device_cpu.device_num = 4
        device_cpu.batch_size = 64
        device_cpu.stage_num = 32
        graph_opts = itex.GraphOptions(sharding=itex.ON, sharding_config=config)
        cfg = itex.ConfigProto(graph_options=graph_opts)
        itex.set_config(cfg)
        new_config = itex.get_config().graph_options
        self.assertIs(new_config.sharding, itex.ON)
        self.assertLen(new_config.sharding_config.devices, 2)
        gpu = new_config.sharding_config.devices[0]
        cpu = new_config.sharding_config.devices[1]
        self.assertIn(gpu.device_type, "gpu")
        self.assertIs(gpu.device_num, 2)
        self.assertIs(gpu.batch_size, 128)
        self.assertIs(gpu.stage_num, 16)
        self.assertIn(cpu.device_type, "cpu")
        self.assertIs(cpu.device_num, 4)
        self.assertIs(cpu.batch_size, 64)
        self.assertIs(cpu.stage_num, 32)

if __name__ == "__main__":
    test.main()
