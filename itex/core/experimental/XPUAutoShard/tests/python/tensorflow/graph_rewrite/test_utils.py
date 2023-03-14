#  Copyright (c) 2023 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This file test whether the function is ready in utils.py

import os
import unittest
from utils.rewrite_graph_helper import rewrite_graph

cwd = os.getcwd()

class TestRewriteFunctionCall(unittest.TestCase):

  def test_python_frontend(self):
    ori_graph_input_dir = "./pbtxt/fixed_dim/nhwc/resnet"
    ori_graph_input_name = "resnet_block_ori.pbtxt"
    rewritten_graph_output_dir = cwd
    rewritten_graph_output_name = 'resnet_block_1_1_py.pbtxt'

    rewrite_graph(ori_graph_input_dir,
                  ori_graph_input_name,
                  rewritten_graph_output_dir,
                  rewritten_graph_output_name,
                  rewrite_method="PYTHON_FRONTEND",
                  device_ids=[1, 2],
                  device_names=['GPU:0', 'GPU:1'],
                  device_scores=[1.0, 1.0],
                  device_stages=[1, 1],
                  total_bs=64)

  def test_cpp_auto_shard(self):
    ori_graph_input_dir = "./pbtxt/fixed_dim/nhwc/resnet"
    ori_graph_input_name = "resnet_block_ori.pbtxt"
    rewritten_graph_output_dir = cwd
    rewritten_graph_output_name = 'resnet_block_1_1_cpp'

    rewrite_graph(ori_graph_input_dir,
                  ori_graph_input_name,
                  rewritten_graph_output_dir,
                  rewritten_graph_output_name,
                  rewrite_method="CPP_AUTO_SHARD",
                  device_ids=[1, 2],
                  device_names=['GPU:0', 'GPU:1'],
                  device_scores=[1.0, 1.0],
                  device_stages=[1, 1],
                  total_bs=64,
                  useNCCLBackend=True
                  )
    


if __name__ == '__main__':
  unittest.main()
