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

# This script provides util functions for test only.
import os
import sys
import subprocess
from pathlib import Path
# append python graph_rewrite into path
sys.path.append(
    os.path.join(os.path.dirname(__file__),
                 '../../../../../python/xpuautoshard/tensorflow/graph_rewrite'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rewrite_graph_main import python_graph_rewrite


class _Device:
  """A description of device.

  [Note:] This class should not be used outside this utils.py. Due to
  that there are too many places have the same Device info. (CPP has
  DeviceInfo class, Python frontend have DeviceHint) This is a uniform
  wrapper to integrate both PYTHON_FRONTEND and CPP_AUTO_SHARD device
  description.
  """

  def __init__(self, device_id, device_name, score, ratio, num_stages) -> None:
    self.id_ = device_id
    self.name = device_name
    self.score = score
    self.ratio = ratio
    self.num_stages = num_stages

    if "CPU" in device_name or "cpu" in device_name:
      self.type = "CPU"
    else:
      assert ("GPU" in device_name or "gpu" in device_name)
      self.type = "GPU"


def initialize_device_info_list(device_ids: list[int], device_names: list[str],
                                device_scores: list[float],
                                device_stages: list[int]) -> list[_Device]:
  """Initialize a list of device info from given arguments."""
  total_score = sum(device_scores)
  device_info_list = []
  for i, device_id in enumerate(device_ids):
    device_info_list.append(
        _Device(device_id,
                device_names[i],
                score=device_scores[i],
                ratio=device_scores[i] / total_score,
                num_stages=device_stages[i]))
  return device_info_list


def rewritten_graph_name_helper(test_method_name: str, device_stages: list[int],
                                rewrite_method: str):
  """Return the output name from given config.
  
  Return string like 'resnet_block_1_1_cpp'.
  """
  device_stage_str = '_'.join(str(i) for i in device_stages)
  rewrite_method_suffix = "py" if rewrite_method == "PYTHON_FRONTEND" else "cpp"
  return "_".join([test_method_name, device_stage_str, rewrite_method_suffix])


def rewrite_graph(
    ori_graph_input_dir: str,
    ori_graph_input_name: str,
    rewritten_graph_output_dir: str,
    rewritten_graph_output_name: str,
    rewrite_method: str,
    device_ids: list[int],
    device_names: list[str],
    device_scores: list[float],
    device_stages: list[int],
    total_bs: int,

    # Python frontend Related args
    py_allreduce_on: str = "GPU:0",

    # CPP_AUTO_SHARD related args
    cpu_host: bool = False,
    host_score: float = 1.0,
    host_name: str = "",
    strategy: str = "HEURISTIC",
    useNCCLBackend: bool = False):
  '''
  The wrapper function to rewrite given single device graph to multi-device graph.
  Arguments:

  **Common arguments** for both PYTHON_FRONTEND & CPP_AUTO_SHARD:

  - `ori_graph_input_dir`: original single device graph dir.
  - `ori_graph_input_name`: original single device graph name, "a.pbtxt" for example.
  - `rewritten_graph_output_dir`: output rewritten multi device graph dir.
  - `rewritten_graph_output_name`: output rewritten multi device graph name. Due to the mismatch convention of
                TF CPP function `DumpGraphDefToFile` and Python call `tf.io.write_graph`, the Python call will
                have no suffix "pbtxt". In order to not modify the suffix, thus here the output name should 
                be something like "filename.pbtxt". Instead of only "filename".
  - `rewrite_method`: Whether use `PYTHON_FRONTEND` or `CPP_AUTO_SHARD`.
  - `device_ids`: A list or ints, of device ids. E.g., [1,2,3].
  - `device_names`: A list of device name strings. E.g., ['GPU:0', 'GPU:1'].
  - `device_scores`: A list containing device compute capacity. e.g.,[0.8,0.8]. Here does not require the score adds
                up to 1. It could be any positive numbers. The final split batch size will be set according to the 
                relative ratio.
  - `device_stages`: A list of how many stages should that device will have. E.g.,[1,2,4].
  - `total_bs`: A integer of total batch size across all the devices.

  **PYTHON_FRONTEND Related Args**
  - `py_allreduce_on` : A string, indicating which one is the reduction device. Default is GPU:0.

  **CPP_AUTO_SHARD Related Args**
  - `cpu_host` : A bool, whether use CPU as host. Default is False.
  - `host_score`: A float, indicating compute capacity of host. Default is 1.
  - `host_name` : A str of host name. Default is empty "".
  - `strategy` : A str indicating which config strategy to use. Choose from `(CPU_HOST|HEURISTIC|LEARNED)`. Default is `HEURISTIC`.
  - `useNCCLBackend` : A bool, whether to use `NCCLAllReduce` or `CCLAllReduce`. Default is false.
                
  '''
  # initialize device info list and parse for corresponding arguments
  device_info_list = initialize_device_info_list(device_ids, device_names,
                                                 device_scores, device_stages)

  # Currently, there is mismatch API for passing arguments for Python frontend and CPP auto shard.
  # Thus, there needs a parser to pase device info.
  if rewrite_method == "PYTHON_FRONTEND":
    # use python frontend under python/xpuautoshard/tensorflow/graph_rewrite/rewrite_graph_main.py
    # Python frontend does not need to set total batch size.

    # parse device info
    # [Note:] This parsing process only applies to Python frontend
    cpu_num = 0
    gpu_num = 0
    cpu_num_stage = []
    gpu_num_stage = []
    cpu_bs_per_stage = []
    gpu_bs_per_stage = []

    # TODO : Currently only set grain_size = 1 for simplicity.
    grain_size = 1
    for device in device_info_list:
      bs = int(device.ratio * total_bs / grain_size * grain_size)
      if device.type == "CPU":
        cpu_num += 1
        cpu_num_stage.append(device.num_stages)
        cpu_bs_per_stage.append(bs)
      else:
        gpu_num += 1
        gpu_num_stage.append(device.num_stages)
        gpu_bs_per_stage.append(bs)

    # call python graph rewrite
    python_graph_rewrite(ori_graph_input_dir, ori_graph_input_name,
                         rewritten_graph_output_dir,
                         rewritten_graph_output_name, cpu_num, cpu_num_stage,
                         cpu_bs_per_stage, gpu_num, gpu_num_stage,
                         gpu_bs_per_stage, py_allreduce_on)

  elif rewrite_method == "CPP_AUTO_SHARD":
    # [TODO:] Current setting, assumes that the total bs for the graph is set already.
    # For example, a target 10:10 multi-stage with each stage have bs=32, would have to rewrite
    # a single graph with total bs = (10+10)*32 = 640. If one user wants to change to 20:20, one will
    # have to regenerate the single device graph with total bs = 1280.
    input_name = os.path.join(ori_graph_input_dir, ori_graph_input_name)
    # output_name = os.path.join(rewritten_graph_output_dir,rewritten_graph_output_name)
    rel_binary_file_path = Path(
        "../../../../build/tests/cpp/tensorflow/auto_shard_test_app")
    abs_binary_path = rel_binary_file_path.absolute()

    result=subprocess.run([abs_binary_path,'-verbose=True','-i',input_name,
                    '-o', rewritten_graph_output_name,'-out_dir',rewritten_graph_output_dir,
                    '-host_score',str(host_score),
                    '-host_name',host_name,
                    '-strategy', strategy, 
                    '-cpu_host={}'.format(cpu_host), 
                    '-useNCCLBackend={}'.format(useNCCLBackend),
                    '-device_ids',','.join(map(str,device_ids)), '-device_names',','.join(map(str,device_names)),
                    '-device_scores',','.join(map(str,device_scores)),'-device_stages',','.join(map(str,device_stages))
                    ], stdout=subprocess.PIPE)
    if (result.returncode != 0):
      print("Error result is {}".format(result))

  else:
    raise ValueError(
        "Expect the rewrite method is either PYTHON_FRONTEND or CPP_AUTO_SHARD,but get : ",
        rewrite_method)


if __name__ == '__main__':
  ori_graph_input_dir = "./pbtxt/fixed_dim/nhwc/conv"
  ori_graph_input_name = "conv_training_ori.pbtxt"
  rewritten_graph_output_dir = os.getcwd()

  def _generate_python_frontend_graph():
    rewritten_graph_output_name = 'conv_training_1_1_py.pbtxt'

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

  def _generate_cpp_AS_graph():
    rewritten_graph_output_name = 'conv_training_1_1_cpp'

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
                  useNCCLBackend=True)

  # Default call these two generating method.
  _generate_python_frontend_graph()
  _generate_cpp_AS_graph()
