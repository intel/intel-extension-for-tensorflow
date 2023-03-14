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

import dataclasses
import enum
from typing import Any, Callable, Dict, List
import tensorflow as tf
from dataclasses import field


class GenerateGraphMode(enum.Enum):
  DYNAMIC_SHAPE_GRAPH = 0
  CONCRETE_SHAPE_GRAPH = 1
  BOTH = 2


class RunMode(enum.Enum):
  """Which mode should the model runs on. It contains configurations about
  individual test dumping graph as well as.

  Attributes:
    `SINGLE_DEVICE`: run the model on single device only (without distribution strategy).
    `AUTO_SHARDING`: run the model with AutoSharding backend. It reads the pbtxt from given path and run.
      If not given, will try to run on a single device and generate the pbtxt.
      This contains both running with AutoSharding CPP backend and Python frontend generated graph.
    `RUN_IMPORTED_GRAPH`: The same with AUTO_SHARDING. Except that it requires that the pbtxt should be given.
    `DISTRIBUTE_STRATEGY`: run the model with distribution strategy. Currently only support MirroredStrategy.
  """
  SINGLE_DEVICE = 0
  AUTO_SHARDING = 1
  RUN_IMPORTED_GRAPH = 2
  DISTRIBUTE_STRATEGY = 3


class GraphRunProperty():
  """Graph property that every test run. It will contain infos controlling what
  mode should the test run, how the graph will be dumped,etc.

  Args:
  `run_mode`(RunMode): What mode should this test run.
  `graph_dump_prefix`(str): What folder should the dumped graph be placed. It will set the TF CPP
    `TF_DUMP_GRAPH_PREFIX` flag. Default is current folder, i.e. `"."`.
  `generate_original_graph(bool)`: A bool flag for whether the original graph would be dumped. (TODO: remove it, not needed)
    Default is False.
  `pbtxt_path(str)`: Path to the pbtxt file. If it is not None, then the test will set corresponding environment
    and read the graph to run. Default is None.
  `run_dynamic_shape_flag(bool)`: Whether the graph run in dynamic shape flag. i.e., whether the batch size
    is specified.
  `dist_strategy(tf.distribute.Strategy)`: Distribute strategy.
  
  It also supports indicating the rewrite_graph() args. See rewrite_graph_helper.py for detail.
  """

  def __init__(self,
               run_mode: RunMode,
               graph_dump_prefix: str = ".",
               generate_original_graph: bool = False,
               pbtxt_path: str = None,
               run_dynamic_shape_flag: bool = False,
               dist_strategy: tf.distribute.Strategy = None,
               graph_name:str = None,
               **kwargs,
               ) -> None:
    self.run_mode: RunMode = run_mode
    self.pbtxt_path: str = pbtxt_path
    self.generate_original_graph: bool = generate_original_graph
    self.graph_dump_prefix = graph_dump_prefix
    self.dist_strategy = dist_strategy
    # The testMethodName will be set outside the __init__ function. It is set when configuring the ASConfig.
    #   The purpose of this design is to reduce the arguments that the user have to set when initializing RunMode.
    self.test_method_name: str = None
    self.graph_name: str = graph_name

    # If true, generate dynamic graph, otherwise generate concrete graph.
    self.run_dynamic_flag: bool = run_dynamic_shape_flag

    # Rewrite graph related args.
    self.rewrite_method: str = kwargs.pop("rewrite_method", "CPP_AUTO_SHARD")
    self.device_ids: list[int] = kwargs.pop("device_ids",[1,2])
    self.device_names: list[str] = kwargs.pop("device_names", ['GPU:0', 'GPU:1'])
    self.device_scores: list[float] = kwargs.pop("device_scores",[1.0,1.0])
    self.device_stages: list[int] = kwargs.pop("device_stages",[1,1])
    self.total_bs:int = kwargs.pop("total_bs", 64)
                  
    # Python frontend Related args
    self.py_allreduce_on:str=kwargs.pop("py_allreduce_on", "GPU:0")
                  
    # CPP_AUTO_SHARD related args
    self.cpu_host:bool=kwargs.pop("cpu_host", False)
    self.host_score:float=kwargs.pop("host_score", 1.0)
    self.host_name:str=kwargs.pop("host_name", "")
    self.strategy:str=kwargs.pop("strategy", "HEURISTIC")
    self.useNCCLBackend:bool=kwargs.pop("useNCCLBackend", True)



  def get_graph_name(self):
    # Construct testMethod Name
    if self.test_method_name is None:
      raise ValueError(
          "The testMethodName is not set. This normally means that the runmode is not set."
      )
    if self.graph_name is None:
      import_graph_suffix = "_no_graph" if self.pbtxt_path is None else "_w_graph"
      self.graph_name: str = self.test_method_name + "_" + self.run_mode.name + import_graph_suffix

    return self.graph_name


@dataclasses.dataclass
class ASConfig:
  """The AutoSharding Config for running tests. It acts as a common config for
  all test cases. If one wants to override common config, one could create a
  new ASConfig object and use that one.

  This class contains flags when constructing the testcase.
  """
  # Model runnable related flags
  construct_runnable: bool = True
  run_synthetic: bool = True
  generate_data: bool = True
  random_seed = 42

  # Running related flags
  graph_run_properties_list: List[GraphRunProperty] = field(
      default_factory=lambda: [GraphRunProperty(RunMode.SINGLE_DEVICE)])

  # training / inference mode.
  training: bool = True

  # tf.function related
  ignore_input_signature: bool = True

  # Training related, optimizers and loss_fn
  # TODO : learning rate currently set as a const, should be a callable instead.
  learning_rate: float = 1e-3
  reduction: Any = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

  optimizer: Callable = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  loss_fn: Callable = tf.keras.losses.MeanSquaredError(reduction=reduction)

  # Single test specified field
  input_shape: List[int] = None
  label_shape: List[int] = None
  input_dtype = tf.float32
  label_dtype = tf.float32

  # Grappler options, by default, the layout optimizer is closed.
  grappler_options: Dict[str, Any] = field(
      default_factory=lambda: {"layout_optimizer": False})

  # Distribute related options.
  # tf.distribute.ReduceOp, choose between "MEAN" and "SUM".
  strategy_reduce_op: "str" = "MEAN"

  def set_graph_run_properties(
      self,
      graph_run_properties_list: List[GraphRunProperty],
      test_method_name: str,
      graph_dump_prefix: str = "."):
    """A helper function to setUp GraphRunProperties.

    It will set the common fields in all elements in
    graph_run_properties_list.
    """
    self.graph_run_properties_list = graph_run_properties_list
    # Iteratively set common fields in every run_mode.
    for graph_run_property in graph_run_properties_list:
      graph_run_property.test_method_name = test_method_name
      graph_run_property.graph_dump_prefix = graph_dump_prefix
