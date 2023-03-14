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

import atexit
from copy import deepcopy
from typing import Any, Callable, Dict, List
import os

import tensorflow as tf
from intel_extension_for_tensorflow.python.test_func import test_util
import utils.config_definitions as cfg
import contextlib


class ASUnitTestBase(test_util.TensorFlowTestCase):
  """AutoSharding Unit Test Base class.

  This class acts as a base class for every unittest case.
  It offers member functions for the following:

  `setUp()`: A function which implicitly call define_common_config() and define_models() function.
  `define_common_config()`: A function which defines default configs. The derived class could override it and set 
    corresponding configurations.
  `define_model()`: The function which defines a fwd model. This function is expected to have pure keras model + a call() function. 
    The return value should be a model instance.
  `run_and_compare()` : The entry point of running all tests. User is encouraged to use this function instead of using other
    helper functions like `run_single_device()`.

  Helper Functions:
  `generate_data()` : Generate synthetic data. Return a tuple (input_data, input_label).
  `_construct_runnable_fn()` : Construct a single iteration runnable object. Internal use only.
  `run_single_device()` : Run the model on single device.
  `run_imported_graph()` : Run the model with imported graph.
  `run_with_strategy()` : Run the model using given tf.distribute.strategy()
  """

  def setUp(self, verbose=False):
    """Setup following args by calling corresponding `define_xyz` function.

    This function sets the following fields:
    `_common_config`: Set the commonly used config for all test cases under current UT. It will set by calling
        `self.define_common_config()` function. The user could override the `define_common_config()` function
        for common controls.
    `_model`: The model used in this UT. This model will be shared in all test cases of current UT. Note that this
        model only contains a keras model without loss/bwd passes. The loss/opt/gradient calculation should be put
        into separate test case. This is because we would like the user to have a full control over different losses
        as well as bwd function. The full runnable is constructed using `self._construct_runnable_fn()`.
    """
    super(ASUnitTestBase, self).setUp()
    self._common_config: cfg.ASConfig = self.define_common_config()
    
    self.set_verbose_flags(verbose)
    self._model = self.define_model()

  def define_model(self):
    """Define Model.

    Return a model object with forward pass.This function
    is expected to have pure keras model + a call() function. 
    The return value should be a model instance, i.e., 
    
    ```Python
    def define_model():
      class Model(tf.keras.Model):
        def __init__(self):
          pass
        def call(self):
          # A forward pass when calling the model.
          pass
    return Model()
    ```
    """
    pass

  def get_separate_test_name(self):
    """
    Return the Test name of current test. Should be something like 
    `TestClassName_TestMethodName`.
    """
    return self.__class__.__name__ + '_' + self._testMethodName

  def define_common_config(self):
    """Define Common Configuration in all testes in this class.

    Return a ASConfig obj containing all common configs. See
    `ASConfig` definition under config_definitions.py for detail.

    The child testcase would override this function by setting its own
    define_common_config() function.
    """
    config = cfg.ASConfig()

    return config

  def generate_data(self, local_config: cfg.ASConfig = None):
    """An internal wrapper which generates data using self._data_fn.

    Return (data,label) pair.
    """

    config = self._common_config if local_config is None else local_config
    return self.generate_synthetic_data(input_shape=config.input_shape,
                                        label_shape=config.label_shape,
                                        input_dtype=config.input_dtype,
                                        label_dtype=config.label_dtype,
                                        random_seed=config.random_seed)

  def generate_synthetic_data(self,
                              input_shape,
                              label_shape,
                              input_dtype=tf.float32,
                              label_dtype=tf.float32,
                              random_seed=42):
    """Generate synthetic data for given input and label shape.

    If user wish to generate using other random number generator, one
    could override this function.
    """
    # Set random seed
    tf.random.set_seed(random_seed)

    inputs = tf.random.normal(input_shape, dtype=input_dtype, name='inputs')
    labels = tf.random.normal(label_shape, dtype=label_dtype, name='labels')

    return inputs, labels

  def construct_runnable_fn(self, **kwargs: Dict[str, Any]):
    """A helper class to construct runnable function. This should be a single
    device training/inference step. It will be put into distribution strategy
    context.

    Expected Args:
    `graph_run_property(GraphRunProperty)` : Graph run property for this run. Required.
    `local_config(ASConfig)` : The ASConfig class. See ASConfig class under config_definitions.py
      for detail. Required.
    `custom_model(Optional[keras.model])`: The model in this run.
    `kwargs`: Any (key,value) pairs that would useful in this function.

    To make the fn suitable for all cases, the above three args are stored in kwargs using [key,value] pair.
     The user could override this function to provide a flexible construction for the runnable fn.
    """
    graph_run_property = kwargs.pop("graph_run_property")
    local_config = kwargs.pop("local_config")
    custom_model = kwargs.pop("custom_model", None)

    input_shape = deepcopy(local_config.input_shape)
    label_shape = deepcopy(local_config.label_shape)
    if custom_model is not None:
      model = custom_model
    else:
      model = self._model

    if graph_run_property.run_dynamic_flag is True:
      #   # TODO: Current assumes that dynamic graph only folds batch size.
      #   # Thus set the batch dim to None
      input_shape[0] = None
      label_shape[0] = None

    input_dtype = local_config.input_dtype
    label_dtype = local_config.label_dtype

    loss_fn = local_config.loss_fn
    # TODO: A workaround to make input_signature to None when using distributing
    # strategy. This is because during the running, it will throw error
    # "argument after ** must be a mapping, not Tensor"
    if graph_run_property.run_mode is cfg.RunMode.DISTRIBUTE_STRATEGY or \
      local_config.ignore_input_signature == True:
      input_signature = None
    else:
      input_signature = [
          tf.TensorSpec(shape=input_shape, dtype=input_dtype),
          tf.TensorSpec(shape=label_shape, dtype=label_dtype)
      ]
    if local_config.training == True:
      optimizer = local_config.optimizer
      # Put everything inside tf.function for graph mode.
      @tf.function(input_signature=input_signature)
      def single_step_fn(*args):
        x, y = args
        with tf.GradientTape() as tape:
          logits = model(x, training=True)
          loss_val = loss_fn(y, logits)
        grads = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_val, grads
    else:

      @tf.function(input_signature=input_signature)
      def single_step_fn(x, y):
        loss_val = model(x, training=False)
        loss_val = loss_fn(loss_val, y)
        return loss_val, 0

    # TODO: Currently, there is no used case in kwargs, thus it is just
    #     kept for future use. If user provides any additional args, we
    #     throw an output as a notice.
    if len(kwargs) != 0:
      print("The args given in construct_running_fn is not fully handled")
      print("The unhandled args:", list(kwargs.keys()))

    return single_step_fn

  def run_single_device(self, *args, **kwargs):
    """A function which run on single device. This function is retrieved as the
    original single device running for conveniently debugging original model as
    well as dumping the original graph.

    Args:
    `args`: Input data and label. Normally would be a (input_data, label_data) tuple.
    `graph_run_property(GraphRunProperty)`: A GraphRunProperty object for this single run.
    `local_config(ASConfig)`: An ASConfig argument which overrides the default one.
      If not specified, will use the default one defined in the `self._common_config`.
    `runnable(Callable)`: A callable function for a single step train.

    To make the fn suitable for all cases, the above three args are stored in kwargs using [key,value] pair.
     The user could override this function to provide a flexible construction for the runnable fn.
    """
    graph_run_property = kwargs.pop("graph_run_property")
    local_config = kwargs.pop("local_config")
    runnable = kwargs.pop("runnable", None)

    # Construct runnable
    if runnable is None:
      runnable = self.construct_runnable_fn(
          graph_run_property=graph_run_property, local_config=local_config)

    results = runnable(*args)

    return results

  def run_with_strategy(self, *args, **kwargs):
    """A function which run the runnable function using given distribute strategy.

    Args:
    `strategy`: a tf.distribute.Strategy instance. 
      E.g.:  tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"]).
    `args`: Input data and label. Normally would be a (input_data, label_data) tuple.
    `graph_run_property(GraphRunProperty)`: A GraphRunProperty object for this single run.
    `local_config(ASConfig)`: An ASConfig argument which overrides the default one.
      If not specified, will use the default one defined in the `self._common_config`.
    `runnable(Callable)`: A callable function for a single step train.

    To make the fn suitable for all cases, the above three args are stored in kwargs using [key,value] pair.
     The user could override this function to provide a flexible construction for the runnable fn.
    """
    strategy = kwargs.pop("strategy")
    graph_run_property = kwargs.pop("graph_run_property")
    local_config = kwargs.pop("local_config")
    runnable = kwargs.pop("runnable", None)

    # [Note]: A workaround for explicitly closing multiprocessing during shutdown
    # Refer to : https://github.com/tensorflow/tensorflow/issues/50487
    atexit.register(strategy._extended._collective_ops._pool.close)

    # Distribute data.

    # TODO : Currently only supports distributing the dataset obj.
    input_data, label = args
    dataset = tf.data.Dataset.from_tensors((input_data, label))
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    with strategy.scope():
      distribute_model = self.define_model()
      distribute_model(input_data, training=False)
      distribute_model.set_weights(self._model.get_weights())

      if runnable is None:
        runnable = self.construct_runnable_fn(
            graph_run_property=graph_run_property,
            local_config=local_config,
            custom_model=distribute_model)

    # construct distributing runnable.
    def dist_runnable(dist_dataset):
      x, y = dist_dataset
      per_replica_losses, per_replica_grads = strategy.run(runnable,
                                                           args=(x, y))
      reduced_loss = strategy.reduce(local_config.strategy_reduce_op,
                                     per_replica_losses,
                                     axis=None)

      reduced_grads = strategy.reduce(local_config.strategy_reduce_op,
                                      per_replica_grads,
                                      axis=None)
      return reduced_loss, reduced_grads

    # TODO : Only run one iteration for now
    for dist_data in dist_dataset:
      reduced_results = dist_runnable(dist_data)
      return reduced_results

  def run_imported_graph(self, *args, **kwargs):
    """A function which run the given graph.

    This function is used by both the autosharding and naive run with graph.

    Args:
    `args`: Input data and label. Normally would be a (input_data, label_data) tuple.
    `graph_run_property(GraphRunProperty)`: A GraphRunProperty object for this single run.
    `local_config(ASConfig)`: An ASConfig argument which overrides the default one.
      If not specified, will use the default one defined in the `self._common_config`.
    `runnable(Callable)`: A callable function for a single step train.

    To make the fn suitable for all cases, the above three args are stored in kwargs using [key,value] pair.
     The user could override this function to provide a flexible construction for the runnable fn.
    """
    graph_run_property = kwargs.pop("graph_run_property")
    local_config = kwargs.pop("local_config")
    runnable = kwargs.pop("runnable", None)

    if graph_run_property.run_mode is not cfg.RunMode.RUN_IMPORTED_GRAPH and \
       graph_run_property.run_mode is not cfg.RunMode.AUTO_SHARDING:
      raise ValueError(
          "Trying to call run_imported_graph with unsupported run_mode : {}".
          format(graph_run_property.run_mode.name))

    # Construct runnable
    if runnable is None:
      runnable = self.construct_runnable_fn(
          graph_run_property=graph_run_property, local_config=local_config)

    # Set the corresponding environments.
    if graph_run_property.pbtxt_path is None:
      # For RUN_IMPORTED_GRAPH mode, it is required to provide a pbtxt.
      if graph_run_property.run_mode is cfg.RunMode.RUN_IMPORTED_GRAPH:
        raise ValueError(
            "pbtxt path must be set when running in mode {}!".format(
                graph_run_property.run_mode.name))

    results = runnable(*args)

    return results

  def run_and_compare(self,
                      local_config: cfg.ASConfig = None,
                      custom_runnable: Callable = None,
                      compare_losses: bool = False,
                      compare_grads: bool = False,
                      direct_run_flag: bool = False,
                      *args):
    """Run and compare following the config.

    By default, this will run without comparing anything. The user could set `compare_losses=True` and
    `compare_grads=True` to specify comparing metrics.

    Args:
    `local_config`: The config for the running. See `ASConfig` class for full definition.
            If not given, the config is the default `self._common_config`.

    `custom_runnable`: A runnable function. If not given, then it will be constructed using self._runnable_fn_ptr().
    `direct_run_flag(bool)` : A flag indicating whether direct run the callable custom_runnable.
      If set to True, the function will call `custom_runnable(*args)` directly.
    `compare_losses(bool)`: If true, will compare losses.
    `compare_grads(bool)` : If true, will compare grads.
    `*args`: Running args.
    """

    if direct_run_flag == True:
      return self.direct_run(local_config, custom_runnable, *args)

    # If the local_config not given, then use the default one.
    if local_config is None:
      local_config = self._common_config

    # [Note]: Since assertAllClose only supports 2 arguments, and there is no big motivation to compare a list of config,
    # thus, we only support two GraphRunProperty in one test run for now.
    if len(local_config.graph_run_properties_list) > 2 and \
      (compare_losses==True or compare_grads==True):
      raise NotImplementedError(
          "Currently, only implement comparing at most 2 result!")

    # Generate input_data and label, this will be used in all the tests.
    input_data, label = self.generate_data(local_config=local_config)

    # Run for every configuration in `local_config.run_mode`` and test the result.

    # Default compare against loss & grads, they stored in a tuple (loss,val)
    results_list: list = []

    # [Note]: Must explicitly run for one iteration to construct model and get weights.
    self._model(input_data, training=False)
    # Store weights
    weight = self._model.get_weights()

    for graph_run_property in local_config.graph_run_properties_list:
      self.clean_env_running_args()
      # Reset weight before every run
      self._model.set_weights(weight)

      # Set HS running flags in TF.
      if graph_run_property.run_mode is cfg.RunMode.AUTO_SHARDING:
        import intel_extension_for_tensorflow as itex
        config = itex.ShardingConfig()
        config.auto_mode = False
        device_gpu = config.devices.add()
        device_gpu.device_type = "gpu"
        device_gpu.device_num = 2
        device_gpu.batch_size = 64
        device_gpu.stage_num = 1
        graph_opts = itex.GraphOptions(sharding=itex.ON, sharding_config=config)
        itex_cfg = itex.ConfigProto(graph_options=graph_opts)
        itex.set_config(itex_cfg)
        # Directly run with the given config
        results = self.run_single_device(input_data,
                                         label,
                                         graph_run_property=graph_run_property,
                                         local_config=local_config,
                                         runnable=custom_runnable)

      elif graph_run_property.run_mode is cfg.RunMode.RUN_IMPORTED_GRAPH:
        results = self.run_imported_graph(input_data,
                                          label,
                                          graph_run_property=graph_run_property,
                                          local_config=local_config,
                                          runnable=custom_runnable)

      elif graph_run_property.run_mode == cfg.RunMode.SINGLE_DEVICE:
        import intel_extension_for_tensorflow as itex
        graph_opts = itex.GraphOptions(sharding=itex.OFF)
        itex_cfg = itex.ConfigProto(graph_options=graph_opts)
        itex.set_config(itex_cfg)
        # Directly run with the given config
        results = self.run_single_device(input_data,
                                         label,
                                         graph_run_property=graph_run_property,
                                         local_config=local_config,
                                         runnable=custom_runnable)
      elif graph_run_property.run_mode == cfg.RunMode.DISTRIBUTE_STRATEGY:
        results = self.run_with_strategy(
            input_data,
            label,
            strategy=graph_run_property.dist_strategy,
            graph_run_property=graph_run_property,
            local_config=local_config,
            runnable=custom_runnable)
      else:
        raise ValueError(
            "Not implemented for the mode {}".format(graph_run_property))

      # Only append loss & grads when there is no custom runnable. If
      results_list.append(results)
    self.compare_metrics(results=results_list,
                         compare_losses=compare_losses,
                         compare_grads=compare_grads)

  def direct_run(self,
                 local_config: cfg.ASConfig = None,
                 custom_runnable: Callable = None,
                 *args):
    """Direct run given runnable function. It is used when there is already a runnable defined outside.
    """
    for graph_run_property in local_config.graph_run_properties_list:
      # This dump is only for debugging purpose.
      if graph_run_property.generate_original_graph == True:
        os.environ["AS_DUMP_ORIGINAL_GRAPH"] = "true"
        os.environ[
            "TF_DUMP_GRAPH_PREFIX"] = graph_run_property.graph_dump_prefix
        os.environ["AS_TASK_NAME"] = graph_run_property.get_graph_name()

      if graph_run_property.run_mode is cfg.RunMode.AUTO_SHARDING or \
          graph_run_property.run_mode is cfg.RunMode.RUN_IMPORTED_GRAPH:

        if graph_run_property.pbtxt_path is None:
          # For RUN_IMPORTED_GRAPH mode, it is required to provide a pbtxt.
          if graph_run_property.run_mode is cfg.RunMode.RUN_IMPORTED_GRAPH:
            raise ValueError(
                "pbtxt path must be set when running in mode {}!".format(
                    graph_run_property.run_mode.name))

      result = custom_runnable(*args)
      return result

  def compare_metrics(self, results: List[Any], **kwargs):
    """Compare results. User could override this function for custom comparison.
    
    """
    # By default , the returned list would be a list[(loss,val)] list.
    # Thus, we first retrieve it out and compare.
    # Otherwise, do nothing by default.
    # Directly return true if there is only one result.
    if len(results) < 2:
      self.assertTrue(True)
    if kwargs.pop("compare_losses", True) == True:
      loss_0 = results[0][0]
      loss_1 = results[1][0]
      self.assertAllClose(loss_0, loss_1)

    if kwargs.pop("compare_grads", False) == True:
      grad_0 = results[0][1]
      grad_1 = results[1][1]
      self.assertAllClose(grad_0, grad_1)

  def clean_env_running_args(self):
    """ Will call before each test. 
    This is commonly a clean flag/global variables related to each test."""

    env_vars = [
        "AS_DUMP_ORIGINAL_GRAPH", "AS_FILE_NAME", "AS_TASK_NAME",
        "AS_ROUND_TRIP", "TF_DUMP_GRAPH_PREFIX"
    ]
    set_env_vars = set(env_vars) - set(os.environ)

    for env_var in set_env_vars:
      os.unsetenv(env_var)

  def _get_ori_pbtxt_name(self):
    """Return original pbtxt name. The design is one TestModule shares the same model,
    thus, the original graph is the same, the original generated pbtxt would be 'TestModule_ori.pbtxt'"""
    return self.__class__.__name__ + "_ori"

  def _get_ori_pbtxt_folder_name(self):
    return self.__class__.__name__
  
  def set_verbose_flags(self, verbose = False):
    """Set verbose mode for TF. By default, will bypass log info.
    """
    # This flag suppress debugging information of tensorflow.
    if verbose == False:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
      tf.get_logger().setLevel('ERROR')
  
  def new_local_config(self):
    """Return a copy of common config.
    """
    return self._common_config
    