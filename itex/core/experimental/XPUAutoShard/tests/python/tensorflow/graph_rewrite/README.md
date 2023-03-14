This folder contains test case for single conv model as well as resnet50.

# Quick Start

## Build tensorflow with patch
The patch is on `XPUAutoShard/patches` folder. It is tested on r2.7.0, and is supposed to work on other tf version.

This patch adds a roundtrip optimizer pass the same as tfg. It directly copies the graph. It is close by default, the user have to enable by setting the flag:

```Python
os.environ["AS_ROUND_TRIP"]="true" # flag which triggers reading the graph
```

The user could control the graph dumping using the following flags:

```Python
os.environ["AS_DUMP_ORIGINAL_GRAPH"] = "true" # flags, whether dump original graph.

os.environ["AS_FILE_NAME"] = pbtxt_path # absolute path of the pbtxt graph to read
os.environ["AS_TASK_NAME"] = graph_run_property # dumped graph name.

```

## Run the model
Rerun the roundtrip.py
  ```Shell
  $PYTHONPATH=$PYTHONPATH:$PWD:/path/to/models \
   python conv_roundtrip.py
  ```

# API and File Structure

## Interface
The `utils/base_test_class.py` provides a base test class `ASUnitTestBase`. Every test should inherit this base class. The base class provides the following API:

```Python
class TestModule(ASUnitTestBase):
  """Test Module which inherit ASUnitBase
  """

  def define_model(self):
    '''Define Models used by all testcases in this UT.
    
    Return a callable keras model object. Note that this function does not 
    defines training/inference function. e.g., the loss_fn/optimizers are not
    defined in this function. They should be construct in each test run. 
    '''
    class Model(tf.keras.Model):
      pass
    return Model()

  def setUp(self):
    super(TestModule, self).setUp()

  def define_common_config(self):
    '''Define common configs for all testcases in this UT. This will set loss_fn,
    input_shape, etc. If not set, will use all defaults.
    
    '''
  def test_case1(self):
    # Copy common config and set test_specified config.
    local_config = deepcopy(self._common_config)
    # Set seperate run_properties
    local_config.set_graph_run_properties(
      graph_run_properties_list=[
        cfg.GraphRunProperty(run_mode=cfg.RunMode.SINGLE_DEVICE,
                             generate_original_graph=True),
        cfg.GraphRunProperty(cfg.RunMode.AUTO_SHARDING)
      ],
      test_method_name=self.get_separate_test_name(),
      graph_dump_prefix=self.get_separate_test_name()
    )

    self.run_and_compare(local_config)

```
The user is expected to override the `define_models()` to define a callable forward model, and override the `define_common_config()` to provide `loss_fn`, `optimizer` etc.

For each test, the running mode is set between 4 modes: `SINGLE_DEVICE`,`AUTO_SHARDING`,`RUN_IMPORTED_GRAPH`,`DISTRIBUTE_STRATEGY`. See `utils/config_definitions.py` for detail.

When the config is set, the user could call `run_and_compare(config)` to run the test. This function is a uniform API for all test case. 

`run_and_compare()` function will run according to the given `graph_run_properties_list()`. It is expected that the each run will return a tuple (loss,grads), if the user set the `compare_losses=True` or `compare_grads=True` flag, then the test will compare against each run. Assuming that all the running result should be the same.

Examples:
1. `test_conv_roundtrip.py` for the simpliest running config.
2. `test_resnet_block.py` for defining custom loss_fn and running with `tf.distribute` strategy.
3. `test_resnet_full_training.py` for defining the model outside and only use the interface of the class.

## Use already generated graph
On each folder there is already two pbtxt files. One with `*_ori.pbtxt` is the original graph,
and `*_1_2.pbtxt` is the modified graph. 

Under `pbtxt` folder ,there are two classes of graph. 

- `dynamic_dim`, The node's batch dim is dynamic, and it does not specialize shapes.
-  `fixed_dim`, which means the total batch size is fixed for data node. "x" in Conv node for example. Note that the fixed graph under this folder is not traced from actual running, but modified directly from corresponding dynamic graph. This is because we want to avoid shape specialization on some unwanted nodes (e.g., const node). For example, for resnet_block, we specialize shape for only "x" and "y", but don't specialize shape for other nodes.

For each folder, there are two data layouts : `nchw` and `nhwc`. The `nhwc` is by default converted to `nchw` because of the grappler's `layout_optimizer`. Thus, the graph under `nhwc` folder is traced with `layout_optimizer` disabled.

### Conv Graph
Those graph are all with bs=64 and two GPUs case.
The "1_2" means GPU:0 takes bs=32 and 1 iteration, GPU:1 takes bs=16 on 2 iterations. For cases like `*_2_0.pbtxt`, it means GPU:0 takes 2 iteration on single device.

The folder `pbtxt/conv` contains pbtxt for following models:

- conv_fwd : single conv graph only for forward pass
- conv_training : single conv training graph

### ResNet Graph
The graph inside set every iteration have bs=32. All graphs are on GPU device.
The "1_2" means GPU:0 takes bs=32 and 1 iteration, GPU:1 takes bs=32 on 2 iterations. Thus the total batch size is 96. For cases like `*_2_0.pbtxt`, it means GPU:0 takes 2 iteration on single device.

The folder `pbtxt/resnet` contains pbtxt for following models:

- resnet/resnet_block : simplified ResNet graph, containing simple conv+bn plus an additional shortcut.
- resnet/rn50 : Full resnet-50 model training. Note that for test purpose, we change the [`num_replicas`](https://github.com/tensorflow/models/blob/r2.7.0/official/vision/image_classification/resnet/resnet_runnable.py#L161) in resnet model to hardcoded actual GPU numbers. Thus guarantee the Mirrored Strategy would get the same number as XPUAutoShard, because XPUAutoShard would get a single device graph, i.e., it will be `1` in current case, which is not match with MirroredStrategy case.

## Regenerate graph
### Dump original graph
By default, if the user set the `run_mode` to `SINGLE_DEVICE`, and set `generate_original_graph=True`, then the code will run the model once and dump the single device graph under `./TestModuleName/` folder. The original graph will have a suffix `_ori`.

```Python
cfg.GraphRunProperty(run_mode=cfg.RunMode.SINGLE_DEVICE,
                             generate_original_graph=True),
```
### Rewrite the graph by hand
If the `run_mode` is `AUTO_SHARDING`, and the `pbtxt` is not set:
```Python
cfg.GraphRunProperty(run_mode=cfg.RunMode.AUTO_SHARDING,
              auto_sharding_config # See rewrite_graph() function for full flags.
)
```
Then the code will try to auto shard the graph using given config. The config is the same with `rewrite_graph()` function under [rewrite_graph_helper.py](utils/rewrite_graph_helper.py). Please see that for detail.

You could also rewrite the graph by hand using the `rewrite_graph()` function.

# API details
## `config_definitions.py`
Providing two classes to control run-time behavior.

- `GraphRunProperty`: A class owning per-single-test running a property. It controls how to run for a single time. For example, its `run_mode` contains how to run this iteration. If the `run_mode` is `SINGLE_DEVICE`, then it runs on a single device. If the `run_mode` is `AUTO_SHARDING`, then it run the auto-sharded graph.
-  `ASConfig`: A class contains configurations which are shared across all the `TestModule`, all `TestModule.test_case` shares the same `ASConfig` instance. (but local_config is also allowed for each separate run). The `ASConfig` class contains a list of runnable construction fields, for example, `optimizer`, `loss_fn`, etc.

## `base_test_class.py`
Providing the main base test class `ASUnitTestBase`. Every unittest is supposed to inherit this class, and use its function to run the test.

### Functions to Override
**Required functions to override:**
`define_models()`: The function defining fwd model. Return a runnable `Model()` obj.

**Optional functions to override:**
- `define_common_config()`: This is where the corresponding `ASConfig.memer_field`s are defined.  For example, the `loss_fn` and `optimizer` should be defined here.
- `generate_data()`: How to generate data, return a `data_generator_fn`. The `data_generator_fn()` is a callable obj which returns (data,label) tuple.
- `construct_runnable_fn()`: How a single iteration runnable obj is constructed. For example, it returns a `single_step_run_fn()` for a single step training.
- `compare_metrics()` : How to use the return results for comparison.

### Functions to Run:
- `run_and_compare` : The only API exposed to the user for running a config. The user is expected to run from this entry function.

- `direct_run`: direct run the callable fn, it acts as a quick running function. However, the user is not expected to use it.
- `run_single_device` : Run the runnable function using single device strategy.
- `run_imported_graph`: Run the runnable function,  using the imported graph. This function is shared in both `RUN_IMPORTED_GRAPH` and `AUTO_SHARDING` mode.
- `run_with_strategy`: Run the runnable with given tf.distribute strategy.



