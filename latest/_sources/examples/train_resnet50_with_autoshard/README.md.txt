# Accelerate ResNet50 Training by XPUAutoShard on Intel GPU

## Introduction

The XPUAutoShard feature of Intel® Extension for TensorFlow* automatically shards the input data to the Intel® GPU devices. Currently, it supports applying the shards on multiple GPU tiles to maximize the hardware utilization and improve performance.  

This example shows ResNet50 training speedup with XPUAutoShard enabled.

## Hardware Requirements

Verified Hardware Platforms:
- Intel® Data Center GPU Max Series
   
## Prerequisites
 
This example only applies to stock TensorFlow* >=2.13.0 and Intel® Extension for TensorFlow* >=2.13.0.0.

### Prepare the Codes
```bash
git clone https://github.com/tensorflow/models tf-models
cd tf-models
git checkout r2.13
git apply ../shard.patch
```
 
### Prepare for GPU

Refer to [Prepare](../common_guide_running.html#prepare)

### Install Other Required Packages

```bash
pip install -r official/requirements.txt
```

### Enable Running Environment

Refer to [Running](../common_guide_running.html#running) to enable oneAPI running environment and virtual running environment.

### Setup PYTHONPATH
Modify `/path/to/tf-models` accordingly, here `~/tf-models` as an example.
```bash
cd official/legacy/image_classification/resnet/
mkdir output
export PYTHONPATH=$PYTHONPATH:/path/to/tf-models:$PWD
```

## Executes the Example with Python API
### Without XPUAutoShard
```bash
export TF_NUM_INTEROP_THREADS=<number of physical core per socket> 
export TF_NUM_INTRAOP_THREADS=<number of physical core per socket>
export BS=256
python resnet_ctl_imagenet_main.py \
--num_gpus=1 \
--batch_size=$BS \
--train_epochs=1 \
--train_steps=30 \
--steps_per_loop=1 \
--log_steps=1 \
--skip_eval \
--use_synthetic_data=true \
--distribution_strategy=off \
--use_tf_while_loop=false \
--use_tf_function=true --enable_xla=false \
--enable_tensorboard=false --enable_checkpoint_and_export=false \
--data_format=channels_last --single_l2_loss_op=True \
--model_dir=output \
--dtype=bf16 2>&1 | tee resnet50.log
```

### With XPUAutoShard

#### Python API
Intel® Extension for TensorFlow* provides Python APIs to enable XPUAutoShard feature as follws:

```python
config = itex.ShardingConfig()
config.auto_mode = False
device_gpu = config.devices.add()
device_gpu.device_type = "gpu"
device_gpu.device_num = 2
device_gpu.batch_size = 256
device_gpu.stage_num = 10
graph_opts = itex.GraphOptions(sharding=itex.ON, sharding_config = config)
itex_cfg = itex.ConfigProto(graph_options=graph_opts)
itex.set_config(itex_cfg)
```

#### Sharding Parameters Setting

In this example, the above code has been added to `resnet_ctl_imagenet_main.py` with the patch and you can enable XPUAutoShard via simply adding `--use_itex_sharding=True` to the command-line. You can optionally modify the following parameters in the `ShardingConfig` based on your need.

|Prameters|Config Suggestions|
|-|-|
|device_num|2 for Intel® Data Center GPU Max Series with 2 tiles|
|batch_size|batch size on each device in each loop of each iteration|
|stage_num|number of training loops on each device with each iteration <br> before the All-reduce and weight updating on GPU devices, <br> set it >=2 to improve scaling efficiency|

The global batch size should be `device_num` * `batch_size` * `stage_num`. In this example, the default global batch size is 2x256x10=5120.

#### Further Settings
For further performance speedup, you can enable multi-stream via setting `ITEX_ENABLE_MULTIPLE_STREAM=1` to create multiple queues for each device.

#### Executing Command
```bash
export TF_NUM_INTEROP_THREADS=<number of physical core per socket> 
export TF_NUM_INTRAOP_THREADS=<number of physical core per socket>
export BS=5120
export ITEX_ENABLE_MULTIPLE_STREAM=1
python resnet_ctl_imagenet_main.py \
--num_gpus=1 \
--batch_size=$BS \
--train_epochs=1 \
--train_steps=30 \
--steps_per_loop=1 \
--log_steps=1 \
--skip_eval \
--use_synthetic_data=true \
--distribution_strategy=off \
--use_tf_while_loop=false \
--use_tf_function=true --enable_xla=false \
--enable_tensorboard=false --enable_checkpoint_and_export=false \
--data_format=channels_last --single_l2_loss_op=True \
--model_dir=output \
--dtype=bf16 \
--use_itex_sharding=true 2>&1 | tee resnet50_itex-shard.log
```

The following output log indicates XPUAutoShard has been enabled successfully:<br>
`I itex/core/graph/tfg_optimizer_hook/tfg_optimizer_hook.cc:280] Run AutoShard pass successfully`

## Example Output
With successful execution, it will print out the following results:

```
...
I0324 07:55:20.594147 140348344015936 keras_utils.py:145] TimeHistory: xxxxx seconds, xxxxx examples/second between steps 0 and 1
I0324 07:55:20.597360 140348344015936 controller.py:479] train | step:      1 | steps/sec:    xxxxx | output: {'train_accuracy': 0.0, 'train_loss': 12.634554}
I0324 07:55:22.161625 140348344015936 keras_utils.py:145] TimeHistory: xxxxx seconds, xxxxx examples/second between steps 1 and 2
I0324 07:55:22.163815 140348344015936 controller.py:479] train | step:      2 | steps/sec:    xxxxx | output: {'train_accuracy': 0.0, 'train_loss': 12.634554}
I0324 07:55:23.790632 140348344015936 keras_utils.py:145] TimeHistory: xxxxx seconds, xxxxx examples/second between steps 2 and 3
I0324 07:55:23.792936 140348344015936 controller.py:479] train | step:      3 | steps/sec:    xxxxx | output: {'train_accuracy': 1.0, 'train_loss': 9.103148}
I0324 07:55:25.416651 140348344015936 keras_utils.py:145] TimeHistory: xxxxx seconds, xxxxx examples/second between steps 3 and 4
I0324 07:55:25.419072 140348344015936 controller.py:479] train | step:      4 | steps/sec:    xxxxx | output: {'train_accuracy': 1.0, 'train_loss': 5.3359284}
I0324 07:55:27.025180 140348344015936 keras_utils.py:145] TimeHistory: xxxxx seconds, xxxxx examples/second between steps 4 and 5
I0324 07:55:27.027671 140348344015936 controller.py:479] train | step:      5 | steps/sec:    xxxxx | output: {'train_accuracy': 1.0, 'train_loss': 5.3343554}
...
```
