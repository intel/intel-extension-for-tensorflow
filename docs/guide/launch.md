# Launch Script User Guide

- [Overview](#overview)
- [Common Execution Mode](#common-execution-mode)
  - [Latency Mode](#latency-mode)
  - [Throughput Mode](#throughput-mode)
- [Basic Settings](#basic-settings)
  - [Launch Log](#launch-log)
- [Advanced Settings](#advanced-settings)
  - [Multi-instance](#multi-instance)
  - [NUMA Control](#numa-control)
  - [Memory Allocator](#memory-allocator)
  - [Environment Viriables](#environment-variables)
- [Examples](#examples)

## Overview

As introduced in the [Practice Guide](practice_guide.md), there are several factors that influence performance. Setting configuration options properly contributes to a performance boost. However, there is no unified configuration that is optimal to all topologies. Users need to try different combinations. A *launch* script is provided to automate these configuration settings to free users from this complicated work. This guide helps you to learn the *launch* script common usage and provides examples that cover many optimized configuration cases as well.

The configurations are mainly around the following perspectives.

- NUMA Control: numactl specifies NUMA scheduling and memory placement policy
- Number of instances: [Single instance (default) | Multiple instances]
- Memory allocator: [TCMalloc | JeMalloc | default Malloc] If unspecified, launcher will choose for user.

## Common Execution Mode

The *launch* script is provided as a module of Intel® Extension for TensorFlow*. Run the following command to use it. If no knob is given, your script will be executed using all physical cores.

```bash
python -m intel_extension_for_tensorflow.python.launch [knobs] <your_script> [your_script_args]
```

In most cases for better performance, *```--latency_mode```* or *```--throughput_mode```* is often enabled. The launcher script will automatically calculate the number of instance and number of cores used for each instance, so no manual setting is required. If you want to customize your execution, see [Advanced Setting](#advanced-settings).

### Latency mode

With *```--latency_mode```*, each instance uses 4 cores and all physical cores are used. This knob is mutually exclusive with *```--throughput_mode```*.

```bash
python -m intel_extension_for_tensorflow.python.launch --latency_mode infer_resnet50.py
```

### Throughput mode

With *```--throughput_mode```*, one numa node corresponds to one instance and all physical cores are used. This knob is mutually exclusive with *```--latency_mode```*.

```bash
python -m intel_extension_for_tensorflow.python.launch --throughput_mode infer_resnet50.py
```

## Basic Settings

### Launch Log

The *launch* script execution creates log files under a designated log directory, so that you can conduct some investigations afterward. By default, creating logs is disabled to avoid undesired log files. You can enable logging by setting knob *```--log_path```* to be:

- directory to save log files. Both absolute path and relative path are supported.
- types of log files to generate. One file (*```<prefix>_timestamp_instances.log```*) contains command and information when the script was launched. Another type of file (*```<prefix>_timestamp_instance_N_core#-core#....log```*) contain stdout print of each instance.

For example:

```text
run_20210712212258_instances.log
run_20210712212258_instance_0_cores_0-43.log
```

## Advanced Settings

The following table lists all available knobs.

| Knob | Type | Default Value | Description |
| :-- | :--: | :--: | :-- |
| *```-m```*, *```--module```* | - | None | Changes each process to interpret the launch script as a python module, executing with the same behavior as *```python -m```*. |
| *```--no_python```* | BOOLEAN | False | Useful when the script is not a Python script. Do not prepend your script with *```python```*, execute it directly. |
| *```--latency_mode```* | BOOLEAN | False | By default each instance uses 4 cores and all physical cores are used. |
| *```--throughput_mode```* | BOOLEAN | False | By default one numa node corresponds to one instance and all physical cores are used. |
| *```--log_path```* | STRING | "" | The log file path. Default path is '', which means disable logging to files. |
| *```--log_file_prefix```* | STRING | "run" | Log file prefix. |

If you want to set the number of instances, core allocation or some environment variables yourself, use the knobs described below, which are all exclusive to *```--latency_mode```* and *```--throughput_mode```*.

### Multi-instance

You may want to launch multiple instances for better performance, for example, when batch size is small. The following knobs will be helpful.

| Knob | Type | Default Value | Description |
| :-- | :--: | :--: | :-- |
| *```--ninstances```* | INTEGER | -1 | Number of instances. |
| *```--instance_idx```* | INTEGER | -1 | Run specified instance_idx instance among multiple instances (instance index starts at index 0). Useful when running each instance independently. |
| *```--ncore_per_instance```* | INTEGER | -1 | Cores per instance. |

### NUMA Control

These knobs are used to set the NUMA policy to better utilize your harware resource.


| Knob | Type | Default Value | Description |
| :-- | :--: | :--: | :-- |
| *```--node_id```* | INTEGER | -1 | Run on the specified node (node index starts at index 0). |
| *```--skip_cross_node_cores```* | BOOLEAN | False | When specifying --ncore_per_instance, set --skip_cross_node_cores to skip any cross-node cores. |
| *```--disable_numactl```* | BOOLEAN | False | Disable numactl. |
| *```--disable_taskset```* | BOOLEAN | False | Disable taskset. |
| *```--use_logical_core```* | BOOLEAN | False | Whether use logical cores. |
| *```--core_list```* | STRING | None | Specify the core list as *`core_id, core_id, ....`*. |

### Memory Allocator

This script provides users three memory allocator types, specified with the following knobs. If not specified, the script will automatically check the installation of allocators on the execution machine, and then select in the order of TCMalloc/JeMalloc/Default Malloc.


| Knob | Type | Default Value | Description |
| :-- | :--: | :--: | :-- |
| *```--enable_tcmalloc```* | BOOLEAN | False | Enable tcmalloc allocator. Ensure TCMalloc is installed before use. |
| *```--enable_jemalloc```* | BOOLEAN | False | Enable jemalloc allocator. Ensure JeMalloc is installed before use. |
| *```--use_default_allocator```* | BOOLEAN |  False | Use default memory allocator. |

### Environment Variables

The *launch* script respects existing environment variables on launch. If you prefer some certain environment variables, you can set them before executing the *launch* script. Intel OpenMP library uses an environment variable *`KMP_AFFINITY`* to control its behavior. Different settings bring different performance. By default, the *launch* script will set *`KMP_AFFINITY`* to "granularity=fine,verbose,compact,1,0" or "granularity=fine,verbose,compact," depending on whether hyper threading is on or off. If you want to try other values, you can use *```export```* command on Linux to set *`KMP_AFFINITY`* before you run the *launch* script. In this case, the script will not set the default value but take the existing value of *`KMP_AFFINITY`*, and print a message to stdout.

Our launcher also automatically set some environment variables related to TensorFlow and Intel® Extension for TensorFlow*. By default, *`TF_NUM_INTEROP_THREADS`* and *`TF_NUM_INTRAOP_THREADS`* are set to *`1`* and number of cores per instance. [ITEX AMP](advanced_auto_mixed_precision.md) and [Intel® Extension for TensorFlow* layout optimization](practice_guide.md#memory-layout-format) are disabled.
Users can change them by the following knobs.

| Knob | Type | Default Value | Description |
| :-- | :--: | :--: | :-- |
| *```--tf_num_intraop_threads```* | STRING | None | By Default, this argument is None, and set environment variable *`TF_NUM_INTRAOP_THREADS`* as the number of cores per instance. |
| *```--tf_num_interop_threads```* | STRING | None | By Default, this argument is None, and set environment variable *`TF_NUM_INTEROP_THREADS`*=1. |
| *```--enable_itex_amp```* | BOOLEAN | False | Set environment variable *`ITEX_AUTO_MIXED_PRECISION=1`*. |
| *```--enable_itex_layout_opt```* | BOOLEAN | False | Set environment variable *`ITEX_LAYOUT_OPT=0`* or *`1`*. |

## Examples

Example script [infer_resnet50.py](../../examples/infer_resnet50/infer_resnet50.py) will be used in this guide.

- Single instance for inference
  - [I. Use all physical cores](#i-use-all-physical-cores)
  - [II. Use all cores including logical cores](#ii-use-all-cores-including-logical-cores)
  - [III. Use physical cores on one node](#iii-use-physical-cores-on-one-node)
  - [IV. Use your designated number of cores](#iv-use-your-designated-number-of-cores)
- Multiple instances for inference
  - [V. Throughput mode (i.e. number of numa node instances, each instance runs on 1 numa node)](#v-throughput-mode)
  - [VI. Latency mode (Use 4 cores for each instance)](#vi-latency-mode)
  - [VII. Your designated number of instances](#vii-your-designated-number-of-instances)
  - [VIII. Your designated number of instances and instance index](#viii-your-designated-number-of-instances-and-instance-index)
- Set environment variables for inference
  - [IX. TF_NUM_INTRAOP_THREADS](#ix-set-environment-variable-tf_num_intraop_threads)
  - [X. TF_NUM_INTEROP_THREADS](#x-set-environment-variable-tf_num_interop_threads)
- Usage of Jemalloc/TCMalloc/Default memory allocator
  - [Jemalloc](#jemalloc)
  - [TCMalloc](#tcmalloc)
  - [Default memory allocator](#default-memory-allocator)

### Single instance for inference

#### I. Use all physical cores

```bash
python -m intel_extension_for_tensorflow.python.launch --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009103552_instance_0_cores_0-43.log
    └── run_20221009103552_instances.log
```

The ```run_20221009103552_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009103552_instances.log
2022-10-09 10:35:53,136 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 10:35:53,136 - __main__ - INFO - OMP_NUM_THREADS=96
2022-10-09 10:35:53,136 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 10:35:53,136 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 10:35:53,136 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 10:35:53,136 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=96
2022-10-09 10:35:53,136 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 10:35:53,136 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 10:35:53,137 - __main__ - INFO - numactl --localalloc -C 0-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009103552_instance_0_cores_0-95.log
```

#### II. Use all cores including logical cores

```bash
python -m intel_extension_for_tensorflow.python.launch --use_logical_core --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009104740_instances.log
    └── run_20221009104740_instance_0_cores_0-191.log
```

The ```run_20221009104740_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009104740_instances.log
2022-10-09 10:47:40,908 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 10:47:40,909 - __main__ - INFO - OMP_NUM_THREADS=192
2022-10-09 10:47:40,909 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 10:47:40,909 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 10:47:40,909 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 10:47:40,909 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=192
2022-10-09 10:47:40,909 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 10:47:40,909 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 10:47:40,909 - __main__ - INFO - numactl --localalloc -C 0-191 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009104740_instance_0_cores_0-191.log
```

#### III. Use physical cores on one node

```bash
python -m intel_extension_for_tensorflow.python.launch --node_id 1 --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009105044_instances.log
    └── run_20221009105044_instance_0_cores_12-23.log

```

The ```run_20221009105044_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009105044_instances.log
2022-10-09 10:50:44,693 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 10:50:44,693 - __main__ - INFO - OMP_NUM_THREADS=12
2022-10-09 10:50:44,693 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 10:50:44,693 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 10:50:44,693 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 10:50:44,693 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=12
2022-10-09 10:50:44,693 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 10:50:44,693 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 10:50:44,694 - __main__ - INFO - numactl --localalloc -C 12-23 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105044_instance_0_cores_12-23.log
```

#### IV. Use your designated number of cores

```bash
python -m intel_extension_for_tensorflow.python.launch --ninstances 1 --ncore_per_instance 10 --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009105320_instances.log
    └── run_20221009105320_instance_0_cores_0-9.log
```

The ```run_20221009105320_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009105320_instances.log
2022-10-09 10:53:21,089 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 10:53:21,089 - __main__ - INFO - OMP_NUM_THREADS=10
2022-10-09 10:53:21,089 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 10:53:21,089 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 10:53:21,089 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 10:53:21,089 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=10
2022-10-09 10:53:21,089 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 10:53:21,089 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 10:53:21,090 - __main__ - INFO - numactl --localalloc -C 0-9 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105320_instance_0_cores_0-9.log
```

### Multiple instances for inference

#### V. Throughput mode

```bash
python -m intel_extension_for_tensorflow.python.launch --throughput_mode --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009105838_instances.log
    ├── run_20221009105838_instance_0_cores_0-11.log
    ├── run_20221009105838_instance_1_cores_12-23.log
    ├── run_20221009105838_instance_2_cores_24-35.log
    ├── run_20221009105838_instance_3_cores_36-47.log
    ├── run_20221009105838_instance_4_cores_48-59.log
    ├── run_20221009105838_instance_5_cores_60-71.log
    ├── run_20221009105838_instance_6_cores_72-83.log
    └── run_20221009105838_instance_7_cores_84-95.log
```

The ```run_20221009105838_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009105838_instances.log
2022-10-09 10:58:38,757 - __main__ - WARNING - --throughput_mode is exclusive to --ninstances, --ncore_per_instance, --node_id and --use_logical_core. They won't take effect even if they are set explicitly.
2022-10-09 10:58:38,772 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 10:58:38,772 - __main__ - INFO - OMP_NUM_THREADS=12
2022-10-09 10:58:38,772 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 10:58:38,772 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 10:58:38,772 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 10:58:38,772 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=12
2022-10-09 10:58:38,772 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 10:58:38,772 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 10:58:38,772 - __main__ - INFO - numactl --localalloc -C 0-11 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_0_cores_0-11.log
2022-10-09 10:58:38,784 - __main__ - INFO - numactl --localalloc -C 12-23 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_1_cores_12-23.log
2022-10-09 10:58:38,795 - __main__ - INFO - numactl --localalloc -C 24-35 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_2_cores_24-35.log
2022-10-09 10:58:38,806 - __main__ - INFO - numactl --localalloc -C 36-47 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_3_cores_36-47.log
2022-10-09 10:58:38,817 - __main__ - INFO - numactl --localalloc -C 48-59 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_4_cores_48-59.log
2022-10-09 10:58:38,828 - __main__ - INFO - numactl --localalloc -C 60-71 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_5_cores_60-71.log
2022-10-09 10:58:38,839 - __main__ - INFO - numactl --localalloc -C 72-83 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_6_cores_72-83.log
2022-10-09 10:58:38,850 - __main__ - INFO - numactl --localalloc -C 84-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009105838_instance_7_cores_84-95.log
```

#### VI. Latency mode

```bash
python -m intel_extension_for_tensorflow.python.launch --latency_mode --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009110327_instances.log
    ├── run_20221009110327_instance_0_cores_0-3.log
    ├── run_20221009110327_instance_1_cores_4-7.log
    ├── run_20221009110327_instance_2_cores_8-11.log
    ├── run_20221009110327_instance_3_cores_12-15.log
    ├── run_20221009110327_instance_4_cores_16-19.log
    ├── run_20221009110327_instance_5_cores_20-23.log
    ├── run_20221009110327_instance_6_cores_24-27.log
    ├── run_20221009110327_instance_7_cores_28-31.log
    ├── run_20221009110327_instance_8_cores_32-35.log
    ├── run_20221009110327_instance_9_cores_36-39.log
    ├── run_20221009110327_instance_10_cores_40-43.log
    ├── run_20221009110327_instance_11_cores_44-47.log
    ├── run_20221009110327_instance_12_cores_48-51.log
    ├── run_20221009110327_instance_13_cores_52-55.log
    ├── run_20221009110327_instance_14_cores_56-59.log
    ├── run_20221009110327_instance_15_cores_60-63.log
    ├── run_20221009110327_instance_16_cores_64-67.log
    ├── run_20221009110327_instance_17_cores_68-71.log
    ├── run_20221009110327_instance_18_cores_72-75.log
    ├── run_20221009110327_instance_19_cores_76-79.log
    ├── run_20221009110327_instance_20_cores_80-83.log
    ├── run_20221009110327_instance_21_cores_84-87.log
    ├── run_20221009110327_instance_22_cores_88-91.log
    └── run_20221009110327_instance_23_cores_92-95.log
```

The ```run_20221009110327_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009110327_instances.log
2022-10-09 11:03:27,198 - __main__ - WARNING - --latency_mode is exclusive to --ninstances, --ncore_per_instance, --node_id and --use_logical_core. They won't take effect even if they are set explicitly.
2022-10-09 11:03:27,215 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 11:03:27,215 - __main__ - INFO - OMP_NUM_THREADS=4
2022-10-09 11:03:27,215 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:03:27,215 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:03:27,215 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 11:03:27,215 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=4
2022-10-09 11:03:27,215 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:03:27,215 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:03:27,216 - __main__ - INFO - numactl --localalloc -C 0-3 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_0_cores_0-3.log
2022-10-09 11:03:27,229 - __main__ - INFO - numactl --localalloc -C 4-7 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_1_cores_4-7.log
2022-10-09 11:03:27,241 - __main__ - INFO - numactl --localalloc -C 8-11 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_2_cores_8-11.log
2022-10-09 11:03:27,254 - __main__ - INFO - numactl --localalloc -C 12-15 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_3_cores_12-15.log
2022-10-09 11:03:27,266 - __main__ - INFO - numactl --localalloc -C 16-19 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_4_cores_16-19.log
2022-10-09 11:03:27,278 - __main__ - INFO - numactl --localalloc -C 20-23 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_5_cores_20-23.log
2022-10-09 11:03:27,290 - __main__ - INFO - numactl --localalloc -C 24-27 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_6_cores_24-27.log
2022-10-09 11:03:27,302 - __main__ - INFO - numactl --localalloc -C 28-31 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_7_cores_28-31.log
2022-10-09 11:03:27,315 - __main__ - INFO - numactl --localalloc -C 32-35 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_8_cores_32-35.log
2022-10-09 11:03:27,327 - __main__ - INFO - numactl --localalloc -C 36-39 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_9_cores_36-39.log
2022-10-09 11:03:27,339 - __main__ - INFO - numactl --localalloc -C 40-43 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_10_cores_40-43.log
2022-10-09 11:03:27,351 - __main__ - INFO - numactl --localalloc -C 44-47 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_11_cores_44-47.log
2022-10-09 11:03:27,364 - __main__ - INFO - numactl --localalloc -C 48-51 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_12_cores_48-51.log
2022-10-09 11:03:27,376 - __main__ - INFO - numactl --localalloc -C 52-55 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_13_cores_52-55.log
2022-10-09 11:03:27,388 - __main__ - INFO - numactl --localalloc -C 56-59 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_14_cores_56-59.log
2022-10-09 11:03:27,400 - __main__ - INFO - numactl --localalloc -C 60-63 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_15_cores_60-63.log
2022-10-09 11:03:27,413 - __main__ - INFO - numactl --localalloc -C 64-67 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_16_cores_64-67.log
2022-10-09 11:03:27,425 - __main__ - INFO - numactl --localalloc -C 68-71 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_17_cores_68-71.log
2022-10-09 11:03:27,438 - __main__ - INFO - numactl --localalloc -C 72-75 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_18_cores_72-75.log
2022-10-09 11:03:27,452 - __main__ - INFO - numactl --localalloc -C 76-79 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_19_cores_76-79.log
2022-10-09 11:03:27,465 - __main__ - INFO - numactl --localalloc -C 80-83 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_20_cores_80-83.log
2022-10-09 11:03:27,480 - __main__ - INFO - numactl --localalloc -C 84-87 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_21_cores_84-87.log
2022-10-09 11:03:27,494 - __main__ - INFO - numactl --localalloc -C 88-91 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_22_cores_88-91.log
2022-10-09 11:03:27,509 - __main__ - INFO - numactl --localalloc -C 92-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110327_instance_23_cores_92-95.log
```

#### VII. Your designated number of instances

```bash
python -m intel_extension_for_tensorflow.python.launch --ninstances 4 --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009110849_instances.log
    ├── run_20221009110849_instance_0_cores_0-10.log
    ├── run_20221009110849_instance_1_cores_11-21.log
    ├── run_20221009110849_instance_2_cores_22-32.log
    └── run_20221009110849_instance_3_cores_33-43.log
```

The ```run_20221009110849_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009110849_instances.log
2022-10-09 11:08:49,891 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 11:08:49,891 - __main__ - INFO - OMP_NUM_THREADS=24
2022-10-09 11:08:49,891 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:08:49,891 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:08:49,892 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 11:08:49,892 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=24
2022-10-09 11:08:49,892 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:08:49,892 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:08:49,892 - __main__ - INFO - numactl --localalloc -C 0-23 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110849_instance_0_cores_0-23.log
2022-10-09 11:08:49,908 - __main__ - INFO - numactl --localalloc -C 24-47 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110849_instance_1_cores_24-47.log
2022-10-09 11:08:49,930 - __main__ - INFO - numactl --localalloc -C 48-71 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110849_instance_2_cores_48-71.log
2022-10-09 11:08:49,951 - __main__ - INFO - numactl --localalloc -C 72-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009110849_instance_3_cores_72-95.log
```

#### VIII. Your designated number of instances and instance index

Launcher by default runs all `ninstances` for multi-instance inference and training as shown above. You can specify `instance_idx` to independently run that instance only among `ninstances`

```bash
python -m intel_extension_for_tensorflow.python.launch --ninstances 4 --instance_idx 0 --log_path ./logs infer_resnet50.py
```

you can confirm usage in log file:

```text
2022-10-09 11:10:34,586 - __main__ - INFO - assigning 24 cores for instance 0
2022-10-09 11:10:34,604 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 11:10:34,604 - __main__ - INFO - OMP_NUM_THREADS=24
2022-10-09 11:10:34,605 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:10:34,605 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:10:34,605 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 11:10:34,605 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=24
2022-10-09 11:10:34,605 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:10:34,605 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:10:34,605 - __main__ - INFO - numactl --localalloc -C 0-23 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009111034_instance_0_cores_0-23.log
```

```bash
python -m intel_extension_for_tensorflow.python.launch --ninstances 4 --instance_idx 1 --log_path ./logs infer_resnet50.py
```

you can confirm usage in log file:

```text
2022-10-09 11:12:40,129 - __main__ - INFO - assigning 24 cores for instance 1
2022-10-09 11:12:40,144 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 11:12:40,144 - __main__ - INFO - OMP_NUM_THREADS=24
2022-10-09 11:12:40,144 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:12:40,144 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:12:40,144 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 11:12:40,144 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=24
2022-10-09 11:12:40,144 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:12:40,144 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:12:40,145 - __main__ - INFO - numactl --localalloc -C 24-47 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009111239_instance_0_cores_24-47.log
```

### Set environment variables for inference

#### IX. Set environment variable TF_NUM_INTRAOP_THREADS

```bash
python -m intel_extension_for_tensorflow.python.launch --tf_num_intraop_threads 8 --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009111753_instances.log
    ├── run_20221009111753_instance_0_cores_0-95.log
```

The ```run_20221009111753_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009111753_instances.log
2022-10-09 11:17:53,947 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 11:17:53,947 - __main__ - INFO - OMP_NUM_THREADS=96
2022-10-09 11:17:53,947 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:17:53,947 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:17:53,948 - __main__ - INFO - TF_NUM_INTEROP_THREADS=2
2022-10-09 11:17:53,948 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=96
2022-10-09 11:17:53,948 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:17:53,948 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:17:53,948 - __main__ - INFO - numactl --localalloc -C 0-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009111753_instance_0_cores_0-95.log
```

#### X. Set environment variable TF_NUM_INTEROP_THREADS

```bash
python -m intel_extension_for_tensorflow.python.launch --tf_num_interop_threads 2 --log_path ./logs infer_resnet50.py
```

Check your log directory, its structure is as below.

```text
.
├── infer_resnet50.py
└── logs
    ├── run_20221009111951_instances.log
    └── run_20221009111951_instance_0_cores_0-95.log

```

The ```run_20221009111951_instances.log``` contains information and command that were used for this execution launch.

```text
$ cat logs/run_20221009111951_instances.log
2022-10-09 11:19:51,404 - __main__ - WARNING - Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or /home/sdp/.local/lib/ so the LD_PRELOAD environment variable will not be set. This may drop the performance.
2022-10-09 11:19:51,405 - __main__ - INFO - OMP_NUM_THREADS=96
2022-10-09 11:19:51,405 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:19:51,405 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:19:51,405 - __main__ - INFO - TF_NUM_INTEROP_THREADS=2
2022-10-09 11:19:51,405 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=96
2022-10-09 11:19:51,405 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:19:51,405 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:19:51,405 - __main__ - INFO - numactl --localalloc -C 0-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009111951_instance_0_cores_0-95.log
```

### Usage of TCMalloc/Jemalloc/Default memory allocator

Memory allocator can influence performance. If users do not designate a desired memory allocator, the *launch* script searches them in the order of TCMalloc > Jemalloc > Tensorflow default memory allocator, and takes the first matched one.

#### Jemalloc

__Note:__ You can set your preferred value to *MALLOC_CONF* before running the *launch* script if you do not want to use its default setting.

```bash
python -m intel_extension_for_tensorflow.python.launch --enable_jemalloc --log_path ./logs infer_resnet50.py
```

you can confirm usage in log file:

```text
2022-10-09 11:27:20,549 - __main__ - INFO - Use JeMalloc memory allocator
2022-10-09 11:27:20,550 - __main__ - INFO - MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto
2022-10-09 11:27:20,550 - __main__ - INFO - OMP_NUM_THREADS=96
2022-10-09 11:27:20,550 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:27:20,550 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:27:20,550 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 11:27:20,550 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=96
2022-10-09 11:27:20,550 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:27:20,550 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:27:20,550 - __main__ - INFO - numactl --localalloc -C 0-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009112720_instance_0_cores_0-95.log
```

#### TCMalloc

```bash
python -m intel_extension_for_tensorflow.python.launch --enable_tcmalloc --log_path ./logs infer_resnet50.py
```

you can confirm usage in log file:

```text
2022-10-09 11:29:05,206 - __main__ - INFO - Use TCMalloc memory allocator
2022-10-09 11:29:05,207 - __main__ - INFO - OMP_NUM_THREADS=96
2022-10-09 11:29:05,207 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:29:05,207 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:29:05,207 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 11:29:05,207 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=96
2022-10-09 11:29:05,207 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:29:05,207 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:29:05,207 - __main__ - INFO - numactl --localalloc -C 0-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009112905_instance_0_cores_0-95.log
```

#### Default memory allocator

```bash
python -m intel_extension_for_tensorflow.python.launch --use_default_allocator --log_path ./logs infer_resnet50.py
```

you can confirm usage in log file:

```text
2022-10-09 11:29:56,911 - __main__ - INFO - OMP_NUM_THREADS=96
2022-10-09 11:29:56,911 - __main__ - INFO - KMP_AFFINITY=granularity=fine,verbose,compact,1,0
2022-10-09 11:29:56,911 - __main__ - INFO - KMP_BLOCKTIME=1
2022-10-09 11:29:56,911 - __main__ - INFO - TF_NUM_INTEROP_THREADS=1
2022-10-09 11:29:56,911 - __main__ - INFO - TF_NUM_INTRAOP_THREADS=96
2022-10-09 11:29:56,911 - __main__ - INFO - TF_ENABLE_ONEDNN_OPTS=1
2022-10-09 11:29:56,911 - __main__ - INFO - ITEX_LAYOUT_OPT=0
2022-10-09 11:29:56,911 - __main__ - INFO - numactl --localalloc -C 0-95 <VIRTUAL_ENV>/bin/python -u infer_resnet50.py 2>&1 | tee ./logs/run_20221009112956_instance_0_cores_0-95.log
```
