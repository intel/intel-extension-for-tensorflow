# Frequently Asked Questions

1. How to check whether GPU drivers are installed successfully?

Run `import tensorflow` and it will show which platform you are running on: Intel Level-Zero(default) or Intel OpenCL.

And the high level API of TensorFlow `tf.config.experimental.list_physical_devices()` will tell you the device types that are registered to TensorFlow core.

```
$ python
>>> import tensorflow as tf
2021-07-01 06:40:55.510076: I itex/core/devices/gpu/dpcpp_runtime.cc:116] Selected platform: Intel(R) Level-Zero.
>>> tf.config.experimental.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')]
```

   
2. How to know the configurations and rate of utilization of local GPU devices?

[System Monitoring Utility](https://github.com/intel/pti-gpu/tree/master/tools/sysmon) tool can be used to show the capability (clock frequency, EU count, amount of device memory, and so on) of your devices and usage of each sub-module (device memory, GPU engines, and so on).


3. What's the relationship of `TensorFlow*`, `Intel® Optimization of TensorFlow*` and `Intel® Extension for TensorFlow*`?

`Intel® Optimization of TensorFlow*` is designed to optimize for Intel CPU. It could replace `stock TensorFlow*` (`Google TensorFlow*`) for Intel CPU. All Intel optimizations are available in both `Intel® Optimization for TensorFlow*` and `stock TensorFlow*` (since 2.9) for Intel CPU. That means you only need to install one of them. **DO NOT** install them in same time, the impact is unknown.

`Intel® Extension for TensorFlow*` is an extension of `stock TensorFlow*` and help extend to accelerate on Intel CPU or support Intel GPU.
`Intel® Extension for TensorFlow*` only co-works with `stock TensorFlow*`. Please **DO NOT** install Intel® Extension for TensorFlow* with Intel® Optimization for TensorFlow*.

Starting in Q1 2024, Intel® Optimization for TensorFlow* will be discontinued. Intel optimization will be available only through `Intel® Extension for TensorFlow*` co-works with the `stock TensorFlow*`. Intel will continue to upstream advanced optimization to `stock TensorFlow*` in the future.

Currently, Intel® Extension for TensorFlow* has two releases: CPU & GPU.

For Intel CPU, `Intel® Extension for TensorFlow* for CPU` + `stock TensorFlow*` could replace `Intel® Optimization of TensorFlow*`. Install command: `pip install --upgrade intel-extension-for-tensorflow[cpu]`.

For Intel GPU, `Intel® Extension for TensorFlow* for GPU` + `stock TensorFlow*` is only way to make TensorFlow* support Intel GPU. Install command: `pip install --upgrade intel-extension-for-tensorflow[gpu]`.

## Troubleshooting

This section shows common problems and solutions for compilation and runtime issues you may encounter.



### Build from source

| Error                                                        | Solution                     | Comments                                                 |
| ------------------------------------------------------------ | ---------------------------- | -------------------------------------------------------- |
| external/onednn/src/sycl/level_zero_utils.cpp:33:10: fatal error: 'level_zero/ze_api.h' file not found<br/>#include <level_zero/ze_api.h><br/>         ^~~~~~~~~~~~~~~~~~~~~ | install `level-zero-dev` lib | `level-zero-dev` lib is needed when building from source |



### Runtime

| Error                                                        | Solution                              | Comments                            |
| ------------------------------------------------------------ | ------------------------------------- | ----------------------------------- |
| ModuleNotFoundError: No module named 'tensorflow'            | install TensorFlow                    | Intel® Extension for TensorFlow* depends on TensorFlow          |
| tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory | `source /opt/intel/oneapi/setvars.sh` | set env vars of oneAPI Base Toolkit |
| version GLIBCXX_3.4.30' not found | `conda install -c conda-forge gxx_linux-64==12.1.0` | install higher version glibcxx |  
