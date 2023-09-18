# Frequently Asked Questions

1. **How do I check that GPU drivers are installed successfully?**

    Run `import tensorflow` and it will show which platform you are running on: Intel® oneAPI Level-Zero (default) or OpenCL™.

    The high level API of TensorFlow `tf.config.experimental.list_physical_devices()` will tell you the device types that are registered to TensorFlow core.

    ```
    $ python
    >>> import tensorflow as tf
    2021-07-01 06:40:55.510076: I itex/core/devices/gpu/dpcpp_runtime.cc:116] Selected platform: Intel(R) Level-Zero.
    >>> tf.config.experimental.list_physical_devices()
    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')]
    ```

2. **How can I see the configurations and rate of utilization of local GPU devices?**

    Use the [System Monitoring Utility](https://github.com/intel/pti-gpu/tree/master/tools/sysmon) tool to show the capability (clock frequency, EU count, amount of device memory, and so on) of your devices and usage of each sub-module (device memory, GPU engines, and so on).


3. **What's the relationship of TensorFlow\*, Intel® Optimization of TensorFlow\* and Intel® Extension for TensorFlow\*?**

    - **TensorFlow** is an open-source machine learning library developed and maintained by Google. It is widely used for building and training machine learning models, particularly neural networks.<p/>

    - **Intel® Optimization of TensorFlow** is an optimized library to run TensorFlow on Intel CPUs and replaces stock TensorFlow\* for Intel CPUs. Since the TensorFlow 2.9 release, all Intel optimizations for Intel CPUs are upstreamed and available in stock TensorFlow. That means you only need to install stock TensorFlow. **DO NOT** install both at the same time, the impact is unknown.

      Starting in Q1 2024, the separate Intel® Optimization for TensorFlow* will be discontinued. Intel optimization will be available directly from continuing upstreamed contributions to stock TensorFlow*.

    - **Intel® Extension for TensorFlow** is an extension of stock TensorFlow* and helps extend acceleration on Intel CPUs and supported Intel GPUs.
      Intel® Extension for TensorFlow* co-works with stock TensorFlow* (that
      includes upstreamed optimizations from Intel).

      Currently, Intel® Extension for TensorFlow* has two releases: CPU & GPU.

      - For Intel CPUs, Intel® Extension for TensorFlow* for CPU + stock TensorFlow\* provides the best performance of TensorFlow\* on Intel CPUs. Install command: `pip install --upgrade intel-extension-for-tensorflow[cpu]`.

      - For Intel GPUs, Intel® Extension for TensorFlow* for GPU + stock TensorFlow\* provides the best performance of TensorFlow* on Intel GPUs. Install command: `pip install --upgrade intel-extension-for-tensorflow[gpu]`.

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



