# Frequently Asked Questions

I. How to check whether GPU drivers are installed successfully?

Run `import tensorflow` and it will show which platform you are running on: Intel(R) Level-Zero(default) or Intel(R) OpenCL.

And the high level API of TensorFlow `tf.config.experimental.list_physical_devices()` will tell you the device types that are registered to TensorFlow core.

```
$ python
>>> import tensorflow as tf
2021-07-01 06:40:55.510076: I itex/core/devices/gpu/dpcpp_runtime.cc:116] Selected platform: Intel(R) Level-Zero.
>>> tf.config.experimental.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')]
```

   
II. How to know the configurations and rate of utilization of local GPU devices?

[System Monitoring Utility](https://github.com/intel/pti-gpu/tree/master/tools/sysmon) tool can be used to show the capability (clock frequency, EU count, amount of device memory and so on) of your devices and usage of each sub-module (device memory, GPU engines and so on).



## Troubleshooting

This section shows common problems and solutions for compilation and runtime issues you may encounter.



### Build from source

| Error                                                        | Solution                     | Comments                                                 |
| ------------------------------------------------------------ | ---------------------------- | -------------------------------------------------------- |
| external/onednn/src/sycl/level_zero_utils.cpp:33:10: fatal error: 'level_zero/ze_api.h' file not found<br/>#include <level_zero/ze_api.h><br/>         ^~~~~~~~~~~~~~~~~~~~~ | install `level-zero-dev` lib | `level-zero-dev` lib is needed when building from source |



### Runtime

| Error                                                        | Solution                                                     | Comments                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| ModuleNotFoundError: No module named 'tensorflow'            | install TensorFlow                                           | ITEX depends on TensorFlow                                |
| tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory | `source /opt/intel/oneapi/setvars.sh`                        | set env vars of oneAPI Base Toolkit                       |
| INTEL MKL ERROR: /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx512.so.2: undefined symbol: mkl_sparse_optimize_bsr_trsm_i8.<br/>Intel MKL FATAL ERROR: Cannot load libmkl_avx512.so.2 or libmkl_def.so.2. | `export LD_PRELOAD=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so:/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_tbb_thread.so` | It's a known issue and will be fixed in the next release. |

