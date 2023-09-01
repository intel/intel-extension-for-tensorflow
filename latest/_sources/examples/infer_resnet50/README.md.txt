# ResNet50 Inference on Intel CPU and GPU

## Prerequisites

### Prepare for GPU (Skip this step for CPU)

Refer to [Prepare](../common_guide_running.html#prepare)

### Setup Running Environment

* Setup for GPU
```bash
./set_env_gpu.sh
```

* Setup for CPU
```bash
./set_env_cpu.sh
```

### Enable Running Environment

* For GPU, refer to [Running](../common_guide_running.html#running)

* For CPU, 
```bash
source env_itex_cpu/bin/activate
```

## Executes the Example with Python API
If `intel-extension-for-tensorflow[cpu]` is installed, it will be executed on the CPU automatically, while if `intel-extension-for-tensorflow[gpu]` is installed, GPU will be the backend.
```
python infer_resnet50.py
```

## Example Output
With successful execution, it will print out the following results:

```
...

[('n02123159', 'tiger_cat', 0.22355853)]
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.
``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
