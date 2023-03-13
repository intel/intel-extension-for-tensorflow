# Speed up Inference of Inception v4 by Advanced Automatic Mixed Precision on Intel CPU and GPU

## Introduction
Advanced Automatic Mixed Precision (Advanced AMP) uses lower-precision data types (such as float16 or bfloat16) to make model run with 16-bit and 32-bit mixed floating-point types during training and inference to make it run faster with less memory consumption in CPU and GPU.

For detailed info, please refer to [Advanced Automatic Mixed Precision](../../docs/guide/advanced_auto_mixed_precision.md)

This example shows the acceleration of inference by Advanced AMP on Intel CPU or GPU.

In this example, we will test and compare the performance of FP32 and Advanced AMP (mix BF16/FP16 and FP32) on Intel CPU or GPU.


## Step

1. Download the Inception v4 model from internet.
2. Test the performance of original model (FP32) on Intel CPU or GPU.
2. Test the performance of original model by Advanced AMP (BF16 or FP16) on Intel CPU or GPU.
3. Compare the latency and throughputs of above two cases; print the result.

Users need to indicate the CPU or GPU as the backend, and choose Advanced AMP data type from BF16 and FP16 based on the requirements and hardware support.


## Hardware Requirement

Advanced AMP supports two 16 bit floating-point types: BF16 and FP16.

|Data Type|GPU|CPU|
|-|-|-|
|BF16|Intel® Data Center GPU Flex Series 170<br>Needs to be checked for your Intel GPU|Intel® 4th Generation Intel® Xeon® Scalable Processor (Sapphire Rapids)|
|FP16|Intel® Data Center GPU Flex Series 170<br>Supported by most of Intel GPU||


This example supports both types. Set the parameter according to the requirement and hardware support.

### Prepare for GPU

Refer to [Prepare](../common_guide_running.md##Prepare)

### Setup Running Environment

```
./set_env_cpu.sh

or

./set_env_gpu.sh
```

### Enable Running Environment

* For GPU, refer to [Running](../common_guide_running.md##Running)

* For CPU,
```
source env_itex/bin/activate
```

### Enable Advanced AMP Method

There are two methods to enable Advanced AMP based on Intel® Extension for TensorFlow*: Python API & Environment Variable Configuration.

1. Python API

Add code in the beginning of python code:

For BF16:
```
import intel_extension_for_tensorflow as itex


auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
auto_mixed_precision_options.data_type = itex.BFLOAT16


graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options)
graph_options.auto_mixed_precision = itex.ON

config = itex.ConfigProto(graph_options=graph_options)
itex.set_config(config)
```

For FP16, modify one line above:
```
auto_mixed_precision_options.data_type = itex.BFLOAT16
->
auto_mixed_precision_options.data_type = itex.FLOAT16
```


2. Environment Variable Configuration

Execute commands in bash:

```
export ITEX_AUTO_MIXED_PRECISION=1

export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE=BFLOAT16
#export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE=FLOAT16
```
For FP16, modify one line above:
```
export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE=BFLOAT16
->
export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE=FLOAT16
```

## Download Model

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/inceptionv4_fp32_pretrained_model.pb
```

## Execute Testing and Comparing the Performance of FP32 and Advanced AMP on CPU and GPU

The example supports both by two scripts:
- Use Python API : **infer_fp32_vs_amp.py**
- Use Environment Variable Configuration: **infer_fp32_vs_amp.sh**


### Python API

Run with CPU and BF16 data type:
```
python infer_fp32_vs_amp.py cpu bf16
```

Run with GPU and BF16 data type:
```
python infer_fp32_vs_amp.py gpu bf16
```

Run with GPU and FP16 data type:
```
python infer_fp32_vs_amp.py gpu fp16
```

### Environment Variable Configuration

Run with CPU and BF16 data type:
```
./infer_fp32_vs_amp.sh cpu bf16
```

Run with GPU and BF16 data type:
```
./infer_fp32_vs_amp.sh gpu bf16
```

Run with GPU and FP16 data type:
```
./infer_fp32_vs_amp.sh gpu fp16
```

### Result
All cases above will output result in screen like:

```
Compare Result
Model                           FP32                    BF16
Latency (s)                     X.01837550401687622     X.0113076031208038
Throughputs (FPS) BS=128        Y.92880015134813        Y.1691980294577

Model                           FP32                    BF16
Latency Normalized              1                       X.6153628825864496
Throughputs Normalized          1                       X.867908472383153
```

**Note, if the data type (BF16, FP16) is not supported by the hardware, the training will be executed by converting to FP32. That will make the performance is worse than FP32 case.**

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.
``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
