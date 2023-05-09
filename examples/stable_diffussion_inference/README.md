# Stable Diffusion Inference for Text2Image on Intel GPU

## Introduction

Intel® Extension for TensorFlow* is compatible with stock TensorFlow*. 
This example shows Stable Diffusion Inference for Text2Image.

Install the Intel® Extension for TensorFlow* in legacy running environment, Tensorflow will execute the Inference on Intel GPU.

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Max Series
 - Intel® Data Center GPU Flex Series 170
 
## Prerequisites

### Model Code change
We optimized official keras-cv Stable Diffusion, for example, concatenate two forward passes, combine computation in loops to reduce op number, and add fp16 mode for model. However, this optimization hasn't been up streamed. To get better performance, instead of installing official keras-cv, you may want to clone keras-cv, apply patch, then install it as shown here:
```
git clone https://github.com/keras-team/keras-cv.git
cd keras-cv
git reset --hard f2de9510460fe8368c23df8fd168827c02c45679
git apply patch
pip install .
```

### Prepare for GPU (Skip this step for CPU)

Refer to [Prepare](../common_guide_running.md##Prepare)

### Setup Running Environment


* Setup for GPU
```bash
./pip_set_env.sh
```

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md##Running)


### Executes the Example with Python API
#### FP32 Inference
```
python stable_diffusion_inference.py --precision fp32
```

#### FP16 Inference
```
python stable_diffusion_inference.py --precision fp16
```


## Example Output
With successful execution, it will print out the following results:

```
50/50 [==============================] - 4s 88ms/step
Start plotting the generated images to fp16_1imgs_50steps.png
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.
``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
