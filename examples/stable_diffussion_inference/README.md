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

### Environment Vasriable
`export TF_USE_LEGACY_KERAS=1`

### Model Code change
We optimized official keras-cv Stable Diffusion, for example, concatenate two forward passes, combine computation in loops to reduce op number, and add fp16 mode for model. However, this optimization hasn't been up streamed. To get better performance, instead of installing official keras-cv, you may want to clone keras-cv, apply patch, then install it as shown here:
```
git clone https://github.com/keras-team/keras-cv.git
cd keras-cv
git reset --hard 66fa74b6a2a0bb1e563ae8bce66496b118b95200
git apply patch
pip install .
```

### Prepare for GPU (Skip this step for CPU)

Refer to [Prepare](../common_guide_running.md#prepare)

### Setup Running Environment


* Setup for GPU
```bash
./pip_set_env.sh
```

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md#running)

### Running the Jupyter Notebook
   * Add kernel for env_itex environment:
 ```
 python3 -m ipykernel install --name env_itex --user
  ```
   * Change to the sample directory.
   * Launch Jupyter Notebook.
 ```
 jupyter notebook --no-browser --port=8888 
  ```
   * Follow the instructions to open the URL with the token in your browser.
   * Locate and select the Notebook.
 ```
 stable_diffussion_inference.ipynb
  ```
   * Run every cell in the Notebook in sequence.

### Executes the Example with Python API
#### FP32 Inference
```
python stable_diffusion_inference.py --precision fp32
```

#### FP16 Inference
```
python stable_diffusion_inference.py --precision fp16
```

#### Accuracy
Note: At present, we evaluate accuracy by calculating the Fréchet inception distance (FID) between FP16 outcomes of 7 images on XPU and FP16 outcomes on NVIDIA A100. This may change with subsequent releases.
```shell
python stable_diffusion_accuracy.py --precision fp16 \
  --load_ref_result --ref_result_dir "./nv_results/img_arrays_for_acc.txt"
```

## Example Output
With successful execution, it will print out the following results:

```
latency 81.1146879196167 ms, throughput 12.328223477737884 it/s
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.
``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
