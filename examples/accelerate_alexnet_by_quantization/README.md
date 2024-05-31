#  Accelerate AlexNet by Quantization with Intel® Extension for Tensorflow*

## Background

Low-precision inference can speed up inference, by converting the FP32 model to INT8 or BF16 model. Intel provides hardware technology to accelerate the low precision model on Intel GPUs that supports INT8.

Intel® Neural Compressor helps you simplify converting the FP32 model to INT8.

At the same time, Intel® Neural Compressor will tune the quantization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

Intel® Neural Compressor is released in Intel® AI Analytics Toolkit and works with Intel® Optimization of TensorFlow*.

Refer to the official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)

## Introduction

With Intel® Extension for Tensorflow*, it's easy to quantize FP32 model to INT8 model and be accelerated on Intel CPUs and GPUs.

This example reuses the existing End-To-End example: [Intel® Neural Compressor Sample for TensorFlow](https://github.com/intel/neural-compressor/tree/master/examples/notebook/tensorflow/alexnet_mnist) provided by Intel® Neural Compressor, to show a pipeline to build up a CNN model to recognize handwriting number and speed up AI model with quantization by Intel® Neural Compressor.

The original example is designed to run on Intel CPU with Stock Tensorflow* or Intel® Optimization for Tensorflow*. After installing Intel® Extension for Tensorflow*, it could run on Intel GPU.

All steps follow this existed example. **There is no any code to be changed.**

Read the example guide for detailed information.

We will learn the acceleration of AI inference by Intel AI technology:

1. Intel GPU that supports INT8

2. Intel® Neural Compressor

3. Intel® Extension for Tensorflow*

## Hardware Environment

The example can run on Intel GPU by Intel® Extension for Tensorflow*.

### GPU

Support: Intel® Data Center Flex Series GPU.

#### Local Server

Install the GPU driver and oneAPI packages by referring to [Intel GPU Software Installation](/docs/install/install_for_xpu.md).

#### Intel® DevCloud

If you have no such Intel GPU support INT8, you could register to Intel® DevCloud for oneAPI applications and try this example on the compute node with Intel GPU. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html).

In Intel® DevCloud, the GPU driver and oneAPI packages are already installed.

## Running Environment

### Set up Base Running Environment

Please refer to the example: [Intel® Neural Compressor Sample for TensorFlow](https://github.com/intel/neural-compressor/tree/master/examples/notebook/tensorflow/alexnet_mnist) to setup running environment.

There are new requirements:

1. Python should be 3.9 or newer version.

2. TensorFlow should be 2.15.1

### Set up Intel® Extension for Tensorflow* for GPU

Install Intel® Extension for Tensorflow* in the running environment:

```
python -m pip install --upgrade intel-extension-for-tensorflow[xpu]

```

## Execute

Please refer to the example: [Intel® Neural Compressor Sample for TensorFlow](https://github.com/intel/neural-compressor/tree/master/examples/notebook/tensorflow/alexnet_mnist) to execute the sample code and check the result.
