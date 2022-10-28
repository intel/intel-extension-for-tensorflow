#  Accelerate AlexNet by Quantization with Intel® Extenstion for Tensorflow*

## Background

Low-precision inference can speed up inference obviously, by converting the FP32 model to INT8 or BF16 model. Intel provides hardware technology to accelerate the low precision model on Intel CPU & GPU:

1. Intel® Deep Learning Boost: It is present in the Second Generation Intel® Xeon® Scalable Processors and newer Xeon®, which supports to speed up INT8 and BF16 model by hardware.

2. Intel GPU which supports INT8.

Intel® Neural Compressor helps the user to simplify the processing to convert the FP32 model to INT8.

At the same time, Intel® Neural Compressor will tune the quantization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

Intel® Neural Compressor is released in Intel® AI Analytics Toolkit and works with Intel® Optimization of TensorFlow*.

Please refer to the official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)

## Introduction

By Intel® Extenstion for Tensorflow*, it's easy to quantize FP32 model to INT8 model and be accelerated on Intel CPU and GPU.

The example reuses the existed End-To-End example: [Intel® Neural Compressor Sample for TensorFlow](https://github.com/intel/neural-compressor/tree/master/examples/notebook/tensorflow/alexnet_mnist) provided by Intel® Neural Compressor, to show a pipeline to build up a CNN model to recognize handwriting number and speed up AI model with quantization by Intel® Neural Compressor.

The original example is designed to run on Intel CPU. After installing Intel® Extenstion for Tensorflow*, it could run on Intel CPU and GPU.

All steps follow this existed example. **There is no any code to be changed.**

Please read the example guide for detailed information.

We will learn the acceleration of AI inference by Intel AI technology:

1. Intel® Deep Learning Boost on CPU

2. Intel GPU which supports INT8

3. Intel® Neural Compressor

4. Intel® Extenstion for Tensorflow*

## Hardware Environment

The example can run on Intel CPU & GPU by Intel® Extenstion for Tensorflow*.

### CPU

This demo is recommended to use 2nd Generation Intel® Xeon® Scalable Processors or newer, which include:

1. Intel® AVX512 instruction to speed up training & inference AI model.

2. Intel® Deep Learning Boost: Vector Neural Network Instruction (VNNI) to accelerate AI/DL Inference with INT8/BF16 Model.

With Intel® Deep Learning Boost, the performance will be increased obviously. Without it, maybe it's 1.x times of FP32.


#### Intel® DevCloud

If you have no such CPU support Intel® Deep Learning Boost, you could register to Intel® DevCloud and try this example on new Xeon with Intel® Deep Learning Boost freely. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)

### GPU

Support: Intel® Data Center Flex Series GPU.

For local server, please install the GPU driver and oneAPI packages by refer to [Intel GPU Software Installation](/docs/install/install_for_gpu.md).

For Intel® DevCloud, the GPU driver and oneAPI packages are already installed.

## Running Environment

### Set up Base Running Environment

Please refer to the example: [Intel® Neural Compressor Sample for TensorFlow](https://github.com/intel/neural-compressor/tree/master/examples/notebook/tensorflow/alexnet_mnist) to setup running environment.

There are new requirements:

1. Python should be 3.9 or newer version.

2. Tensorflow should be 2.10.0 or newer version.

### Set up Intel® Extenstion for Tensorflow*

Please install Intel® Extenstion for Tensorflow* in the running envrionment:

1. CPU

```
python -m pip install --upgrade intel-extension-for-tensorflow[cpu]

```

2. GPU

```
python -m pip install --upgrade intel-extension-for-tensorflow[gpu]

```

## Execute

Please refer to the example: [Intel® Neural Compressor Sample for TensorFlow](https://github.com/intel/neural-compressor/tree/master/examples/notebook/tensorflow/alexnet_mnist) to execute the sample code and check the result.
