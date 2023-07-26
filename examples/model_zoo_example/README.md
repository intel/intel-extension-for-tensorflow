# Accelerating Deep Learning Inference for Model Zoo Workloads on Intel CPU and GPU

## Introduction
This example shows the guideline to run Model Zoo workloads on Intel CPU and GPU with the optimizations from Intel® Extension for TensorFlow*, without any model code changes.

## Prerequisites
For Intel CPU, refer to [Intel CPU software installation](../../docs/install/install_for_cpu.md#intel-cpu-software-installation).
For Intel GPU, refer to [Intel XPU software installation](../../docs/install/install_for_xpu.md#intel-gpu-software-installation).

## Execute

### Prepare the Codes
```bash
git clone https://github.com/IntelAI/models
cd models
git checkout v2.8.0
```

### Sample Use cases


|Model|Mode|Model Documentation|
|-|-|-|
|Inception V3|Inference|[FP32](https://github.com/IntelAI/models/blob/v2.8.0/benchmarks/image_recognition/tensorflow/inceptionv3/inference/fp32/README.md)  [INT8](https://github.com/IntelAI/models/blob/v2.8.0/benchmarks/image_recognition/tensorflow/inceptionv3/inference/int8/README.md)|
|Inception V4|Inference|[FP32](https://github.com/IntelAI/models/blob/v2.8.0/benchmarks/image_recognition/tensorflow/inceptionv4/inference/fp32/README.md)  [INT8](https://github.com/IntelAI/models/blob/v2.8.0/benchmarks/image_recognition/tensorflow/inceptionv4/inference/int8/README.md)|
|ResNet50 V1.5|Inference|[FP32](https://github.com/IntelAI/models/blob/v2.8.0/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/fp32/README.md)  [INT8](https://github.com/IntelAI/models/blob/v2.8.0/benchmarks/image_recognition/tensorflow/resnet50v1_5/inference/int8/README.md)|

### Performance Optimization
- FP16/BF16 INT8 Inference Optimization  

  Refer to the above FP32 model documentation, and only set one extra environment variable to enable advanced auto mixed precision Graph optimization before running inference.
  ```bash
  export ITEX_AUTO_MIXED_PRECISION=1
  ```
- INT8 Inference Optimization
  
  To avoid memory copy on GPU, we provide a tool to convert the const to host const for INT8 pretrained-models.

  Take the ResNet50 v1.5 INT8 pb for example,
  ```

  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/resnet50v1_5_int8_pretrained_model.pb

  python host_const.py -i <path to the frozen graph downloaded above>/resnet50v1_5_int8_pretrained_model.pb -b -o <path to save the converted frozen graph>/resnet50v1_5_int8_pretrained_model-hostconst.pb
  ```
  Use the new INT8 pb for INT8 inference, After converting to the new INT8 pb.

## FAQ
1.  During the Inception V3 INT8 batch inference, if running with real data, you might encounter a message "Running out of images from dataset". It is a known issue of Model Zoo script.

Solution: 

- Option 1: Please use dummy data instead. 

- Option 2: If you want to run inference with real data, use the command below. And comment the last line of below int8_batch_inference.sh script to unspecify the warmup_steps and steps.  
```bash
cd models
vi ./quickstart/image_recognition/tensorflow/inceptionv3/inference/cpu/int8/int8_batch_inference.sh
```
