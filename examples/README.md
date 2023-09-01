# Examples

## Prepare for Running

Before running the training/inference code based on Intel® Extension for TensorFlow*, there are several prepare steps to be executed. Please refer to [Common Guide for Running](./common_guide_running.md).

## Examples

A wide variety of examples are provided to demonstrate the usage of Intel® Extension for TensorFlow*.

|Name|Description|Hardware|
|-|-|-|
|[Quick Example](quick_example.md)|Quick example to verify Intel® Extension for TensorFlow* and running environment.|CPU & GPU|
|[ResNet50 Inference](./infer_resnet50/README.md)|ResNet50 inference on Intel CPU or GPU without code changes.|CPU & GPU|
|[BERT Training for Classifying Text](./train_bert/README.md)|BERT training with Intel® Extension for TensorFlow* on Intel CPU or GPU.<br>Use the TensorFlow official example without code change.|CPU & GPU|
|[Speed up Inference of Inception v4 by Advanced Automatic Mixed Precision via Docker Container or Bare Metal](./infer_inception_v4_amp/README.md)|Test and compare the performance of inference with FP32 and Advanced Automatic Mixed Precision (AMP) (mix BF16/FP16 and FP32).<br>Shows the acceleration of inference by Advanced AMP on Intel CPU and GPU via Docker Container or Bare Metal.|CPU & GPU|
|[Accelerate AlexNet by Quantization with Intel® Extension for TensorFlow*](./accelerate_alexnet_by_quantization/README.md)| An end-to-end example to show a pipeline to build up a CNN model to <br>recognize handwriting number and speed up AI model with quantization <br>by Intel® Neural Compressor and Intel® Extension for TensorFlow* on Intel GPU.|GPU|
|[Accelerate Deep Learning Training and Inference for Model Zoo Workloads on Intel GPU](./model_zoo_example/README.md)|Examples on running Model Zoo workloads on Intel GPU with the optimizations from Intel® Extension for TensorFlow*.|GPU|
|[Quantize Inception V3 by Intel® Extension for TensorFlow* on Intel® Xeon®](./quantize_inception_v3/README.md)|An end-to-end example to show how Intel® Extension for TensorFlow* provides quantization feature by cooperating with Intel® Neural Compressor and oneDNN Graph. It will provide better quantization: better performance and accuracy loss is in controlled.|CPU|
|[Mnist training with Intel® Optimization for Horovod*](./train_horovod/mnist/README.md)|Mnist distributed training example on Intel GPU. |GPU|
|[ResNet50 training with Intel® Optimization for Horovod*](./train_horovod/resnet50/README.md)|ResNet50 distributed training example on Intel GPU. |GPU|
|[Stable Diffusion Inference for Text2Image on Intel GPU](./stable_diffussion_inference/README.md)|Example for running Stable Diffusion Text2Image inference on Intel GPU with the optimizations from Intel® Extension for TensorFlow*.|GPU|
|[Accelerate ResNet50 Training by XPUAutoShard on Intel GPU](./train_resnet50_with_autoshard/README.md)|Example on running ResNet50 training on Intel GPU with the XPUAutoShard feature.|GPU|
|[Accelerate BERT-Large Pretraining on Intel GPU](./pretrain_bert/README.md)|Example on running BERT-Large pretraining on Intel GPU with the optimizations from Intel® Extension for TensorFlow*.|GPU|
|[Accelerate Mask R-CNN Training w/o horovod on Intel GPU](./train_maskrcnn/README.md)|Example on running Mask R-CNN training on Intel GPU with the optimizations from Intel® Extension for TensorFlow*.|GPU|
|[Accelerate 3D-UNet Training w/o horovod for medical image segmentation on Intel GPU](./train_3d_unet/README.md)|Example on running 3D-UNet training for medical image segmentation on Intel GPU with the optimizations from Intel® Extension for TensorFlow*.|GPU|
