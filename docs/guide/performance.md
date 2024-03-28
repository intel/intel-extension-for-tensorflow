# Performance Data

- [Overview](#overview)
- [Models](#models)
  - [Training Workloads](#training-workloads)
  - [Inference Workloads](#inference-workloads)
- [Training Accuracy Results](#training-accuracy-results)
  - [Training Accuracy on 1-node of 4x Intel Data Center GPU Max 1550](#training-accuracy-on-1-node-of-4x-intel-data-center-gpu-max-1550)
- [Training Performance Results](#training-performance-results)
  - [Training Performance on 1-node of 4x Intel Data Center GPU Max 1550](#training-performance-on-1-node-of-4x-intel-data-center-gpu-max-1550)
    - [ResNet50v1-5 Training Performance Results](#resnet50v1-5-training-performance-results)
    - [BERT-Large Phase2 Training Performance Results](#bert-large-phase2-training-performance-results)
    - [Mask-RCNN Training Performance Results](#mask-rcnn-training-performance-results)
    - [Medical Image 3D U-Net Training Performance Results](#medical-image-3d-u-net-training-performance-results)
- [Inference Performance Results](#inference-performance-results)
  - [Inference Performance on 1x Intel Data Center GPU Flex 170](#inference-performance-on-1x-intel-data-center-gpu-flex-170)
    - [ResNet50v1-5 Inference Performance Results](#resnet50v1-5-inference-performance-results)
    - [EfficientNet-B0 Inference Performance Results](#efficientnet-b0-inference-performance-results)
    - [EfficientNet-B3 Inference Performance Results](#efficientnet-b3-inference-performance-results)
    - [Mask-RCNN Inference Performance Results](#mask-rcnn-inference-performance-results)
    - [Stable Diffusion v1-4 Inference Performance Results](#stable-diffusion-v1-4-inference-performance-results)
- [Configuration](#configuration)
  - [Software Configuration](#software-configuration)
    - [Software Configuration for Intel Max 1550 GPU](#software-configuration-for-intel-max-1550-gpu)
    - [Software Configuration for Intel Flex 170 GPU](#software-configuration-for-intel-flex-170-gpu)
  - [Hardware Configuration](#hardware-configuration)
    - [Hardware Configuration for Intel Max 1550 GPU](#hardware-configuration-for-intel-max-1550-gpu)
    - [Hardware Configuration for Intel Flex 170 GPU](#hardware-configuration-for-intel-flex-170-gpu)
- [Additional Performance Data for Intel AI Data Center Products](#additional-performance-data-for-intel-ai-data-center-products)

## Overview

This document demonstrates the training and inference performance as well as accuracy results on several popular AI workloads with Intel® Extension for TensorFlow\* benchmarked on Intel GPUs. You can easily reproduce these results following the guidlines in [examples](../../examples/README.md).


## Models

The following tables provide the links where you can get the original code repository and step-by-step guide running on Intel GPUs for each model.

### Training Workloads

|Model|Original Model Repo|ITEX Step-by-Step Guide|
|-|-|-|
|ResNet50v1.5|[TensorFlow-Models/ResNet50v1.5](https://github.com/tensorflow/models/tree/v2.14.0/official/legacy/image_classification/)|[Resnet50 train on Intel GPU](../../examples/train_resnet50/README.md)|
|BERT-Large|[DeepLearningExamples/BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT/)|[Accelerate BERT-Large Pretraining on Intel GPU](../../examples/pretrain_bert/README.md)
|Mask-RCNN|[DeepLearningExamples/Mask-RCNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN/)|[Accelerate Mask R-CNN Training on Intel GPU](../../examples/train_maskrcnn/README.md)|
|3D-UNet|[DeepLearningExamples/3D-UNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical/)|[Accelerate 3D-UNet Training for medical image segmentation on Intel GPU](../../examples/train_3d_unet/README.md)|

### Inference Workloads

|Model|Original Model Repo|ITEX Step-by-Step Guide|
|-|-|-|
|ResNet50v1.5|[Intel-Reference-Models/ResNet50v1.5](https://github.com/IntelAI/models/tree/v3.1.0/models_v2/tensorflow/resnet50v1_5/inference/gpu/)|[ResNet50v1.5 Model Inference with Intel® Extention for TensorFlow\*](https://github.com/IntelAI/models/tree/v3.1.0/models_v2/tensorflow/resnet50v1_5/inference/gpu/)|
|EfficientNet-B0|[Keras-Applications/EfficientNet](https://keras.io/api/applications/efficientnet/)|Use the exact same codes and instructions as in the orignal model repo|
|EfficientNet-B3|[Keras-Applications/EfficientNet](https://keras.io/api/applications/efficientnet/)|Use the exact same codes and instructions as in the orignal model repo|
|Mask-RCNN|[DeepLearningExamples/Mask-RCNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN/)|Use the exact same codes and instructions as in the orignal model repo|
|Stable Diffusion v1-4|[KerasCV/Stable-Diffusion](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion)|[Stable Diffusion Inference for Text2Image on Intel GPU](../../examples/stable_diffussion_inference/README.md)|


## Training Accuracy Results

### Training Accuracy on 1-node of 4x Intel Data Center GPU Max 1550

The following table shows the BERT-Large performance, training loss and time-to-train (TTT) results for both the pre-training and fine-tuning phases on 1-node of 4x Intel® Data Center GPU Max 1550 (600W OAM, 2-stack for each GPU).

||Pre-training Phase1|Pre-training Phase2|Fine-Tuning|
|-|-|-|-|
|**Dataset**|[Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](https://yknzhu.wixsite.com/mbweb/)|[Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](https://yknzhu.wixsite.com/mbweb/)|[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 1.1|
|**Maximum Sequence Length**|128|512|384|
|**Data Type**|BF16|BF16|BF16|
|**Throughput (sequences/sec)**|3265.35|699.25|523.55|
|**Time to Train (hours)**|39.32|20.40|0.67|
|**Loss**|1.6047|1.3870|0.6867|


## Training Performance Results

### Training Performance on 1-node of 4x Intel Data Center GPU Max 1550

The following tables show the performance numbers for several popular training workloads on 1-node of 4x Intel® Data Center GPU Max 1550 (600W OAM, 2-stack for each GPU). For these workloads, we enable and benchmark both FP32 training and BF16 automatic mixed precision (AMP) training with 1-Stack of 1x Max 1550, 2-Stack of 1x Max 1550 as well as 4x Max 1550 (with 8 Stacks in total), to showcase the performance boost and scalability with Intel® Extension for TensorFlow\* and Intel® Optimization for Horovod\*. 

> **Note**: The training performance result on each workload below for `1x Max 1550 w/ 1-Stack` represents the minimum value of the performance results on 2 stacks of single GPU, with 2 instances initiated simultaneously, while each stack of the GPU executing the workload separately, without distributed training.

#### ResNet50v1-5 Training Performance Results

|GPUs|Ranks|Local Batch Size: <br>FP32, BF16|Training <br>Steps|Throughput w/ <br>TF32 (images/sec)|Throughput w/ <br>BF16 (images/sec)|Throughput Speedup <br>w/ AMP|Weak Scaling <br>w/ TF32|Weak Scaling <br>w/ BF16|
|-|-|-|-|-|-|-|-|-|
|1x Max 1550 w/ 1-Stack|1|256, 512|5000|918.96|1766.53|1.92x|1.00|1.00|
|1x Max 1550 w/ 2-Stack|2|256, 512|5000|1762.76|3461.86|1.96x|1.92|1.96|
|4x Max 1550|8|256, 256|5000|NA|12278.32|NA|NA|6.95|

#### BERT-Large Phase2 Training Performance Results

|GPUs|Ranks|Local <br>Batch Size <br>x Accumulation Steps|Training <br>Steps|Throughput <br> w/ TF32 <br>(sequences/sec)|Throughput <br>w/ BF16 <br>(sequences/sec)|Throughput Speedup <br>w/ AMP|Weak Scaling <br>w/ TF32|Weak Scaling <br>w/ BF16|
|-|-|-|-|-|-|-|-|-|
|1x Max 1550 w/ 1-Stack|1|32 x 30|20|36.22|93.22|2.57x|1.00|1.00|
|1x Max 1550 w/ 2-Stack|2|32 x 30|20|74.40|182.57|2.45x|2.05|1.96|
|4x Max 1550|8|32 x 30|20|NA|692.11|NA|NA|7.42|

#### Mask-RCNN Training Performance Results

|GPUs|Ranks|Local Batch Size|Training Steps|Throughput w/ BF16 (images/sec)|Weak Scaling w/ BF16|
|-|-|-|-|-|-|
|1x Max 1550 w/ 1-Stack|1|4|20|29.03|1.00|
|1x Max 1550 w/ 2-Stack|2|4|20|55.51|1.91|

#### Medical Image 3D U-Net Training Performance Results

|GPUs|Ranks|Local Batch Size|Training Steps|Throughput w/ BF16 (samples/sec)|Weak Scaling w/ BF16|
|-|-|-|-|-|-|
|1x Max 1550 w/ 1-Stack|1|1|1000|12.81|1.00|
|1x Max 1550 w/ 2-Stack|2|1|1000|23.56|1.84|
|4x Max 1550|8|1|1000|87.07|6.80|


## Inference Performance Results

### Inference Performance on 1x Intel Data Center GPU Flex 170

The following tables show the performance numbers for several popular inference workloads on 1x Intel® Data Center GPU Flex 170 (150W PCIe, 1-stack for each GPU).

>**Note**: Inference with online mode refers to running the workloads using 1 as the batch size, while inference with batch mode utilizes larger batch size.

#### ResNet50v1-5 Inference Performance Results

|GPUs|Dataset|Image Size|Mode|Batch Size|Data Type|Inference Steps|Throughput (images/sec)|
|-|-|-|-|-|-|-|-|
|1x Flex 170|Dummy|224x224|Online|1|INT8|5000|435.01|
|1x Flex 170|Dummy|224x224|Batch|1024|INT8|5000|9842.75|

#### EfficientNet-B0 Inference Performance Results

|GPUs|Dataset|Image Size|Mode|Batch Size|Data Type|Inference Steps|Throughput (images/sec)|
|-|-|-|-|-|-|-|-|
|1x Flex 170|Dummy|224x224|Batch|64|FP16 (AMP)|50|3007.60|
|1x Flex 170|Dummy|224x224|Batch|128|FP16 (AMP)|50|3587.29|

#### EfficientNet-B3 Inference Performance Results

|GPUs|Dataset|Image Size|Mode|Batch Size|Data Type|Inference Steps|Throughput (images/sec)|
|-|-|-|-|-|-|-|-|
|1x Flex 170|Dummy|300x300|Batch|64|FP16 (AMP)|50|928.56|
|1x Flex 170|Dummy|300x300|Batch|128|FP16 (AMP)|50|968.83|

#### Mask-RCNN Inference Performance Results

|GPUs|Dataset|Mode|Batch Size|Data Type|Inference Steps|Throughput (images/sec)|
|-|-|-|-|-|-|-|
|1x Flex 170|COCO 2017|Online|1|FP16 (AMP)|5000|19.38|
|1x Flex 170|COCO 2017|Batch|16|FP16 (AMP)|312|43.02|

#### Stable Diffusion v1-4 Inference Performance Results

|GPUs|Dataset|Output <br>Image Size|Mode|Batch Size|Data Type|Diffusion Steps|Throughput <br>(iterations/sec)|Throughput Speedup <br>w/ FP16|
|-|-|-|-|-|-|-|-|-|
|1x Flex 170|Text Prompt|512x512|Online|1|FP32|50|2.91|1.00x|
|1x Flex 170|Text Prompt|512x512|Online|1|FP16 (pure)|50|6.53|2.24x|


## Configuration

### Software Configuration

#### Software Configuration for Intel Max 1550 GPU

|Software Component|Version|
|-|-|
|GPU Driver|[736.25](https://dgpu-docs.intel.com/releases/stable_736_25_20231031.html)|
|Intel® oneAPI Base Toolkit|2024.0|
|TensorFlow|v2.14.0|
|Intel® Extension for TensorFlow\*|v2.14.0.1|
|Intel® Optimization for Horovod\*|v0.28.1.2|

#### Software Configuration for Intel Flex 170 GPU

|Software Component|Version|
|-|-|
|GPU Driver|[736.25](https://dgpu-docs.intel.com/releases/stable_736_25_20231031.html)|
|Intel® oneAPI Base Toolkit|2024.0|
|TensorFlow|v2.14.0|
|Intel® Extension for TensorFlow\*|v2.14.0.1|

### Hardware Configuration

#### Hardware Configuration for Intel Max 1550 GPU

|GPU System|4x Intel® Data Center GPU Max 1550|
|-|-|
|**Number of Nodes**|1|
|**Xe®-Cores per GPU**|128 in total 2-Stack|
|**Memory Size per GPU**|128 GB HBM2e in total 2-Stack|
|**TDP per GPU**|600W|
|**GPU ECC Setting**|OFF|
|**Server Board**|Intel® Denali Pass D50DNP1SBB|
|**OS**|SUSE Linux Enterprise Server 15 SP4|
|**Kernel**|5.14.21-150400.24.69-default|
|**CPU Model**|Intel® Xeon® Platinum 8480+ @ 2.00 GHz|
|**Number of Sockets**|2|
|**CPU Cores per Socket**|56|
|**Hyper Threading**|ON|
|**Turbo Boost**|ON|
|**Automatic NUMA Balancing**|Enabled|
|**CPU Frequency Governor**|Performance|
|**TDP per CPU**|350W|
|**Installed Memory**|1024GB (16x64GB 4800 MT/s DDR5)|
|**NIC**|1x Intel® Ethernet Controller X710 for 10GBASE-T|
|**Storage**|1x WD® WD_BLACK SN850X 2TB NVMe SSD|

#### Hardware Configuration for Intel Flex 170 GPU

|GPU System|1x Intel® Data Center GPU Flex 170|
|-|-|
|**Number of Nodes**|1|
|**Xe®-Cores per GPU**|32|
|**Memory Size per GPU**|16 GB GDDR6|
|**TDP per GPU**|150W|
|**GPU ECC Setting**|ON|
|**Server Board**|Intel® Whitley|
|**OS**|Ubuntu 22.04.3 LTS|
|**Kernel**|5.15.0-57-generic|
|**CPU Model**|Intel® Xeon® Gold 6336Y CPU @ 2.40GHz|
|**Number of Sockets**|2|
|**CPU Cores per Socket**|24|
|**Hyper Threading**|ON|
|**Turbo Boost**|ON|
|**Automatic NUMA Balancing**|Enabled|
|**CPU Frequency Governor**|Performance|
|**TDP per CPU**|185W|
|**Installed Memory**|128GB (8x16GB 3200 MT/s DDR4)|
|**NIC**|2x Intel® Ethernet Controller X710 for 10GBASE-T, <br>1x Intel® 82574L Gigabit Ethernet Controller|
|**Storage**|1x Intel® SSDSC2KG960G8, <br>1x Samsung® 870 EVO 1TB SSD|

## Additional Performance Data for Intel AI Data Center Products

You can find the latest performance data on other Intel® AI Data Center Products such as 3rd, 4th, and 5th Gen Intel® Xeon® Scalable processors via [Performance Data for Intel® AI Data Center Products](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/performance.html/).
