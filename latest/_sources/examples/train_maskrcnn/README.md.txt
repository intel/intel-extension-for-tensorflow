# Accelerate Mask R-CNN Training w/o horovod on Intel GPU

## Introduction

Intel® Extension for TensorFlow* is compatible with stock TensorFlow*. 
This example shows Mask R-CNN Training. It contains single-tile training scripts and multi-tile training scripts with horovod.

Install the Intel® Extension for TensorFlow* in legacy running environment, Tensorflow will execute the Training on Intel GPU.

## Hardware Requirements

Verified Hardware Platforms:

 - Intel® Data Center GPU Max Series

## Prerequisites

### Model Code change

To get better performance, instead of installing the official repository, you can apply the patch and install it as shown here:

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
git checkout c481324031ecf0f70f8939516c02e16cac60446d
git apply patch  # When applying this patch, please move it to the above MaskRCNN dir first.
```

### Prepare for GPU

Refer to [Prepare](../common_guide_running.html##Prepare).

### Setup Running Environment

You can use `./pip_set_env.sh` to setup for GPU. It contains the following two steps: creating virtual environment and installing python packages.

+ Create Virtual Environment

```
python -m venv env_itex
source source env_itex/bin/activate
```

+ Install

```
pip install --upgrade pip
pip install --upgrade intel-extension-for-tensorflow[gpu]
pip install intel-optimization-for-horovod
pip install opencv-python-headless pybind11
pip install "git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI"
pip install -e "git+https://github.com/NVIDIA/dllogger#egg=dllogger"
```

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.html##Running)

### Prepare Dataset

Assume current_dir is `examples/train_maskrcnn/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN`. So as the following parts.

+ Download and preprocess the [COCO 2017 dataset](http://cocodataset.org/#download).

```
cd dataset
bash download_and_preprocess_coco.sh ./data
```

+ Download the pre-trained ResNet-50 weights.

```
python scripts/download_weights.py --save_dir=./weights
```

## Execute the Example

Here we provide single-tile training scripts and multi-tile training scripts with horovod. The datatype can be float32 or bfloat16.

```
DATASET_DIR=./data
PRETRAINED_DIR=./weights
OUTPUT_DIR=/the/path/to/output_dir
```

+ Single tile with fp32

```
python main.py train \
--data_dir $DATASET_DIR \
--model_dir=$OUTPUT_DIR \
--train_batch_size 4 \
--seed=0 --use_synthetic_data \
--epochs 1 --steps_per_epoch 20 --log_every=1 --log_warmup_steps=1
```

+ Single tile with bf16, it requires `--amp` flag.

```
python main.py train \
--data_dir $DATASET_DIR \
--model_dir=$OUTPUT_DIR \
--train_batch_size 4 \
--amp --seed=0 --use_synthetic_data \
--epochs 1 --steps_per_epoch 20 --log_every=1 --log_warmup_steps=1
```

+ Multi-tile with horovod.  Default datatype is fp32. You can use `--amp` flag for bf16.

```
mpirun -np 2 -prepend-rank -ppn 1 \
python main.py train \
--data_dir $DATASET_DIR \
--model_dir=$OUTPUT_DIR \
--train_batch_size 4 \
--seed=0 --use_synthetic_data \
--epochs 1 --steps_per_epoch 20 --log_every=1 --log_warmup_steps=1
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.

``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```