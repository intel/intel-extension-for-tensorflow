# Accelerate 3D-Unet Training w/o horovod for medical image segmentation on Intel GPU

## Introduction

Intel® Extension for TensorFlow* is compatible with stock TensorFlow*. 
This example shows 3D-UNet Training for medical image segmentation. It contains single-tile training scripts and multi-tile training scripts with horovod.

Install the Intel® Extension for TensorFlow* in legacy running environment, Tensorflow will execute the Training on Intel GPU.

## Hardware Requirements

Verified Hardware Platforms:

 - Intel® Data Center GPU Max Series

## Prerequisites

### Model Code change

To get better performance, instead of installing the official repository, you can apply the patch and install it as shown here. You can choose one patch from single-tile patch `3dunet_itex.patch` and multi-tile patch `3dunet_itex_with_horovod.patch`. 

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/TensorFlow/Segmentation/UNet_3D_Medical/
git checkout 88eb3cff2f03dad85035621d041e23a14345999e
git apply patch  # When applying this patch, please move it to the above 3D-UNet dir first.
```

### Prepare for GPU

Refer to [Prepare](../common_guide_running.md#prepare).

### Setup Running Environment

You can use `./pip_set_env.sh` to setup for GPU. It contains the following two steps: creating virtual environment and installing python packages.

+ Create Virtual Environment

```
python -m venv env_itex
source env_itex/bin/activate
```

+ Install

```
pip install --upgrade pip
pip install --upgrade intel-extension-for-tensorflow[gpu]
pip install intel-optimization-for-horovod
pip install tfa-nightly
pip install git+https://github.com/NVIDIA/dllogger.git
```

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md#running)

### Prepare Dataset

We use [Brain Tumor Segmentation 2019 dataset](https://www.med.upenn.edu/cbica/brats-2019/) for 3D-UNet training. Upon registration, the challenge's data is made available through the `https//ipp.cbica.upenn.edu` service.

The training and test datasets are given as 3D `nifti` volumes that can be read using the Nibabel library and NumPy. It can be converted from `nifti` to `tfrecord` using `./dataset/preprocess_data.py` script.

## Execute the Example

Assume current_dir is `examples/train_maskrcnn/DeepLearningExamples/TensorFlow/Segmentation/UNet_3D_Medical/`.

Here we provide single-tile training scripts and multi-tile training scripts with horovod. The datatype can be float32 or bfloat16.

```
DATASET_DIR=/the/path/to/dataset
OUTPUT_DIR=/the/path/to/output_dir
```

### Single Tile

First apply patch.

```
git apply 3dunet_itex.patch
```

+ float32

```
python main.py --benchmark --data_dir $DATASET_DIR --model_dir $OUTPUT_DIR --exec_mode train --batch_size $BATCH_SIZE --warmup_steps 150 --max_steps 1000 --log_every 1 
```

+ bfloat16

```
python main.py --benchmark --data_dir $DATASET_DIR --model_dir $OUTPUT_DIR --exec_mode train --warmup_steps 150 --max_steps 1000 --batch_size=$BATCH_SIZE --log_every 1 --amp
```

### Multi-tile with horovod

First apply patch.

```
git apply 3dunet_itex_with_horovod.patch
```

+ float32

```
mpirun -np 2 -prepend-rank -ppn 2 \
  python main.py --data_dir=$DATASET_DIR --benchmark --model_dir=$MODEL_DIR --exec_mode train --warmup_steps 150 --max_steps 1000 --batch_size=$BATCH_SIZE
```

+ bf16

```
mpirun -np 2 -prepend-rank -ppn 2 \
  python main.py --data_dir=$DATASET_DIR --benchmark --model_dir=$MODEL_DIR --exec_mode train --warmup_steps 150 --max_steps 1000 --batch_size=$BATCH_SIZE --amp
```

## FAQ

1. If you get the following error log, refer to [Enable Running Environment](#Enable-Running-Environment) to Enable oneAPI running environment.

``` 
tensorflow.python.framework.errors_impl.NotFoundError: libmkl_sycl.so.2: cannot open shared object file: No such file or directory
```
