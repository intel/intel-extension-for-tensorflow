# Resnet50 train on Intel GPU

## Introduction

Intel® Extension for TensorFlow* is compatible with stock Tensorflow*. 
This example shows resnet50 training.

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Max Series
 
## Prerequisites

### Model Code change
We optimized bf16 in resnet50.patch, and enable horovod and LARS in hvd_support.patch, please apply patch
```
git clone -b v2.8.0 https://github.com/tensorflow/models.git tensorflow-models
```

### Prepare for GPU (Skip this step for CPU)

Refer to [Prepare](../common_guide_running.html##Prepare)

### Setup Running Environment

* Setup for GPU
```bash
./pip_set_env.sh
```

### Enable Running Environment

Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.html##Running)

### Apply Patch

#### If not use Horovod
```
git apply path/to/configure/resnet50.patch
```

#### If use Horovod
```
git apply path/to/hvd_configure/hvd_support.patch
```
#### Prepare ImageNet dataset
Using TFDS
classifier_trainer.py supports ImageNet with [TensorFlow Datasets(TFDS)](https://www.tensorflow.org/datasets/overview) .

Please see the following [example snippet](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/scripts/download_and_prepare.py) for more information on how to use TFDS to download and prepare datasets, and specifically the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.html) for manual download instructions.

Legacy TFRecords
Download the ImageNet dataset and convert it to TFRecord format. The following [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) and [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy) provide a few options.

Note that the legacy ResNet runners, e.g. [resnet/resnet_ctl_imagenet_main.py](https://github.com/tensorflow/models/blob/v2.8.0/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py) require TFRecords whereas `classifier_trainer.py` can use both by setting the builder to 'records' or 'tfds' in the configurations.

## Execution
### Set Model Parameters
There are several config yaml files in configure and hvd_configure folder. Set one of them as CONFIG_FILE, then model would correspondly run with `real data` or `dummy data`. Single-tile please use yaml file in configure folder. Distribute training please use yaml file in hvd_configure folder, `itex_bf16_lars.yaml`/`itex_fp32_lars.yaml` for HVD real data and `itex_dummy_bf16_lars.yaml`/`itex_dummy_fp32_lars.yaml` for HVD dummy data.
Export those parameters to script or environment.
```
export PYTHONPATH=/the/path/to/tensorflow-models
MODEL_DIR=/the/path/to/output
DATA_DIR=/the/path/to/imagenet
CONFIG_FILE=path/to/itex_xx.yaml  # itex_bf16.yaml/itex_fp32.yaml for accuracy, itex_dummy_bf16.yaml/itex_dummy_fp32.yaml for benchmark

```

### Command

```
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p $MODEL_DIR
else
    rm -rf $MODEL_DIR && mkdir -p $MODEL_DIR                         
fi

python ${PYTHONPATH}/official/vision/image_classification/classifier_trainer.py \
--mode=train_and_eval \
--model_type=resnet \
--dataset=imagenet \
--model_dir=$MODEL_DIR \
--data_dir=$DATA_DIR \
--config_file=$CONFIG_FILE
```

### Command with Horovod
Set `NUMBER_OF_PROCESS` and `PROCESS_PER_NODE` according to hvd rank number you need. Default value is 2 rank task.

```
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p $MODEL_DIR
else
    rm -rf $MODEL_DIR && mkdir -p $MODEL_DIR                         
fi

NUMBER_OF_PROCESS=2
PROCESS_PER_NODE=2

mpirun -np $NUMBER_OF_PROCESS -ppn $PROCESS_PER_NODE --prepend-rank \
python ${PYTHONPATH}/official/vision/image_classification/classifier_trainer.py \
--mode=train_and_eval \
--model_type=resnet \
--dataset=imagenet \
--model_dir=$MODEL_DIR \
--data_dir=$DATA_DIR \
--config_file=$CONFIG_FILE
```

## Example Output without hvd
```
I0203 02:48:01.006297 139660941027136 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 1900 and 2000
I0203 02:48:16.590331 139660941027136 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 2000 and 2100
I0203 02:48:32.178206 139660941027136 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 2100 and 2200
I0203 02:48:47.790128 139660941027136 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 2200 and 2300
I0203 02:49:03.408512 139660941027136 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 2300 and 2400
```
## Example Output with hvd
```
[0] I0817 00:09:07.602742 139898862851904 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 400 and 600
[1] I0817 00:09:07.603262 140612319840064 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 400 and 600
[0] I0817 00:10:07.917546 139898862851904 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 600 and 800
[1] I0817 00:10:07.917738 140612319840064 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 600 and 800
[0] I0817 00:11:08.277716 139898862851904 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 800 and 1000
[1] I0817 00:11:08.277811 140612319840064 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 800 and 1000
[0] I0817 00:12:08.555174 139898862851904 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 1000 and 1200
[1] I0817 00:12:08.555221 140612319840064 keras_utils.py:145] TimeHistory: xx seconds, xxxx examples/second between steps 1000 and 1200