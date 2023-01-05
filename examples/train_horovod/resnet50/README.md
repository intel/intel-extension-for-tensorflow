# Distributed training example with Intel® Optimization for Horovod*

## Model Information
| **Use Case** |**Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** 
| :---: | :---: | :---: | :---: | :---: |
| Training | Tensorflow | [Tensorflow-Models](https://github.com/tensorflow/models) | v2.8.0 | itex.yaml <br> itex_dummy.yaml <br> hvd_support_light.patch <br> or hvd_support.patch |

<br>

## Dependency
- [Tensorflow](https://pypi.org/project/tensorflow/)
- [Intel® Extension for TensorFlow*](https://pypi.org/project/intel-extension-for-tensorflow/)
- [Intel® Optimization for Horovod*](https://pypi.org/project/intel-optimization-for-horovod/)
- others show as below 
```
pip install gin gin-config tensorflow-addons tensorflow-model-optimization tensorflow-datasets
```

## Model examples preparation

### Model Repo
```
WORKSPACE=xxxx # please set your workspace folder
cd $WORKSPACE
git clone -b v2.8.0 https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git apply path/to/hvd_support_light.patch  # or path/to/hvd_support.patch
```
**hvd_support_light.patch** is the minimum change.
- hvd.init() is horovod initialization, including resource allocation.
- tf.config.experimental.set_memory_growth(): If memory growth is enabled, the runtime initialization will not allocate all memory on the device.
- tf.config.experimental.set_visible_devices(): Set the list of visible devices.
- strategy_scope: Remove native distributed.
- hvd.DistributedOptimizer(): use horovod distributed optimizer.
- dataset.shard(): Multiple workers run the same code but the different data. Dataset is split equally between diferent index workers.  
  
**hvd_support.patch** adds LARS optimizer [paper](https://arxiv.org/abs/1708.03888) 

### Download Dataset
Download imagenet dataset from https://image-net.org/download-images.php

<br>

## Execution
### Set Model Parameters
Export those parameters to script or environment.
```
export PYTHONPATH=${WORKSPACE}/tensorflow-models
MODEL_DIR=${WORKSPACE}/output
DATA_DIR=${WORKSPACE}/imagenet_data/imagenet

CONFIG_FILE=path/to/itex.yaml
NUMBER_OF_PROCESS=2
PROCESS_PER_NODE=2
```
- Download `itex.yaml` or `itex_dummy.yaml` and set one of them as CONFIG_FILE, then model would correspondly run with `real data` or `dummy data`. Default value is itex.yaml.
- Set `NUMBER_OF_PROCESS` and `PROCESS_PER_NODE` according to hvd rank number you need. Default value is a 2 rank task.
### HVD command

```
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p $MODEL_DIR
else
    rm -rf $MODEL_DIR && mkdir -p $MODEL_DIR                         
fi

mpirun -np $NUMBER_OF_PROCESS -ppn $PROCESS_PER_NODE --prepend-rank \
python ${PYTHONPATH}/official/vision/image_classification/classifier_trainer.py \
--mode=train_and_eval \
--model_type=resnet \
--dataset=imagenet \
--model_dir=$MODEL_DIR \
--data_dir=$DATA_DIR \
--config_file=$CONFIG_FILE
```

<br>

## OUTPUT
### Performance Data
```
[1] I0909 03:33:23.323099 140645511436096 keras_utils.py:145] TimeHistory: xxxx seconds, xxxx examples/second between steps 0 and 100
[0] I0909 03:33:23.324534 140611700504384 keras_utils.py:145] TimeHistory: xxxx seconds, xxxx examples/second between steps 0 and 100
[0] I0909 03:33:43.037004 140611700504384 keras_utils.py:145] TimeHistory: xxxx seconds, xxxx examples/second between steps 100 and 200
[1] I0909 03:33:43.037142 140645511436096 keras_utils.py:145] TimeHistory: xxxx seconds, xxxx examples/second between steps 100 and 200
[1] I0909 03:34:03.213994 140645511436096 keras_utils.py:145] TimeHistory: xxxx seconds, xxxx examples/second between steps 200 and 300
[0] I0909 03:34:03.214127 140611700504384 keras_utils.py:145] TimeHistory: xxxx seconds, xxxx examples/second between steps 200 and 300
```
