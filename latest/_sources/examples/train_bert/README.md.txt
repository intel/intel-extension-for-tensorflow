# BERT Training for Classifying Text on Intel CPU and GPU

## Introduction

Intel® Extension for TensorFlow* is compatable with stock Tensorflow*. 
This example uses the tutorial from tensorflow.org [Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) to show training on BERT model without change original code.

Install the Intel® Extension for TensorFlow* in legacy running envrionment, Tensorflow will execute the training on Intel CPU and GPU.

**No need any code change**

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Ponte Vecchio
 - Intel CPU
 
## Prerequisites

### Prepare for GPU (Skip this step for CPU)

Refer to [Prepare](../common_guide_running.md##Prepare)

### Setup Running Environment


* Setup for GPU
```bash
./set_env_gpu.sh
```

* Setup for CPU
```bash
./set_env_cpu.sh
```

### Enable Running Environment

1. Enable oneAPI running environment (only for GPU) and virtual running environment.

   * For GPU, refer to [Running](../common_guide_running.md##Running)

   * For CPU, 
```bash
source env_itex_cpu/bin/activate
```

2. Install the python packages used in [Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert).

The jupyter nodebook script will install other depended pacakge during running.
No need pre-install them.

### Download Jupyter Code:

```
wget https://storage.googleapis.com/tensorflow_docs/text/docs/tutorials/classify_text_with_bert.ipynb
```

## Startup Jupyter Notebook

```
jupyter notebook --notebook-dir=./ --ip=0.0.0.0 --no-browser  --allow-root &
...

http://xxx.xxx:8888/?token=f502f0715979ec73c571ca5676ba58431b916f5f58ee3333

```
Open the url:http://xxx.xxx:8888/?token=f502f0715979ec73c571ca5676ba58431b916f5f58ee3333 above by web browser.

## Execute

1. Open classify_text_with_bert.ipynb by jupyter notebook.

2. Run the tutorial according to the descrition in the Jupyter notebook.

3. The TensorFlow will train and infer the BERT model on Intel CPU or GPU.


## FAQ

1. The following error log is a known issue and it is not caused by Intel® Extension for TensorFlow*. This crash happens when the code is finished and try to release resource and  doesn't impact the result of the Bert training and inference/test.

``` Exception ignored in: <function _CheckpointRestoreCoordinatorDeleter.__del__ at 0x7fa167430d30>
Traceback (most recent call last):
  File "/home/xxx/xxx/env_itex/lib/python3.9/site-packages/tensorflow/python/training/tracking/util.py", line 174, in __del__
TypeError: 'NoneType' object is not callable
```
