# Intel® Extension for Tensorflow* Model Quantization API Example

## Overview

Intel® Extension for TensorFlow* provides Python APIs to support model quantization feature by cooperating with Intel® Neural Compressor. It will provide ideally quantization for better performance.

## Introduction

The example shows how to use Intel® Extension for TensorFlow* python APIs to implement model quantization.

## Environment

### Install Intel® Extension for Tensorflow*
Intel® Extension for Tensorflow* for Intel GPUs.

```shell
pip install tensorflow
pip install --upgrade intel-extension-for-tensorflow[gpu]
```

Intel® Extension for Tensorflow* for Intel CPUs.

```shell
pip install tensorflow
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### Install Intel® Neural Compressor
```shell
pip install neural-compressor>=2.0
```

## Prepare pre-trained model

* Get model from [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4/toolsdownloader/README.html)

```shell
git clone https://github.com/openvinotoolkit/open_model_zoo.git
git checkout 2021.4
cd open_model_zoo/tools/downloader/
./downloader.py --name ${model_name} --output_dir ${model_path}
```
* Download with URL

|	Model name	|	URL	|
|	--------------------------------	|	--------------------------------	|
|	faster_rcnn_resnet101_ava_v2.1	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz	|
|	faster_rcnn_resnet101_kitti	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz	|
|	faster_rcnn_resnet101_lowproposals_coco	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz	|
|	image-retrieval-0001	|	https://download.01.org/opencv/openvino_training_extensions/models/image_retrieval/image-retrieval-0001.tar.gz	|
|	SSD ResNet50 V1 FPN 640x640 (RetinaNet50)	|	http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz	|
|	ssd_inception_v2_coco	|	http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz	|
|	ssd-resnet34 300x300	|	https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/ssd_resnet34_fp32_bs1_pretrained_model.pb	|

## Prepare Dataset
We use dummy data for the model quantization example.

## Apply model quantization
Apply Intel® Extension for Tensorflow* model quantization Python APIs:
```python
from intel_extension_for_tensorflow.python.optimize.quantization import from_model, default_static_qconfig

# Create quantization converter with original model
converter=from_model(<model_path>)

# Specify quantization configure
converter.optimizations = default_static_qconfig()

# Specify dataset as dummy_v2
converter.representative_dataset = default_dataset('dummy_v2')(input_shape=(224, 224, 3), label_shape(1, ))

# Quantize the model
q_model = converter.convert()

# Save the quantized model to local disk
q_model.save(<output>)
```

Run model quantization example:

`itex_quantization.py` is an example of using Intel® Extension for Tensorflow* model quantization Python APIs to generate a quantized model from original model.

```shell
python itex_quantization.py --model_path <model_path> --output <output>
```
