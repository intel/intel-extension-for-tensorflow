# Intel® Extension for Tensorflow* Model Quantization API Example

## Overview

Intel® Extension for TensorFlow* offers improved model quantization performance
over stock TensorFlow by providing Python APIs that cooperate with Intel® Neural
Compressor.

This example shows how to use Intel® Extension for TensorFlow* python APIs to implement model quantization.

## Set Up Software Environment

### Install Intel® Extension for Tensorflow*

a) Choose installing either Intel® Extension for Tensorflow* for Intel GPUs:

   ```shell
   pip install tensorflow
   pip install --upgrade intel-extension-for-tensorflow[xpu]
   ```

b) or installing Intel® Extension for Tensorflow* for Intel CPUs:

   ```shell
   pip install tensorflow
   pip install --upgrade intel-extension-for-tensorflow[cpu]
   ```

### Install Intel® Neural Compressor
```shell
pip install neural-compressor>=2.0
```

## Prepare Pre-Trained Model

The Model Quantization API uses a pre-trained SavedModel protocol buffer file
(`pb` file).  If you don't have your own pre-trained model to try using the
quantization API, this table shows where to download some example pre-trained model data
files for your use:

  |	Model name	|	URL to Model's data files	|
  |	--------------------------------	|	--------------------------------	|
  |	faster_rcnn_resnet101_ava_v2.1	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz	|
  |	faster_rcnn_resnet101_kitti	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_kitti_2018_01_28.tar.gz	|
  |	faster_rcnn_resnet101_lowproposals_coco	|	http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz	|
  |	image-retrieval-0001	|	https://download.01.org/opencv/openvino_training_extensions/models/image_retrieval/image-retrieval-0001.tar.gz	|
  |	SSD ResNet50 V1 FPN 640x640 (RetinaNet50)	|	http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz	|
  |	ssd_inception_v2_coco	|	http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz	|
  |	ssd-resnet34 300x300	|	https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/ssd_resnet34_fp32_bs1_pretrained_model.pb	|

Use your browser or use the `wget` command to download the file into a working
directory. For example:

```
wget <URL to model's data file>
tar -zxvf <downloaded tar.gz file>  # if it's a tar.gz file
```

## Prepare Dataset
In the next section's code, we use dummy data for the model quantization example.

## Apply Model Quantization
Apply Intel® Extension for Tensorflow* model quantization Python APIs. Note that
`<model_path>` here is the path to the original pre-trained model `pb` file and
`<output>` is the path for the output quantized model's `pb` file.

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

## Run Model Quantization Example:

`itex_quantization.py` is an example of using Intel® Extension for Tensorflow* model quantization Python APIs to generate a quantized model from the original model.

```shell
python itex_quantization.py --model_path <path to original pb file> --output <path to output pb file>
```

The script displays the quantization statistics and quantized model path:

```
2023-10-22 00:00:00 [INFO] |**********Mixed Precision Statistics*********|
2023-10-22 00:00:00 [INFO] +-----------------------+-------+------+------+
2023-10-22 00:00:00 [INFO] |        Op Type        | Total | INT8 | FP32 |
2023-10-22 00:00:00 [INFO] +-----------------------+-------+------+------+
2023-10-22 00:00:00 [INFO] |        MaxPool        |   6   |  0   |  6   |
2023-10-22 00:00:00 [INFO] | DepthwiseConv2dNative |   1   |  0   |  1   |
2023-10-22 00:00:00 [INFO] |         MatMul        |   2   |  0   |  2   |
2023-10-22 00:00:00 [INFO] |        ConcatV2       |  210  |  0   | 210  |
2023-10-22 00:00:00 [INFO] |         Conv2D        |   72  |  0   |  72  |
2023-10-22 00:00:00 [INFO] |        AvgPool        |   7   |  0   |  7   |
2023-10-22 00:00:00 [INFO] |       QuantizeV2      |  133  | 133  |  0   |
2023-10-22 00:00:00 [INFO] |       Dequantize      |  163  | 163  |  0   |
2023-10-22 00:00:00 [INFO] |          Cast         |   14  |  0   |  14  |
2023-10-22 00:00:00 [INFO] +-----------------------+-------+------+------+
2023-10-22 00:00:00 [INFO] Pass quantize model elapsed time: 69612.36 ms
...
2023-10-22 00:00:00 [INFO] Save quantized model to <Path to quantized model>.
```
