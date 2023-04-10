# Environment Variables

## Overview

Intel® Extension for TensorFlow* provides environment variables for users to quickly adjust the configuration without any model code changes. Not all configuration options can be set via environment variable. For the rest, you'll need to use the Python APIs. The priority for setting values is Python APIs > Environment Variables > Default value. 

## Environment Variables only

| Environment Variables Names    | Default Value | Definition | 
| ------------------------------ | ------------- | ---------------------------------------------- | 
| ITEX_TILE_AS_DEVICE            | `1`             | The default is `1`, which will configure every tile as TensorFlow individual device in the scenario of one GPU card with multiple tiles. If set to `0`, the whole GPU card will be treated as single Tensorflow device for execution.|
| ITEX_FP32_MATH_MODE            | `FP32`        | Sets oneDNN primitive floating-point math mode. The value can be `FP32` or `TF32` in GPU device and  `FP32` or `BF32` in CPU device. Default will be `FP32`.|
| ITEX_AUTO_MIXED_PRECISION_LOG_PATH | `auto_mixed_precision_log_path` | Sets log path         |
| ITEX_VERBOSE                       | `1`                       | Same semantics as `TF_CPP_MAX_VLOG_LEVEL`, but only works with Intel® Extension for TensorFlow* |

#### ITEX_VERBOSE level definition
* Level 1 is basic verbose information including device, graph, kernel and other infrastructure initialization logs, displayed only once.
* Level 2 is help information including performance, functionality, debug and error logs, displayed only once.
* Level 3 is normal tips or kernel execution logs. It is reported multiple times across different iterations. Note that Intel® Extension for TensorFlow* will generate large size logs using this level.
* Level 4 is for graph information. This level can dump node or large-size whole graph information. Example:
```
node {
  name: "bert/encoder/layer_0/attention/output/dense/BiasAdd"
  op: "BiasAdd"
  input: "bert/encoder/layer_0/attention/output/dense/MatMul"
  input: "bert/encoder/layer_0/attention/output/dense/bias/read"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
```
* Levels 5 or higher are used for any remaining situations.


## Environment Variables with Python APIs


### Backend and Config Protocol

Refer to [Python APIs and preserved environment variable Names](python_api.md#python-apis-and-preserved-environment-variable-names).


### Auto Mixed Precision Options

Refer to [Advanced Auto Mixed Precision](advanced_auto_mixed_precision.md).

### Customized Operators Experimental Override
Refer to [Customized Operators Experimental Override](itex_ops_override.md).
