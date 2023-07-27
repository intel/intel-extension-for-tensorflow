# Quantize Inception V3 by Intel® Extension for Tensorflow* on Intel® Xeon®


## Background

Intel® Extension for TensorFlow* provides quantization feature by cooperating with Intel® Neural Compressor and oneDNN Graph. It will provide better quantization: better performance and accuracy loss under control.

Intel® Neural Compressor executes the calibration process to output the QDQ quantization model, which inserts Quantize and Dequantize layers to includes help information for quantization.

When you use Intel® Extension for Tensorflow* to execute the inference of this model, oneDNN Graph will be called to quantize and optimize the model. Then the quantized model will be executed by Intel® Extension for Tensorflow* and accelerated by Intel® Deep Learning Boost or Intel® Advanced Matrix Extensions on Intel® Xeon® processors.

## Introduction

The example shows an end-to-end pipeline:

1. Train an Inception V3 model with a flower photo dataset by transfer learning.

2. Execute the calibration by Intel® Neural Compressor.

3. Quantize and accelerate the inference by Intel® Extension for Tensorflow* for GPU and CPU.
  

## Configuration

### Intel® Extension for Tensorflow* Version

Install Intel® Extension for Tensorflow* >= 2.13.0.0 for this feature.

### Enable oneDNN Graph

By default, oneDNN Graph is enabled in Intel® Extension for Tensorflow* for INT8 models.
 
Enable it explicitly by:
 
```
  import os
  os.environ["ITEX_ONEDNN_GRAPH"] = "1"
```

### Disable Constant Folding Function

We need to disable Constant Folding function in 2 stages:

1. Intel® Neural Compressor creates QDQ quantization model.

2. Intel® Extension for Tensorflow* executes the oneDNN Graph quantization path.


There are 2 methods to configure:

a. Environment Variable
```
export ITEX_TF_CONSTANT_FOLDING=0
```

b. Python API
```
from tensorflow.core.protobuf import rewriter_config_pb2

infer_config = tf.compat.v1.ConfigProto()
infer_config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF

session = tf.compat.v1.Session(config=infer_config)
tf.compat.v1.keras.backend.set_session(session)
```

## Hardware Environment
Support: Intel® Xeon® CPU & Intel® Data Center Flex Series GPU.

### CPU

It's recommended to run the example on the Intel® Xeon® processors, which supports Intel® Deep Learning Boost or Intel® Advanced Matrix Extensions.

Without the hardware features above for AI workloads, the performance speedup with FP32 will not be increased much.

#### Check Intel® Deep Learning Boost

In Linux, run command:

```
lscpu | grep vnni
```

You are expected to see `avx_vnni` and `avx512-vnni`, otherwise your processors do not support Intel® Deep Learning Boost.

#### Check Intel® Advanced Matrix Extensions

In Linux, run command:

```
lscpu | grep amx
```
You are expected to see `amx_bf16` and `amx_int8`, otherwise your processors do not support Intel® Advanced Matrix Extensions.

### GPU

Support: Intel® Data Center Flex Series GPU.

#### Local Server

Install the GPU driver and oneAPI packages by referring to [Intel GPU Software Installation](/docs/install/install_for_gpu.md).

### Intel® DevCloud

If you have no CPU support Intel® Deep Learning Boost or Intel® Advanced Matrix Extensions or no Intel GPU support INT8, you could register on Intel® DevCloud and try this example on an second generation Intel® Xeon based processors or newer. To learn more about working with Intel® DevCloud, refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html)


## Running Environment

1. Install Python versions >=3.8 and versions <=3.10 supported by Intel® Extension for Tensorflow*.

2. Create the running Python Virtual environment **env_itex**.

```
bash pip_set_env.sh
```

3. Activate

```
source env_itex/bin/activate
```

## Startup Jupyter Notebook

1. Startup
```
bash run_jupyter.sh

...
http://xxx.yyy.com:8888/xxxxxxxx

```

2. Open the link outputted by Jupyter Notebook in your browser.

3. Choose and open the **quantize_inception_v3.ipynb** in Jupyter Notebook.

Set the kernel to "env_itex".

Execute the code as the guide.


## License

Code samples are licensed under the MIT license.
