# Intel® Extension for TensorFlow*

[![Python](https://img.shields.io/pypi/pyversions/intel_extension_for_tensorflow)](https://badge.fury.io/py/intel-extension-for-tensorflow)
[![PyPI version](https://badge.fury.io/py/intel-extension-for-tensorflow.svg)](https://badge.fury.io/py/intel-extension-for-tensorflow)
[![version](https://img.shields.io/github/v/release/intel/intel-extension-for-tensorflow?color=brightgreen)](https://github.com/intel/intel-extension-for-tensorflow/releases)

Intel® Extension for TensorFlow* is a heterogeneous, high performance deep learning extension plugin based on TensorFlow [PluggableDevice](https://github.com/tensorflow/community/blob/master/rfcs/20200624-pluggable-device-for-tensorflow.md) interface to bring Intel XPU(GPU, CPU, etc) devices into [TensorFlow](https://github.com/tensorflow/tensorflow) open source community for AI workload acceleration. It allows flexibly plugging an XPU into TensorFlow on-demand, and exposing computing power inside Intel's hardware.

Documentation: [**Intel® Extension for TensorFlow\* online document website**](https://intel.github.io/intel-extension-for-tensorflow/).

## Installation

### Install for GPU
```
pip install tensorflow==2.13.0
pip install --upgrade intel-extension-for-tensorflow[gpu]
```
Please refer to [GPU installation](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/install/install_for_gpu.html) for details.

### Install for CPU [Experimental]
```
pip install tensorflow==2.13.0
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

## Security
See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html) for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](https://intel.github.io/intel-extension-for-tensorflow/latest/SECURITY.html)
