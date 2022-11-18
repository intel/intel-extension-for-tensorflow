# Quick Get Started

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://pypi.org/project/intel-extension-for-tensorflow)
[![version](https://img.shields.io/badge/release-1.0.0-green)](https://github.com/intel/intel-extension-for-tensorflow/releases)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](LICENSE.txt)

Intel® Extension for TensorFlow* is a heterogeneous, high performance deep learning extension plugin based on TensorFlow [PluggableDevice](https://github.com/tensorflow/community/blob/master/rfcs/20200624-pluggable-device-for-tensorflow.md) interface to bring Intel XPU(GPU, CPU, etc) devices into [TensorFlow](https://github.com/tensorflow/tensorflow) open source community for AI workload acceleration. It allows flexibly plugging an XPU into TensorFlow on-demand, and exposing computing power inside Intel's hardware.

This diagram provides a summary of the TensorFlow* PyPI package ecosystem.

<div align=center>
<img src="docs/guide/images/pip_pkg_deps.png">
</div>


* TensorFlow PyPI packages:
  [estimator](https://www.tensorflow.org/guide/estimator), [keras](https://keras.io), [tensorboard](https://www.tensorflow.org/tensorboard), [tensorflow-base](https://www.tensorflow.org/guide)

* Intel® Extension for TensorFlow* package:
  
   `intel_extension_for_tensorflow` contains:
   * XPU specific implementation
     * kernels & operators
     * graph optimizer
     * device runtime 
   * XPU configuration management
     * XPU backend selection
     * Options turning on/off advanced features

## Install

### Hardware Requirement

Intel® Extension for TensorFlow* provides [Intel GPU](docs/install/install_for_gpu.html#hardware-requirements) support and experimental [Intel CPU](docs/install/experimental/install_for_cpu.html#hardware-requirements) support.

### Software Requirement

|Package|CPU|GPU|Installation|
|-|-|-|-|
|Intel GPU driver||Y|[Install Intel GPU driver](docs/install/install_for_gpu.html#install-gpu-drivers)|
|Intel® oneAPI Base Toolkit||Y|[Install Intel® oneAPI Base Toolkit](docs/install/install_for_gpu.html#install-oneapi-base-toolkit-packages)|
|TensorFlow|Y|Y|[Install TensorFlow 2.10.0](https://www.tensorflow.org/install)|

### Installation Channel:
Intel® Extension for TensorFlow* can be installed from the following channels:

|PyPI|DockerHub|Source|
|-|-|-|
|[GPU](docs/install/install_for_gpu.html#install-via-pypi-wheel-in-bare-metal) \ [CPU](docs/install/experimental/install_for_cpu.html#install-via-pypi-wheel-in-bare-metal)  |[ GPU Container ](docs/install/install_for_gpu.html#install-via-docker-container) \ [ CPU Container](docs/install/experimental/install_for_cpu.html#install-via-docker-container)|[Build from source](docs/install/how_to_build.html)|


### Install for GPU

```
pip install tensorflow==2.10.0
pip install --upgrade intel-extension-for-tensorflow[gpu]
```

Please refer to [GPU installation](docs/install/install_for_gpu.md) for details.

### Install for CPU [Experimental]
```
pip install tensorflow==2.10.0
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

Sanity check by:
```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```


## Support
Please submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues) page.
