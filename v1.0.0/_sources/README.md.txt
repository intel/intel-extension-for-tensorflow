# Intel® Extension for TensorFlow*

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

Intel® Extension for TensorFlow* provides [Intel GPU](docs/install/install_for_gpu.md#hardware-requirements) support and experimental [Intel CPU](docs/install/experimental/install_for_cpu.md#hardware-requirements) support.

### Software Requirement

|Package|CPU|GPU|Installation|
|-|-|-|-|
|Intel GPU driver||Y|[Install Intel GPU driver](docs/install/install_for_gpu.md#install-gpu-drivers)|
|Intel® oneAPI Base Toolkit||Y|[Install Intel® oneAPI Base Toolkit](docs/install/install_for_gpu.md#install-oneapi-base-toolkit-packages)|
|TensorFlow|Y|Y|[Install TensorFlow 2.10.0](https://www.tensorflow.org/install)|

### Installation Channel:
Intel® Extension for TensorFlow* can be installed from the following channels:

|PyPI|DockerHub|Source|
|-|-|-|
|[GPU](docs/install/install_for_gpu.md#install-via-pypi-wheel-in-bare-metal) \ [CPU](docs/install/experimental/install_for_cpu.md#install-via-pypi-wheel-in-bare-metal)  |[ GPU Container ](docs/install/install_for_gpu.md#install-via-docker-container) \ [ CPU Container](docs/install/experimental/install_for_cpu.md#install-via-docker-container)|[Build from source](docs/install/how_to_build.md)|


### Compatibility Table

| Intel ® Extension for TensorFlow*  | Stock TensorFlow |
| ------- | ----------- |    
| v1.0.0  | 2.10        | 

### Install for GPU
```
pip install tensorflow==2.10.0
pip install intel-extension-for-tensorflow[gpu]==1.0.0
```
Please refer to [GPU installation](docs/install/install_for_gpu.md) for details.

### Install for CPU [Experimental]
```
pip install tensorflow==2.10.0
pip install intel-extension-for-tensorflow[cpu]==1.0.0
```

Sanity check by:
```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```


## Documentation 

Visit the [online document website](https://intel.github.io/intel-extension-for-tensorflow/latest/), and then get started the tour from Intel® Extension for TensorFlow* [examples](examples/README.md).

## Contributing

We welcome community contributions to Intel® Extension for TensorFlow*. 

This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](CODE_OF_CONDUCT.md). Please see [contribution guidelines](docs/community/contributing.md) for additional details.

## Resources
- [TensorFlow](https://www.tensorflow.org/)

## Support
Please submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/intel/intel-extension-for-tensorflow/issues) page.

## Security
See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html) for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

## Licence
[Apache License 2.0](LICENSE.txt)

This distribution includes third party software governed by separate license terms. This third party software, even if included with the distribution of the Intel software, may be governed by separate license terms, including without limitation, third party license terms, other Intel software license terms, and open source software license terms. These separate license terms govern your use of the third party programs as set forth in the ["THIRD-PARTY-PROGRAMS"](third-party-programs/THIRD-PARTY-PROGRAMS) file.
