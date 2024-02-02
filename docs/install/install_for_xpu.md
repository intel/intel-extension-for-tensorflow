# Intel XPU Software Installation

This guide shows how to use an Intel® Extension for TensorFlow* XPU package, which provides GPU and CPU support simultaneously.

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Max Series, Driver Version: [736](https://dgpu-docs.intel.com/releases/stable_736_25_20231031.html)
 - Intel® Data Center GPU Flex Series 170, Driver Version: [736](https://dgpu-docs.intel.com/releases/stable_736_25_20231031.html)
 - *Experimental:* Intel® Arc™ A-Series

For experimental support of the Intel® Arc™ A-Series GPUs, please refer to [Intel® Arc™ A-Series GPU Software Installation](experimental/install_for_arc_gpu.md) for details.

## Software Requirements
- Ubuntu 22.04, Red Hat 8.6 (64-bit)
  - Intel® Data Center GPU Flex Series
- Ubuntu 22.04, Red Hat 8.6 (64-bit), SUSE Linux Enterprise Server(SLES) 15 SP4/SP5
  - Intel® Data Center GPU Max Series
- Intel® oneAPI Base Toolkit 2024.0
- TensorFlow 2.14.0
- Python 3.9-3.11
- pip 19.0 or later (requires manylinux2014 support)


## Install GPU Drivers

|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|
|Ubuntu 22.04, Red Hat 8.6|Intel® Data Center GPU Flex Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-flex-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [736](https://dgpu-docs.intel.com/releases/stable_736_25_20231031.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd==23.30.26918.50-736~22.04`|
|Ubuntu 22.04, Red Hat 8.6, SLES 15 SP4/SP5|Intel® Data Center GPU Max Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [736](https://dgpu-docs.intel.com/releases/stable_736_25_20231031.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd==23.30.26918.50-736~22.04`|

## Install via Docker container

The Docker container includes the Intel® oneAPI Base Toolkit, and all other software stack except Intel GPU Drivers. Install the GPU driver in host machine bare metal environment, and then launch the docker container directly.

#### Build Docker container from Dockerfile

Run the following [Dockerfile build procedure](./../../docker/README.md) to build the pip based deployment container.

#### Get docker container from dockerhub

Pre-built docker images are available at [DockerHub](https://hub.docker.com/r/intel/intel-extension-for-tensorflow/tags).
Run the following command to pull Intel® Extension for TensorFlow* Docker container image (`xpu`) to your local machine.

```
$ docker pull intel/intel-extension-for-tensorflow:xpu
$ docker run -it -p 8888:8888 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path intel/intel-extension-for-tensorflow:xpu
```

To use Intel® Optimization for Horovod* with the Intel® oneAPI Collective Communications Library (oneCCL), pull Intel® Extension for TensorFlow* Docker container image (`xpu`) to your local machine and use the script to set the required environment variables after creating the container by the following command. You can also get the script via [horovod-vars.sh](../../docker/horovod-vars.sh)

```
$ docker pull intel/intel-extension-for-tensorflow:xpu
$ docker run -it -p 8888:8888 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host intel/intel-extension-for-tensorflow:xpu
$ source /opt/intel/horovod-vars.sh
```

Then go to your browser on http://localhost:8888/

## Install via PyPI wheel in bare metal

#### Install oneAPI Base Toolkit Packages

Need to install components of Intel® oneAPI Base Toolkit:
- Intel® oneAPI DPC++ Compiler
- Intel® oneAPI Math Kernel Library (oneMKL)
- Intel® oneAPI Threading Building Blocks (TBB), dependency of DPC++ Compiler.
- Intel® oneAPI Collective Communications Library (oneCCL), required by Intel® Optimization for Horovod* only


```bash
$ wget https://registrationcenter-download.intel.com/akdlm//IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564.sh
# 3 components are necessary: DPC++/C++ Compiler, DPC++ Libiary and oneMKL
# if you want to run distributed training with Intel® Optimization for Horovod*, oneCCL is needed too(Intel® oneAPI MPI Library will be installed automatically as its dependency)
$ sudo sh l_BaseKit_p_2024.0.0.49564.sh
```

For any more details, follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

#### Setup environment variables
```bash
# DPC++ Compiler/oneMKL
source /path to basekit/intel/oneapi/compiler/latest/env/vars.sh
source /path to basekit/intel/oneapi/mkl/latest/env/vars.sh

# oneCCL (and Intel® oneAPI MPI Library as its dependency), required by Intel® Optimization for Horovod* only
source /path to basekit/intel/oneapi/mpi/latest/env/vars.sh
source /path to basekit/intel/oneapi/ccl/latest/env/vars.sh
```

You may install more components than Intel® Extension for TensorFlow* needs, and if required, `setvars.sh` can be customized to point to a specific directory by using a [configuration file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html):

```bash
source /opt/intel/oneapi/setvars.sh --config="full/path/to/your/config.txt"
```

#### Install TensorFlow

The Python development and virtual environment setup recommendation by TensorFlow to isolate package installation from the system.

The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be == 2.14.0.


###### Virtual environment install

You can follow the instructions in [stock tensorflow install](https://www.tensorflow.org/install/pip#step-by-step_instructions) to activate the virtual environment.

On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
```bash
(tf)$ pip install --upgrade pip
```

To install in virtual environment, you can run
```bash
(tf)$ pip install tensorflow==2.14.0
```

###### System environment install
If you prefer install tensorflow in $HOME, please append `--user` to the commands.
```bash
$ pip install --user tensorflow==2.14.0
```
And the following system environment install for Intel® Extension for TensorFlow* will also append `--user` to the command.

#### Install Intel® Extension for TensorFlow*

To install a XPU version in virtual environment, which depends on Intel GPU drivers and oneAPI BaseKit, you can run

```bash
(tf)$ pip install --upgrade intel-extension-for-tensorflow[xpu]
```

##### Check the Environment for XPU
```bash
(tf)$ export path_to_site_packages=`python -c "import site; print(site.getsitepackages()[0])"`
(tf)$ bash ${path_to_site_packages}/intel_extension_for_tensorflow/tools/env_check.sh
```

##### Verify the Installation
```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```

Then, you can get the information that both CPU and GPU backends are loaded successfully  from the console log.
```
2023-07-28 12:00:00.374832: I itex/core/wrapper/itex_cpu_wrapper.cc:42] Intel Extension for Tensorflow* AVX512 CPU backend is loaded.
2023-07-28 12:00:00.217981: I itex/core/wrapper/itex_gpu_wrapper.cc:35] Intel Extension for Tensorflow* GPU backend is loaded.
```
**NOTE**: If Intel® Extension for TensorFlow* XPU package is installed on GPU support platform, both CPU and GPU backends will be loaded as pluggable device via TensorFlow. GPU backend will be activated as default backend.

## XPU for CPU only platform
If Intel® Extension for TensorFlow* XPU package is installed on CPU only platform, only CPU backend will be loaded. Please refer to **Intel CPU Software Installation** [Hardware Requirements](./experimental/install_for_cpu.md#hardware-requirements) and [Software Requirements](./experimental/install_for_cpu.md#software-requirements) for the platform requirements.

Verify the Installation
```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```

Then, you can get the information that only CPU backend is loaded successfully from the console log.

```
2023-07-28 12:00:00.205706: I itex/core/wrapper/itex_cpu_wrapper.cc:42] Intel Extension for Tensorflow* AVX512 CPU backend is loaded.
2023-07-28 12:00:00.313231: E itex/core/wrapper/itex_gpu_wrapper.cc:49] Could not load Intel Extension for Tensorflow* GPU backend, GPU will not be used.
If you need help, create an issue at https://github.com/intel/intel-extension-for-tensorflow/issues
```
