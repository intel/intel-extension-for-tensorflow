# Intel GPU Software Installation 

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Max Series, Driver Version: [602](https://dgpu-docs.intel.com/releases/stable_602_20230323.html)
 - Intel® Data Center GPU Flex Series 170, Driver Version: [602](https://dgpu-docs.intel.com/releases/stable_602_20230323.html)
 - *Experimental:* Intel® Arc™ A-Series

For experimental support of the Intel® Arc™ A-Series GPUs, please refer to [Intel® Arc™ A-Series GPU Software Installation](experimental/install_for_arc_gpu.md) for details.

## Software Requirements
- Ubuntu 22.04, RedHat 8.6 (64-bit)
  - Intel® Data Center GPU Flex Series 
- Ubuntu 22.04, RedHat 8.6 (64-bit), SUSE Linux Enterprise Server(SLES) 15 SP3/SP4
  - Intel® Data Center GPU Max Series 
- Intel® oneAPI Base Toolkit 2023.1
- TensorFlow 2.12.0
- Python 3.8-3.10
- pip 19.0 or later (requires manylinux2014 support)


## Install GPU Drivers

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.2.0|Ubuntu 22.04, RedHat 8.6|Intel® Data Center GPU Flex Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-flex-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [602](https://dgpu-docs.intel.com/releases/stable_602_20230323.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd==23.05.25593.18-601~22.04`|
|v1.2.0|Ubuntu 22.04, RedHat 8.6, SLES 15 SP3/SP4|Intel® Data Center GPU Max Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [602](https://dgpu-docs.intel.com/releases/stable_602_20230323.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd==23.05.25593.18-601~22.04`|

## Install via Docker container

The Docker container includes the Intel® oneAPI Base Toolkit, and all other software stack except Intel GPU Drivers. User only needs to install the GPU driver in host machine bare metal environment, and then launch the docker container directly. 

#### Build Docker container from Dockerfile

Run the following [Dockerfile build procedure](./../../docker/README.md) to build the pip based deployment container.

#### Get docker container from dockerhub

Pre-built docker images are available at [DockerHub](https://hub.docker.com/r/intel/intel-extension-for-tensorflow/tags).
Please run the following command to pull Intel® Extension for TensorFlow* Docker container image (`gpu`) to your local machine.

```
$ docker pull intel/intel-extension-for-tensorflow:gpu
$ docker run -it -p 8888:8888 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path intel/intel-extension-for-tensorflow:gpu
```

To use Intel® Optimization for Horovod* with the Intel® oneAPI Collective Communications Library (oneCCL), pull Intel® Extension for TensorFlow* Docker container image (`gpu-horovod`) to your local machine by the following command.

```
$ docker pull intel/intel-extension-for-tensorflow:gpu-horovod
$ docker run -it -p 8888:8888 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host intel/intel-extension-for-tensorflow:gpu-horovod
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
$ wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7deeaac4-f605-4bcf-a81b-ea7531577c61/l_BaseKit_p_2023.1.0.46401_offline.sh
# 3 components are necessary: DPC++/C++ Compiler, DPC++ Libiary and oneMKL
# if you want to run distributed training with Intel® Optimization for Horovod*, oneCCL is needed too(Intel® oneAPI MPI Library will be installed automatically as its dependency)
$ sudo sh ./l_BaseKit_p_2023.1.0.46401_offline.sh
```

For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

#### Setup environment variables
```bash
# DPC++ Compiler/oneMKL/tbb
source /path to basekit/intel/oneapi/compiler/latest/env/vars.sh
source /path to basekit/intel/oneapi/mkl/latest/env/vars.sh

# oneCCL (and Intel® oneAPI MPI Library as its dependency), required by Intel® Optimization for Horovod* only
source /path to basekit/intel/oneapi/mpi/latest/env/vars.sh
source /path to basekit/intel/oneapi/ccl/latest/env/vars.sh
```

A user may install more components than Intel® Extension for TensorFlow* needs, and if required, `setvars.sh` can be customized to point to a specific directory by using a [configuration file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html):

```bash
source /opt/intel/oneapi/setvars.sh --config="full/path/to/your/config.txt"
```

#### Install TensorFlow

The Python development and virtual environment setup recommendation by TensorFlow to isolate package installation from the system.

The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be == 2.12.0.


###### Virtual environment install 

You can follow the instructions in [stock tensorflow install](https://www.tensorflow.org/install/pip#step-by-step_instructions) to activate the virtual environment.

On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
```bash
(tf)$ pip install --upgrade pip
```

To install in virtual environment, you can run 
```bash
(tf)$ pip install tensorflow==2.12.0
```

###### System environment install 
If want to system install in $HOME, please append `--user` to the commands.
```bash
$ pip install --user tensorflow==2.12.0
```
And the following system environment install for Intel® Extension for TensorFlow* will use the same practice. 

#### Install Intel® Extension for TensorFlow*

To install a GPU-only version in virtual environment, which depends on Intel GPU drivers and oneAPI BaseKit, you can run

```bash
(tf)$ pip install intel-extension-for-tensorflow[gpu]==1.2.0
```

##### Check the Environment for GPU
```bash
(tf)$ bash /path to site-packages/intel_extension_for_tensorflow/tools/env_check.sh
```

##### Verify the Installation 
```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```
