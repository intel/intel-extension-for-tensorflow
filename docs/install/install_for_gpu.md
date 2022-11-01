# Intel GPU Software Installation 

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170

## Software Requirements

- Ubuntu 20.04 (64-bit)
- Intel GPU Drivers 
  - Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
- Intel® oneAPI Base Toolkit 2022.3
- TensorFlow 2.10.0
- Python 3.7-3.10
- pip 19.0 or later (requires manylinux2014 support)

  
## Install GPU Drivers

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.0.0|Ubuntu 20.04|Intel® Data Center GPU Flex Series| Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest driver installation. If install the verified Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|

## Install via Docker container

The Docker container includes the Intel® oneAPI Base Toolkit, and all other software stack except Intel GPU Drivers. User only needs to install the GPU driver in host machine bare metal environment, and then launch the docker container directly. 

#### Build Docker container from Dockerfile

Run the following [Dockerfile build procedure](./../../docker/README.md) to build the pip based deployment container.

#### Get docker container from dockerhub

Pre-built docker images are available at [DockerHub](https://hub.docker.com/r/intel/intel-extension-for-tensorflow/tags).
Please run the following command to pull the GPU Docker container image to your local machine.

```
$ docker pull intel/intel-extension-for-tensorflow:gpu
$ docker run -it -p 8888:8888 --device /dev/dri intel/intel-extension-for-tensorflow:gpu
```
Then go to your browser on http://localhost:8888/

## Install via PyPI wheel in bare metal

#### Install oneAPI Base Toolkit Packages

Need to install components of Intel® oneAPI Base Toolkit:
- Intel® oneAPI DPC++ Compiler
- Intel® oneAPI Threading Building Blocks (oneTBB)
- Intel® oneAPI Math Kernel Library (oneMKL)

Download and install the verified DPC++ compiler and oneMKL in Ubuntu 20.04.

```bash
$ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767_offline.sh
# 4 components are necessary: DPC++/C++ Compiler, DPC++ Libiary, Threading Building Blocks and oneMKL
$ sudo sh ./l_BaseKit_p_2022.3.0.8767_offline.sh
```

For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

### Setup environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

A user may install more components than Intel® Extension for TensorFlow* needs, and if required, `setvars.sh` can be customized to point to a specific directory by using a [configuration file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html):

```bash
source /opt/intel/oneapi/setvars.sh --config="full/path/to/your/config.txt"
```

#### Install TensorFlow

The Python development and virtual environment setup recommendation by TensorFlow to isolate package installation from the system.

The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be == 2.10.0. 


##### Virtual environment install 

You can follow the instructions in [stock tensorflow install](https://www.tensorflow.org/install/pip#step-by-step_instructions) to activate the virtual environment.

On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
```bash
(tf)$ pip install --upgrade pip
```

To install in virtual environment, you can run 
```bash
(tf)$ pip install tensorflow==2.10.0
```

##### System environment install 
If want to system install in $HOME, please append `--user` to the commands.
```bash
$ pip install --user tensorflow==2.10.0
```
And the following system environment install for Intel® Extension for TensorFlow* will use the same practice. 

#### Install Intel® Extension for TensorFlow*

To install a GPU-only version in virtual environment, which depends on Intel GPU drivers and oneAPI BaseKit, you can run

```bash
(tf)$ pip install --upgrade intel-extension-for-tensorflow[gpu]
```

##### Verify the Installation 
```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```
