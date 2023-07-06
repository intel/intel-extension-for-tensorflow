# Intel GPU Software Installation 

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170
 - Intel® Data Center GPU Max Series


For experimental support of the Intel® Arc™ A-Series GPUs, please refer to [Intel® Arc™ A-Series GPU Software Installation](experimental/install_for_arc_gpu.md) for details.


## Software Requirements
- Ubuntu 20.04 (64-bit) or SUSE Linux Enterprise Server 15 SP3
- Intel GPU Drivers 
  - For Intel® Data Center GPU Flex Series 170: Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
  - For Intel® Data Center GPU Max Series: agama-ci-devel-pvc-prq-54
- Intel® oneAPI Base Toolkit 2022.3
- TensorFlow 2.10.0
- Python 3.7-3.10


## Install GPU Drivers

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.0.0|Ubuntu 20.04|Intel® Data Center GPU Flex Series| Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest driver installation. If install the verified Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|
|v1.0.0|SLES 15 SP3|Intel® Data Center GPU Max Series| Download intel GPU driver "agama-ci-devel-pvc-prq-54" for Intel® Data Center GPU Max Series from https://ubit-gfx.intel.com/build/15136536/artifacts, and install them. |


## Install via Docker container

The Docker container includes the Intel® oneAPI Base Toolkit, and all other software stack except Intel GPU Drivers. Install the GPU driver in host machine bare metal environment, and then launch the docker container directly. 



### Build Docker container from Dockerfile

Run the following [Dockerfile build procedure](./../../docker/README.md) to build the pip based deployment container.



### Get docker container from dockerhub(For Intel® Data Center GPU Flex Series)

Pre-built docker images are available at [DockerHub](https://hub.docker.com/r/intel/intel-extension-for-tensorflow/tags).
Run the following command to pull Intel® Extension for TensorFlow* Docker container image (`gpu`) to your local machine.

```bash
$ docker pull intel/intel-extension-for-tensorflow:gpu
$ docker run -it -p 8888:8888 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path intel/intel-extension-for-tensorflow:gpu
```
Then go to your browser on http://localhost:8888/


To use Intel® Optimization for Horovod* with the Intel® oneAPI Collective Communications Library (oneCCL), pull Intel® Extension for TensorFlow* Docker container image (`gpu-horovod`) to your local machine by the following command.

```
$ docker pull intel/intel-extension-for-tensorflow:gpu-horovod
$ docker run -it -p 8888:8888 --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host intel/intel-extension-for-tensorflow:gpu-horovod
```

Then go to your browser on http://localhost:8888/



## Install via PyPI wheel in bare metal

### Install oneAPI Base Toolkit Packages

Need to install components of Intel® oneAPI Base Toolkit:


```bash
$ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767_offline.sh
$ sudo sh ./l_BaseKit_p_2022.3.0.8767_offline.sh
```

For any more details, follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

#### Setup environment variables
```bash
source /opt/intel/oneapi/compiler/env/vars.sh
source /opt/intel/oneapi/mkl/env/vars.sh
source /opt/intel/oneapi/tbb/env/vars.sh
# oneCCL (and Intel® oneAPI MPI Library as its dependency), required by Intel® Optimization for Horovod* only
source /path to basekit/intel/oneapi/mpi/latest/env/vars.sh
source /path to basekit/intel/oneapi/ccl/latest/env/vars.sh
```

You may install more components than Intel® Extension for TensorFlow* needs, and if required, `setvars.sh` can be customized to point to a specific directory by using a [configuration file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html):

```bash
source /opt/intel/oneapi/setvars.sh --config="full/path/to/your/config.txt"
```

### Install TensorFlow

The Python development and virtual environment setup recommendation by TensorFlow to isolate package installation from the system.

The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be == 2.12.0.


#### Virtual environment install 

You can follow the instructions in [stock tensorflow install](https://www.tensorflow.org/install/pip#step-by-step_instructions) to activate the virtual environment.

On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
```bash
(tf)$ pip install --upgrade pip
```

To install in virtual environment, you can run 
```bash
(tf)$ pip install --user tensorflow==2.10.0
```

#### System environment install 
If want to system install in $HOME, please append `--user` to the commands.
```bash
(tf)$ pip install --user tensorflow==2.10.0
```
And the following system environment install for Intel® Extension for TensorFlow* will also append `--user` to the command. 

### Install Intel® Extension for TensorFlow*

- for Intel® Data Center GPU Flex Series 170

To install a GPU-only version in virtual environment, which depends on Intel GPU drivers and oneAPI BaseKit, you can run

```bash
(tf)$ pip install intel-extension-for-tensorflow[gpu]==1.0.0
```

- for Intel® Data Center GPU Max Series

```bash
(tf)$ pip install --upgrade intel-extension-for-tensorflow-weekly[gpu]==1.0.0 -f https://developer.intel.com/itex-whl-weekly
```

#### Check the Environment for GPU
```bash
(tf)$ bash /path to site-packages/intel_extension_for_tensorflow/tools/env_check.sh
```

#### Verify the Installation 

```bash
(tf)$ python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```



### Setup environment for distributed training

Distributed training on Intel® Data Center GPU Max Series devices depends on Intel MPI, oneCCL and Intel® Optimization for Horovod*.

- Intel MPI from Intel® oneAPI Base Toolkit
- oneCCL 2021.8-eng03 (For Argonne National Laboratory and Intel internal evaluation only)
- Intel® Optimization for Horovod* 0.22.1up3



#### Setup Intel MPI

```bash
$ source /opt/intel/oneapi/mpi/latest/env/vars.sh
```



#### Setup oneCCL

```bash
$ wget http://mlpc.intel.com/downloads/gpu-new/components/oneCCL/l_ccl_release__2021.8.0_ENG_ww45.20221101.100012.tgz
$ tar -xvzf l_ccl_release__2021.8.0_ENG_ww45.20221101.100012.tgz
$ cd l_ccl_release__2021.8.0_ENG_ww45.20221101.100012

$ unzip -P accept package.zip
$ sh install.sh -s -d $PWD/2021.8-eng03

# setup env vars
$ source {2021.8-eng03 install_dir}/2021.8-eng03/env/vars.sh
```



#### Install Intel® Optimization for Horovod*

```bash
$ pip install http://mlpc.intel.com/downloads/gpu-new/releases/PVC_NDA_2022ww45/ITEX/RC3/intel_optimization_for_horovod-0.22.1up3-cp39-cp39-linux_x86_64.whl
```



