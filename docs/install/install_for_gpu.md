# Intel GPU Software Installation 

## Hardware Requirements

Verified Hardware Platforms:
 - Intel® Data Center GPU Flex Series 170
 - Intel® Data Center GPU Max Series

## Software Requirements

- Ubuntu 20.04 (64-bit) or SUSE Linux Enterprise Server 15 SP3
- Intel GPU Drivers 
  - For Intel® Data Center GPU Flex Series 170: Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html)
  - For Intel® Data Center GPU Max Series: agama-ci-devel-pvc-prq-54
- Intel® oneAPI Base Toolkit 2022.3
- TensorFlow 2.10.0
- Python 3.7-3.10
- pip 19.0 or later (requires manylinux2014 support)

  
## Install GPU Drivers

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.0.0|Ubuntu 20.04|Intel® Data Center GPU Flex Series| Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html) for latest driver installation. If install the verified Intel® Data Center GPU Flex Series [419.40](https://dgpu-docs.intel.com/releases/stable_419_40_20220914.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd=22.28.23726.1+i419~u20.04`|
|v1.0.0|SLES 15 SP3|Intel® Data Center GPU Max Series| Download intel GPU driver "agama-ci-devel-pvc-prq-54" for Intel® Data Center GPU Max Series from https://ubit-gfx.intel.com/build/15136536/artifacts, and install them. |



## Install via Docker container

The Docker container includes the Intel® oneAPI Base Toolkit, and all other software stack except Intel GPU Drivers. User only needs to install the GPU driver in host machine bare metal environment, and then launch the docker container directly. 



### Build Docker container from Dockerfile

Run the following [Dockerfile build procedure](./../../docker/README.md) to build the pip based deployment container.



### Get docker container from dockerhub(For Intel® Data Center GPU Flex Series)

Pre-built docker images are available at [DockerHub](https://hub.docker.com/r/intel/intel-extension-for-tensorflow/tags).
Please run the following command to pull the GPU Docker container image to your local machine.

```bash
$ docker pull intel/intel-extension-for-tensorflow:gpu
$ docker run -it -p 8888:8888 --device /dev/dri intel/intel-extension-for-tensorflow:gpu
```
Then go to your browser on http://localhost:8888/



### Get docker container from harhor(For Intel internal evaluation on Intel® Data Center GPU Max Series only)

Pre-built docker images are available at [Harbor](https://ccr-registry.caas.intel.com/harbor/projects/519/repositories/gpu-max) as well.
Run the following command to pull the GPU Docker container image to your local machine.

```bash
$ docker pull ccr-registry.caas.intel.com/intel-extension-for-tensorflow/gpu-max
$ docker run -it -p 8888:8888 -v /dev/dri/by-path:/dev/dri/by-path --device /dev/dri ccr-registry.caas.intel.com/intel-extension-for-tensorflow/gpu-max
```
Then go to your browser on http://localhost:8888/



## Install via PyPI wheel in bare metal

### Install oneAPI Base Toolkit Packages

Need to install components of Intel® oneAPI Base Toolkit:

```bash
$ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767_offline.sh
$ sudo sh ./l_BaseKit_p_2022.3.0.8767_offline.sh
```

For any more details, please follow the procedure in https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html.

### Setup environment variables
```bash
source /opt/intel/oneapi/compiler/env/vars.sh
source /opt/intel/oneapi/mkl/env/vars.sh
source /opt/intel/oneapi/tbb/env/vars.sh
```

### Install TensorFlow

The Python development and virtual environment setup recommendation by TensorFlow to isolate package installation from the system.

The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be == 2.10.0. 


#### Virtual environment install 

You can follow the instructions in [stock tensorflow install](https://www.tensorflow.org/install/pip#step-by-step_instructions) to activate the virtual environment.

On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
```bash
(tf)$ pip install --upgrade pip
```

To install in virtual environment, you can run 
```bash
(tf)$ pip install tensorflow==2.10.0
```

#### System environment install 
If want to system install in $HOME, please append `--user` to the commands.
```bash
(tf)$ pip install --user tensorflow==2.10.0
```
And the following system environment install for Intel® Extension for TensorFlow* will use the same practice. 

### Install Intel® Extension for TensorFlow*

- for Intel® Data Center GPU Flex Series 170

To install a GPU-only version in virtual environment, which depends on Intel GPU drivers and oneAPI BaseKit, you can run

```bash
(tf)$ pip install intel-extension-for-tensorflow[gpu]==1.0.0
```



- for Intel® Data Center GPU Max Series

```bash
(tf)$ pip install http://mlpc.intel.com/downloads/gpu-new/releases/PVC_NDA_2022ww45/ITEX/RC3/intel_extension_for_tensorflow-1.0.0-cp39-cp39-linux_x86_64.whl http://mlpc.intel.com/downloads/gpu-new/releases/PVC_NDA_2022ww45/ITEX/RC3/intel_extension_for_tensorflow_lib-1.0.0.1-cp39-cp39-linux_x86_64.whl
```



### Verify the Installation 

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



