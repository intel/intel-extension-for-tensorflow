# Conda Environment Installation Instructions

This document provides a recommended installation instruction for Intel® Extension for TensorFlow* v2.15.0.0 and Intel® Optimization for Horovod* v0.28.1.4 deployment for distributed training on Conda environment.

## Preconditions
We assume you have installed the Intel GPU driver and the required components of oneAPI Base Toolkit Packages as per [instructions](../install_for_xpu.md#install_oneapi_base_toolkit_packages) successfully.


## Step by step instructions:

**Miniforge** is the community (conda-forge) driven minimalistic conda installer. It is the recommended approach for installing stock TensorFlow. It creates a separate environment to avoid changing any installed software in your system. This is also the easiest way to install the required software especially for the GPU setup.

You can use the following command to install Miniforge. During installation, you may need to press enter and type "yes". Skip this step, if you have already installed conda.

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

You may need to restart your terminal or `source ~/.bashrc` to enable the conda command. Use this command to test if it is installed successfully.
```
conda -V
```

```
conda update conda

#Take Intel Python 2023.2.0 as an example to conda environment, but generic Python is also recommended.
conda create -n itex -c https://software.repos.intel.com/python/conda/ intelpython3_full==2024.2.0 python=3.9
```

Activate the environment by the following commands.
```
conda activate itex
```
Install stock Tensorflow 2.15.0 and Intel® Extension for TensorFlow* XPU wheels.
```
pip install --upgrade pip
pip install tensorflow==2.15.0
pip install intel-extension-for-tensorflow[xpu]
```

Set required environment variables:

```
# DPC++ compiler/oneMKL
source /path/to/intel/oneapi/compiler/latest/env/vars.sh
source /path/to/intel/oneapi/mkl/latest/env/vars.sh
```

Check the environment for GPU:
* Option1:
```bash
(tf)$ pip install wget
(tf)$ export path_to_site_packages=`python -c "import site; print(site.getsitepackages()[0])"`
(tf)$ python ${path_to_site_packages}/intel_extension_for_tensorflow/tools/python/env_check.py
```

* Option2:
```bash
(tf)$ pip install wget
(tf)$ wget https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/main/tools/python/env_check.py
(tf)$ python env_check.py
```

Verify install:
```
python3 -c "import intel_extension_for_tensorflow as itex; print(itex.version.GIT_VERSION)"
```
Expected result:
```
v2.15.0.0-1ccb4fee
```

In order to install Intel® Optimization for Horovod* v0.28.1.4 for distributed training, you should install oneCCL after you install oneAPI Basekit.

```
source /path/to/intel/oneapi/mpi/latest/env/vars.sh
source /path/to/intel/oneapi/ccl/latest/env/vars.sh

pip install intel-optimization-for-horovod
```

Test multi-node with Intel MPI on cluster System, set environment variables.
```
export FI_PROVIDER=sockets
```
