
- [Overview](#overview)
- [Requirements](#requirements)
  - [Hardware Requirements](#hardware-requirements)
  - [Common Requirements](#common-requirements)
    - [Install Bazel](#install-bazel)
    - [Download Source Code](#download-source-code)
    - [Create a Conda Environment](#create-a-conda-environment)
    - [Install Tensorflow](#install-tensorflow)
  - [Extra Requirements for XPU Build Only](#extra-requirements-for-xpugpu-build-only)
    - [Install Intel GPU Driver](#install-intel-gpu-driver)
    - [Install OneAPI Base Toolkit](#install-oneapi-base-toolkit)
  
- [Build Intel® Extension for TensorFlow* PyPI](#build-intel®-extension-for-tensorflow-pypi)
  - [Configure](#configure)
    - [Configure For CPU](#configure-for-cpu)
    - [Configure For XPU](#configure-for-gpuxpu)
  - [Build Source Code](#build-source-code)
- [Additional](#additional)
  - [Configure Example For CPU](#configure-example-for-gpu-or-xpu)
  - [Configure Example For XPU](#configure-example-for-gpu-or-xpu)


## Overview
This guide shows how to build an Intel® Extension for TensorFlow* PyPI package from source and install it in Ubuntu 22.04 (64-bit).

Normally, you would install the latest released version of Intel® Extension for TensorFlow* using a `pip install` command. There are times though when you might need to build from source code:

1. You want to get the latest feature in development branch.

2. You want to develop a feature or contribute to Intel® Extension for TensorFlow*.

3. Verify your code update.

## Requirements

### Hardware Requirements

Verified Hardware Platforms:
 - Intel® CPU (Xeon, Core)
 - [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html)
 - [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/docs/processors/max-series/overview.html)
 - [Intel® Arc™ Graphics](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) (experimental)

### Common Requirements

#### Install Bazel

To build Intel® Extension for TensorFlow*, install Bazel 5.3.0. Refer to [install Bazel](https://docs.bazel.build/versions/main/install-ubuntu.html).

Here are the recommended commands:

```bash
$ wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
$ bash bazel-5.3.0-installer-linux-x86_64.sh --user
```

Check Bazel is installed successfully and is version 5.3.0:

```bash
$ bazel --version
```

#### Download Source Code

```bash
$ git clone https://github.com/intel/intel-extension-for-tensorflow.git intel-extension-for-tensorflow
$ cd intel-extension-for-tensorflow/
```

#### Create a Conda Environment

1. Install [Conda](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).

2. Create Virtual Running Environment

```bash
$ conda create -n itex_build python=3.10
$ conda activate itex_build
```

Note, we support Python versions 3.9 through 3.11.

#### Install TensorFlow

Install TensorFlow 2.15.0, and refer to [Install TensorFlow](https://www.tensorflow.org/install) for details.

```bash
$ pip install tensorflow==2.15.0
```

Check TensorFlow was installed successfully and is version 2.15.0:

```bash
$ python -c "import tensorflow as tf;print(tf.__version__)"
```

### Optional Requirements for CPU Build Only

#### Install Clang-17 compiler
ITEX CPU uses clang-17 as default compiler instead of gcc. Users can switch back to gcc in [configure](####configure-for-cpu). Clang-17 can be installed through apt on Ubuntu or source build on other systems. Check https://apt.llvm.org/ for more details.
```bash
# Ubuntu 22.04
# Add in /etc/apt/sources.list
deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main
deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy main
# 17
deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main
deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main

$ wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

$ apt update
$ apt-get install clang-17 lldb-17 lld-17
$ apt-get install libomp-17-dev
```

To source build clang-17, use the following command from https://llvm.org/docs/CMake.html:
```bash
# Cmake minimum version 3.20.0
$ mkdir mybuilddir
$ cd mybuilddir
$ cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lldb;lld;openmp" path/to/llvm-17/
$ cmake --build . --parallel 100 
$ cmake --build . --target install
```

### Extra Requirements for XPU Build Only

#### Install Intel GPU Driver
Install the Intel GPU Driver in the building server, which is needed to build with GPU support and AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html)).

Refer to [Install Intel GPU driver](install_for_xpu.md/#install-gpu-drivers) for details.

Note:

1. Make sure to [install developer runtime packages](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#optional-install-developer-packages) before building Intel® Extension for TensorFlow*.

2. **AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html))**

    AOT is a compiling option that reduces the initialization time of GPU kernels at startup time by creating the binary code for a specified hardware platform during compiling. AOT will make the installation package larger but improve performance time.

    Without AOT, Intel® Extension for TensorFlow* will be translated to binary code for local hardware platform during startup. That will prolong startup time when using a GPU to several minutes or more.

    For more information, refer to [Use AOT for Integrated Graphics (Intel GPU)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html).

#### Install oneAPI Base Toolkit

We recommend you install the oneAPI base toolkit using `sudo` (or as root user) to the system directory `/opt/intel/oneapi`.

The following commands assume the oneAPI base tookit is installed in `/opt/intel/oneapi`. If you installed it in some other folder, please update the oneAPI path as appropriate.

Refer to [Install oneAPI Base Toolkit Packages](install_for_xpu.md#install-oneapi-base-toolkit-packages)

The oneAPI base toolkit provides compiler and libraries needed by Intel® Extension for TensorFlow*.

Enable oneAPI components:

```bash
$ source /opt/intel/oneapi/compiler/latest/env/vars.sh
$ source /opt/intel/oneapi/mkl/latest/env/vars.sh
```

## Build Intel® Extension for TensorFlow* PyPI

### Configure

#### Configure For CPU

Configure the system build by running the `./configure` command at the root of your cloned Intel® Extension for TensorFlow* source tree.

```bash
$ ./configure
```

First to choose `n` to build for CPU only, next to choose compiler: `Y' for clang and `n` for gcc. Refer to [Configure Example](#configure-example-for-cpu).

#### Configure For XPU

Configure the system build by running the `./configure` command at the root of your cloned Intel® Extension for TensorFlow* source tree. This script prompts you for the location of Intel® Extension for TensorFlow* dependencies and asks for additional build configuration options (path to DPC++ compiler, for example).

```bash
$ ./configure
```

- Choose `Y` for Intel GPU support. Refer to [Configure Example](#configure-example-for-gpu-or-xpu).

- Specify the Location of Compiler (DPC++).

  Default is `/opt/intel/oneapi/compiler/latest/linux/`, which is the default installed path. Click `Enter` to confirm default location.

  If it's differenct, confirm the compiler (DPC++) installed path and fill the correct path.

- Specify the Ahead of Time (AOT) Compilation Platforms.

  Default is '', which means no AOT.

  Fill one or more device type strings of special hardware platforms, such as `ats-m150`, `acm-g11`.

  Here is the list of GPUs we've verified:

  |GPU|device type|
  |-|-|
  |Intel® Data Center GPU Flex Series 170|ats-m150|
  |Intel® Data Center GPU Flex Series 140|ats-m75|
  |Intel® Data Center GPU Max Series|pvc|
  |Intel® Arc™ A730M|acm-g10|
  |Intel® Arc™ A380|acm-g11|

  To learn how to get the device type, please refer to [Use AOT for Integrated Graphics (Intel GPU)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html) or create an [issue](https://github.com/intel/intel-extension-for-tensorflow/issues) to ask support.

- Choose to Build with oneMKL Support.

  We recommend choosing `y`.

  Default is `/opt/intel/oneapi/mkl/latest`, which is the default installed path. Click `Enter` to confirm default location.

  If it's wrong, please confirm the oneMKL installed path and fill the correct path.

### Build Source Code

For CPU:

```bash
$ bazel build -c opt --config=cpu  //itex/tools/pip_package:build_pip_package
```

For XPU:

```bash
$ bazel build -c opt --config=xpu  //itex/tools/pip_package:build_pip_package
```

Create the Pip Package

```bash
$ bazel-bin/itex/tools/pip_package/build_pip_package WHL/
```

It will generate two wheels under `WHL` directory:
- intel_extension_for_tensorflow-*.whl
- intel_extension_for_tensorflow_lib-*.whl

The Intel_extension_for_tensorflow_lib will differentiate between the CPU version or the xpu version
- CPU version identifier is {ITEX_VERSION}**.0**
- GPU version identifier is {ITEX_VERSION}**.1** (**Deprecated, duplicates of XPU version**)
- XPU version identifier is {ITEX_VERSION}**.2**

For example

|ITEX version|ITEX-lib CPU version|ITEX-lib XPU version|
|------------|--------------------|--------------------|
|1.x.0       |1.x.0.0             |1.x.0.2             |

Install the Package

```bash
$ pip install ./intel_extension_for_tensorflow*.whl
```

or

```bash
$ pip install ./intel_extension_for_tensorflow-*.whl
$ pip install ./intel_extension_for_tensorflow_lib-*.whl
```

Located at `path/to/site-packages/`

```bash
├── intel_extension_for_tensorflow
|   ├── libitex_common.so
│   └── python
│       └── _pywrap_itex.so
├── intel_extension_for_tensorflow_lib
├── tensorflow
├── tensorflow-plugins
|   ├── libitex_cpu.so # for CPU build
│   └── libitex_gpu.so # for XPU build

```

## Additional

### Configure Example for CPU

Here is example output and interaction you'd see while running the `./configure` script:

```bash
You have bazel 5.3.0 installed.
Python binary path: /path/to/envs/itex_build/bin/python

Found possible Python library paths:
['/path/to/envs/itex_build/lib/python3.9/site-packages']

Do you wish to build Intel® Extension for TensorFlow* with GPU support? [Y/n]: n
No GPU support will be enabled for Intel® Extension for TensorFlow*.

Do you want to use Clang to build ITEX host code? [Y/n]:
Clang will be used to compile ITEX host code.

Please specify the path to clang executable. [Default is /usr/lib/llvm-17/bin/clang]:


You have Clang 17.0.5 installed.

Only CPU support is available for Intel® Extension for TensorFlow*.
Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=cpu            # Build Intel® Extension for TensorFlow* with CPU support.
Configuration finished
```

### Configure Example For XPU

Here is example output and interaction you'd see while running the `./configure` script:

```bash
You have bazel 5.3.0 installed.
Python binary path: /path/to/envs/itex_build/bin/python

Found possible Python library paths:
['/path/to/envs/itex_build/lib/python3.9/site-packages']

Do you wish to build Intel® Extension for TensorFlow* with GPU support? [Y/n]:y
GPU support will be enabled for Intel® Extension for TensorFlow*.

Please specify the location where DPC++ is installed. [Default is /opt/intel/oneapi/compiler/latest/linux/]: /path/to/DPC++


Please specify the Ahead of Time(AOT) compilation platforms, separate with "," for multi-targets. [Default is ]: ats-m150


Do you wish to build Intel® Extension for TensorFlow* with MKL support? [y/N]:y
MKL support will be enabled for Intel® Extension for TensorFlow*.

Please specify the MKL toolkit folder. [Default is /opt/intel/oneapi/mkl/latest]: /path/to/oneMKL


Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=xpu            # Build Intel® Extension for TensorFlow* with GPU support.
NOTE: XPU mode which supports both CPU and GPU is disbaled."--config=xpu" only supports GPU, which is same as "--config=gpu"

Configuration finished
```