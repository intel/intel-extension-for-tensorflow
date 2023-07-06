# Build from Source Code

This guide shows how to build an Intel® Extension for TensorFlow* PyPI package from source and install it in Ubuntu 22.04 (64-bit).

When will you need to build from source code?

1. You want to get the latest feature in development branch.

2. You want to develop a feature or contribute to Intel® Extension for TensorFlow*.

3. Verify your code update.

## Prepare

### Hardware Requirement

Verified Hardware Platforms:
 - Intel® CPU (Xeon, Core)
 - [Intel® Data Center GPU Flex Series](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html)
 - [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/docs/processors/max-series/overview.html)
 - [Intel® Arc™ Graphics](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html) (experimental)

### Python Running Environment

1. Conda

Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Create Virtual Running Environment

```
conda create -n itex_build python=3.9
conda activate itex_build
```
Note, support Python 3.8-3.10.

### Intel GPU Driver (Optional, GPU and XPU)

Install the Intel GPU Driver in the building server, which is needed to build with GPU support and **AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html))**.

Refer to [Install Intel GPU driver](install_for_gpu.md).

Note:

1. Make sure to [install developer runtime packages](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-dc.html#optional-install-developer-packages) before building Intel® Extension for TensorFlow*.

2. **AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html))**

AOT is option of compiling, which reduces the initialization time of GPU kernels at startup time, by creating the binary code for specified hardware platform directly during compiling. AOT will make the installation package be with bigger size.

Without AOT, Intel® Extension for TensorFlow* will be translated to binary code for local hardware platform during startup. That will prolong startup time to several minutes or more.

For more info, please refer to [Use AOT for Integrated Graphics (Intel GPU)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html).


### TensorFlow

Install TensorFlow 2.12, and refer to [Install TensorFlow](https://www.tensorflow.org/install).

```
pip install tensorflow==2.12
```
Check TensorFlow version:
```
python -c "import tensorflow as tf;print(tf.__version__)"
```

For detail, refer to [Install TensorFlow](https://www.tensorflow.org/install)

### Install oneAPI Base Toolkit

We recommend to install oneAPI by 'sudo or root' to folder **/opt/intel/oneapi**.

Following commands are based on the folder **/opt/intel/oneapi**. If you install it in other folder, please change the oneAPI path as yours.

Refer to [Install oneAPI Base Toolkit Packages](install_for_gpu.md#install-oneapi-base-toolkit-packages)

It provides compiler and libraries used by Intel® Extension for TensorFlow*.

Enable oneAPI components:

```
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/mkl/latest/env/vars.sh
```

### Install Bazel


To build Intel® Extension for TensorFlow*, install Bazel 5.3.0. Refer to [install Bazel](https://docs.bazel.build/versions/main/install-ubuntu.html).

Here are the recommended commands:

```
$ wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
$ bash bazel-5.3.0-installer-linux-x86_64.sh --user

```
Check Bazel:
```bash
bazel --version
```


### Download the Intel® Extension for TensorFlow* Source Code

```bash
$ git clone https://github.com/intel/intel-extension-for-tensorflow.git intel-extension-for-tensorflow
$ cd intel-extension-for-tensorflow
```

Change to special release/tag (Optional):

The repo defaults to the `master` development branch. You can also check out a release branch or tag to build:

```bash
$ git checkout branch_name/tag_name
```


## Configure

Configure the system build by running the `./configure` command at the root of your Intel® Extension for TensorFlow* source tree.  This script prompts you for the location of Intel® Extension for TensorFlow* dependencies and asks for additional build configuration options (path to DPC++ compiler, for example).

```
./configure
```

### Choose to Build with GPU Support.

'Y' for GPU support; 'N' for CPU only.

### Specify the Location of Compiler (DPC++).

Default is **/opt/intel/oneapi/compiler/latest/linux/**, which is the default installed path. Click **enter** to confirm default location.

If it's differenct, confirm the compiler (DPC++) installed path and fill the correct path.


### Specify the Ahead of Time (AOT) Compilation Platforms.

Default is '', which means no AOT.

Fill one or more device type strings of special hardware platforms, like 'ats-m150,acm-g11'.

Here is the list of GPUs verified:

|GPU|device type|
|-|-|
|Intel® Data Center GPU Flex Series 170|ats-m150|
|Intel® Data Center GPU Flex Series 140|ats-m75|
|Intel® Data Center GPU Max Series|pvc|
|Intel® Arc™ A730M|acm-g10|
|Intel® Arc™ A380|acm-g11|

To learn how to get the device type, please refer to [Use AOT for Integrated Graphics (Intel GPU)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html) or create an [issue](https://github.com/intel/intel-extension-for-tensorflow/issues) to ask support.

### Choose to Build with oneMKL Support.

Recommend to choose 'y'.

Default is **/opt/intel/oneapi/mkl/latest**, which is the default installed path. Click **enter** to confirm default location.

If it's wrong, please confirm the oneMKL installed path and fill the correct path.

### Example

Please refer to [Configure Example](#configure-example).

## Build the Pip Package

### Build Source Code

For GPU:

```bash
$ bazel build -c opt --config=gpu  //itex/tools/pip_package:build_pip_package
```

For XPU:

```bash
$ bazel build -c opt --config=xpu  //itex/tools/pip_package:build_pip_package
```

For CPU only (experimental):

```bash
$ bazel build -c opt --config=cpu  //itex/tools/pip_package:build_pip_package
```

### Create the Pip Package

```bash
$ bazel-bin/itex/tools/pip_package/build_pip_package ./
```
It will generate two wheels:
- intel_extension_for_tensorflow-*.whl
- intel_extension_for_tensorflow_lib-*.whl

The Intel_extension_for_tensorflow_lib will differentiate between the CPU version or the GPU version
- CPU version identifier is {ITEX_VERSION}**.0**
- GPU version identifier is {ITEX_VERSION}**.1**
- XPU version identifier is {ITEX_VERSION}**.2**

For example

|ITEX version|ITEX-lib CPU version|ITEX-lib GPU version|
|------------|--------------------|--------------------|
|1.x.0       |1.x.0.0             |1.x.0.1             |

### Install the Package

```bash
$ pip install ./intel_extension_for_tensorflow*.whl
```
or
```bash
$ pip install ./intel_extension_for_tensorflow-*.whl
$ pip install ./intel_extension_for_tensorflow_lib-*.whl
```

### Installation Package Directory

- located at `path/to/site-packages/`

```
├── intel_extension_for_tensorflow
|   ├── libitex_common.so
│   └── python
│       └── _pywrap_itex.so
├── intel_extension_for_tensorflow_lib
├── tensorflow
├── tensorflow-plugins
|   ├── libitex_cpu.so # for CPU or XPU build
│   └── libitex_gpu.so # for GPU or XPU build

```
## Uninstall

```bash
$ pip uninstall intel_extension_for_tensorflow_lib
$ pip uninstall intel_extension_for_tensorflow
```

## Addtional

### Configure Example

- For GPU or XPU

```
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
        --config=gpu            # Build Intel® Extension for TensorFlow* with GPU support.
Configuration finished
```



- For CPU

```
You have bazel 5.3.0 installed.
Python binary path: /path/to/envs/itex_build/bin/python

Found possible Python library paths:
['/path/to/envs/itex_build/lib/python3.9/site-packages']

Do you wish to build Intel® Extension for TensorFlow* with GPU support? [Y/n]: n
No GPU support will be enabled for Intel® Extension for TensorFlow*.

Only CPU support is available for Intel® Extension for TensorFlow*.
Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=cpu            # Build Intel® Extension for TensorFlow* with CPU support.
Configuration finished
```

