# Build from Source Code

This guide shows how to build an Intel® Extension for TensorFlow* PyPI package from source and install it in Ubuntu 20.04 (64-bit).


## Prepare

### Hardware Requirement

Verified Hardware Platforms:
 - [Intel® Data Center GPU Flex Series 170](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/flex-series/overview.html)
 - [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/docs/processors/max-series/overview.html)

### Python

Python 3.8-3.11

### Intel GPU Driver

An Intel GPU driver is needed to build with GPU support and **AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html))**.

Refer to [Install Intel GPU driver](install_for_gpu.md).

Note: Please make sure to [install developer run-time packages](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-focal-dc.html#optional-install-developer-packages) before building Intel® Extension for TensorFlow*.


### TensorFlow

Install TensorFlow 2.12, and refer to [Install TensorFlow](https://www.tensorflow.org/install).

Check TensorFlow version:
```
python -c "import tensorflow as tf;print(tf.__version__)"
```

### Install oneAPI Base Toolkit

Refer to [Install oneAPI Base Toolkit Packages](install_for_gpu.md#install-oneapi-base-toolkit-packages)

### Install Bazel

To build Intel® Extension for TensorFlow*, install Bazel 5.3.0 or later ones. Refer to [install Bazel](https://docs.bazel.build/versions/main/install-ubuntu.html).

Here are the recommended commands:

```
$ wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
$ bash bazel-5.3.0-installer-linux-x86_64.sh --user
```
Check Bazel:
```bash
bazel --version
```

### Download the Intel® Extension for TensorFlow* source code

```bash
$ git clone https://github.com/intel/intel-extension-for-tensorflow.git intel-extension-for-tensorflow
$ cd intel-extension-for-tensorflow
```

Change to release branch (Optional):

The repo defaults to the `master` development branch. You can also check out a release branch to build:

```bash
$ git checkout branch_name
```


## Configure the build

Configure your system build by running the `./configure` command at the root of your Intel® Extension for TensorFlow* source tree.  This script prompts you for the location of Intel® Extension for TensorFlow* dependencies and asks for additional build configuration options (path to DPC++ compiler, for example).

```
./configure
```

### Sample configuration session

- For GPU

```
You have bazel 5.3.0 installed.
Python binary path: /path/to/python

Found possible Python library paths:
['/path/to/python/site-packages']

Do you wish to build Intel® Extension for TensorFlow* with GPU support? [Y/n]:
GPU support will be enabled for Intel® Extension for TensorFlow*.

Please specify the location where DPC++ is installed. [Default is /opt/intel/oneapi/compiler/latest/linux/]: /path/to/DPC++


Please specify the Ahead of Time(AOT) compilation platforms, separate with "," for multi-targets. [Default is ]: ats-m150


Do you wish to build Intel® Extension for TensorFlow* with MKL support? [y/N]: y
MKL support will be enabled for Intel® Extension for TensorFlow*.

Please specify the MKL toolkit folder. [Default is /opt/intel/oneapi/mkl/latest]: /path/to/oneMKL


Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=gpu            # Build Intel® Extension for TensorFlow* with GPU support.
Configuration finished
```



- For CPU

```
You have bazel 5.3.0 installed.
Python binary path: /path/to/python

Found possible Python library paths:
['/path/to/python/site-packages']

Do you wish to build Intel® Extension for TensorFlow* with GPU support? [Y/n]: N
No GPU support will be enabled for Intel® Extension for TensorFlow*.

Only CPU support is available for Intel® Extension for TensorFlow*.
Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=cpu            # Build Intel® Extension for TensorFlow* with CPU support.
Configuration finished
```



### Configuration options explanation

**GPU support**

Set "build Intel® Extension for TensorFlow* with GPU support" to `Y` if you want to build with Intel GPU support. Set to `N` if build for CPU only. Default is `Y`.



**Location of DPC++**

The path to DPC++ compiler. Default is `/opt/intel/oneapi/compiler/latest/linux/`.



**AOT ([Ahead-of-time compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html))**

An optional option benefits from the initialization time of GPU kernels at runtime, but increases the binary size on the other hand, which requires GPU driver to be ready on building machine.

Reference for AOT compilation platforms:

| GPU card                               | AOT target | Comments                                                     |
| -------------------------------------- | ---------- | ------------------------------------------------------------ |
| Intel® Data Center GPU Flex Series 170 | `ats-m150` | The Intel® Data Center GPU Flex Series (formerly code-named Arctic Sound-M) with high-power, 150W adapter. |
| Intel® Data Center GPU Max Series | `pvc` | Intel® Data Center GPU Max Series (formerly code-named Ponte Vecchio). |

For more GPU platforms, please refer to [Use AOT for Integrated Graphics (Intel GPU)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html).


## Build the pip package



### GPU support

```bash
$ bazel build -c opt --config=gpu  //itex/tools/pip_package:build_pip_package
```



### CPU only (experimental)

```bash
$ bazel build -c opt --config=cpu  //itex/tools/pip_package:build_pip_package
```



### Build the package

```bash
$ bazel-bin/itex/tools/pip_package/build_pip_package ~/
```
It will generate two wheels: 
- intel_extension_for_tensorflow-*.whl
- intel_extension_for_tensorflow_lib-*.whl

The Intel_extension_for_tensorflow_lib will differentiate between the CPU version or the GPU version
- CPU version identifier is {ITEX_VERSION}**.0**
- GPU version identifier is {ITEX_VERSION}**.1**

For example

|ITEX version|ITEX-lib CPU version|ITEX-lib GPU version|
|------------|--------------------|--------------------|
|1.0.0       |1.0.0.0             |1.0.0.1             |

### Install the package

```bash
$ pip install ~/intel_extension_for_tensorflow*.whl
```
or
```bash
$ pip install ~/intel_extension_for_tensorflow-*.whl
$ pip install ~/intel_extension_for_tensorflow_lib-*.whl
```

### Installation package directory

- located at `path/to/site-packages/`

```
├── intel_extension_for_tensorflow
|   ├── libitex_common.so
│   └── python
│       └── _pywrap_itex.so
├── intel_extension_for_tensorflow_lib
├── tensorflow
├── tensorflow-plugins
|   ├── libitex_cpu.so # for CPU-only build
│   └── libitex_gpu.so # for GPU build

```
## Uninstall

```bash
$ pip uninstall intel_extension_for_tensorflow_lib
$ pip uninstall intel_extension_for_tensorflow
```

