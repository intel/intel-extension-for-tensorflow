# *Experimental:* Intel® Arc™ A-Series GPU Software Installation 


## *Experimental Release*

The Intel® Extension for TensorFlow* has early *experimental only support* for Intel® Arc™ A-Series GPUs on Windows Subsystem for Linux 2 with Ubuntu Linux installed and native Ubuntu Linux.

Issues opened for the Intel® Extension for TensorFlow* on Intel® Arc™ A-Series GPUs will be addressed on a best-effort basis, but no guarantee is provided as to when these issues will be fixed.


## Hardware Requirements

Hardware Platforms with Experimental Only Support:
- Intel® Arc™ A-Series GPUs
 

## Software Requirements

### Windows Subsystem for Linux 2 (WSL2)

- For [Windows 10](https://www.microsoft.com/en-us/windows/get-windows-10) or [Windows 11](https://www.microsoft.com/en-us/windows/windows-11):
    - [Windows Subystem for Linux 2](https://learn.microsoft.com/en-us/windows/wsl/about) (WSL2) with Ubuntu 22.04 (64-bit)
    - Windows GPU Drivers: [Intel® Arc™ Graphics Windows Driver 31.0.101.3490](https://www.intel.com/content/www/us/en/download/726609/intel-arc-graphics-windows-dch-driver.html) or later (installation instructions below)

- For Ubuntu Linux 22.04 within WSL2:
    - Linux Runtime Libraries: Intel® Arc™ GPU Drivers [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html) (installation instructions below)
    - Intel® oneAPI Base Toolkit 2023.0 (installation instructions below)
    - TensorFlow 2.11.0 or 2.10.0
    - Python 3.7-3.10
    - pip 19.0 or later (requires manylinux2014 support)

### Native Linux Running Directly on Hardware

- Ubuntu 22.04 (64-bit)
- Intel® GPU Drivers for Linux (installation instructions below)
    - Intel® Arc™ GPU Drivers [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html)
- Intel® oneAPI Base Toolkit 2023.0 (installation instructions below)

- TensorFlow 2.11.0 or 2.10.0
- Python 3.7-3.10
- pip 19.0 or later (requires manylinux2014 support)

## Step-By-Step Instructions


### 1. Install GPU Drivers

#### Windows Subsystem for Linux 2 (WSL2)

When using WSL2, the GPU drivers are installed in the Windows OS and runtime components such as [Level-Zero](https://github.com/oneapi-src/level-zero) are installed within Linux (in WSL2).


##### Windows GPU Drivers

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.1.0|Windows 10, Windows 11|Intel® Arc™ A-Series GPUs|[Intel® Arc™ Graphics Windows DCH Driver](https://www.intel.com/content/www/us/en/download/726609/intel-arc-graphics-windows-dch-driver.html)|

Install the above Intel® Arc™ Graphics Windows DCH Driver in the Windows OS.


##### Ubuntu Linux Installed in WSL2

|Release|OS|Intel GPU|Install Intel Compute Runtime Components|
|-|-|-|-|
|v1.1.0|Ubuntu 22.04 installed in WSL2|Intel® Arc™ A-Series GPUs|Refer to the instructions below for package installation in Ubuntu 22.04. When installing the Intel® Arc™ A-Series GPU Drivers [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html), please be sure to append the specific version after components, as is done below.|

The steps to install the runtime components in Ubuntu Linux (within WSL2) are:

- Add the repositories.intel.com/graphics package repository to your Ubuntu installation:

    ```bash
    sudo apt-get install -y gpg-agent wget
    wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | sudo tee  /etc/apt/sources.list.d/intel.gpu.jammy.list
    sudo apt-get update
    ```

- Install the necessary runtime packages:

    ```bash
    sudo apt-get install \
        intel-opencl-icd=22.43.24595.35+i538~22.04 \
        intel-level-zero-gpu=1.3.24595.35+i538~22.04 \
        level-zero=1.8.8+i524~u22.04
    ```

- Add the Intel® oneAPI library repositories to your Ubuntu installation:

    ```bash
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
    sudo apt-get update
    ```

- Install the necessary Intel® oneAPI library packages:

    ```bash
    sudo apt-get install intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl
    ```

The above commands install only runtime libraries for Intel® oneAPI which are used by the Intel® Extension for TensorFlow*.  If you would instead prefer to install the full Intel® oneAPI, see section [Optional: Install Full Intel® oneAPI Base Toolkit Packages](#optional-install-full-intel®-oneapi).


#### Native Linux Running Directly on Hardware

|Release|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|-|
|v1.1.0|Ubuntu 22.04|Intel® Arc™ A-Series GPUs| Refer to the instructions below for package installation in Ubuntu 22.04. When installing the Intel® Arc™ A-Series GPU Drivers [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd_22.43.24595.35+i538~22.04_amd64.deb`|

The steps to install the runtime components in Ubuntu Linux are:

- The Intel® Extension for TensorFlow* requires a specific set of drivers for native Linux.  Please follow the instructions in [Installation Guides for Intel Arc GPUs](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-arc.html). When installing the Intel® Arc™ A-Series GPU Drivers [540](https://dgpu-docs.intel.com/releases/stable_540_20221205.html), please be sure to append the specific version after components, such as `sudo apt-get install intel-opencl-icd_22.43.24595.35+i538~22.04_amd64.deb`|

- Install the Intel® oneAPI libraries

    - Add the Intel® oneAPI library repositories to your Ubuntu installation:

        ```bash
        wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
        sudo apt-get update
        ```

       - Install the necessary Intel® oneAPI library packages:

        ```bash
        sudo apt-get install intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl
        ```

The above commands install only runtime libraries for Intel® oneAPI which are used by the Intel® Extension for TensorFlow*.  If you would instead prefer to install the full Intel® oneAPI, see section [Optional: Install Full Intel® oneAPI Base Toolkit Packages](#optional-install-full-intel®-oneapi).


### 2. Install TensorFlow* via PyPI Wheel in Linux

The following steps can be used to install the TensorFlow framework and other necessary software in Ubuntu Linux running native (installed directly on hardware) or running within Windows Subsystem for Linux 2.


#### Install TensorFlow

The Python development and virtual environment setup recommendation by TensorFlow is to isolate package installation from the system.

The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be >= 2.10.0.

* ##### Virtual environment install 

    You can follow the instructions in [stock tensorflow install](https://www.tensorflow.org/install/pip#step-by-step_instructions) to activate the virtual environment.

    On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
    ```bash
    (tf)$ pip install --upgrade pip
    ```

    To install in virtual environment, you can run 
    ```bash
    (tf)$ pip install 'tensorflow>=2.10.0'
    ```

* ##### System environment install 

    If want to system install in $HOME, please append `--user` to the commands.
    ```bash
    $ pip install --user 'tensorflow>=2.10.0'
    ```
    And the following system environment install for Intel® Extension for TensorFlow* will use the same practice. 

### 3. Install Intel® Extension for TensorFlow*

To install a GPU-only version in virtual environment, which depends on Intel GPU drivers and oneAPI BaseKit, you can run

```bash
(tf)$ pip install --upgrade intel-extension-for-tensorflow[gpu]
```

### 4. Verify the Installation 

```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```

You can also run a [quick_example](../../../examples/quick_example.md) to verify the installation.


### Optional: Install Full Intel® oneAPI

If you prefer to have access to full Intel® oneAPI you need to install at least the following:

- Intel® oneAPI DPC++ Compiler

- Intel® oneAPI Math Kernel Library (oneMKL)

Download and install the verified DPC++ compiler and oneMKL in Ubuntu 22.04.

```bash
$ wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767_offline.sh
# 3 components are necessary: DPC++/C++ Compiler, DPC++ Libiary and oneMKL
$ sudo sh ./l_BaseKit_p_2022.3.0.8767_offline.sh
```

For any more details, please follow the procedure in [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).

#### Setup environment variables

When using the full Intel® oneAPI Base Toolkit, you will need to set up necessary environment variables with:

```bash
source /opt/intel/oneapi/setvars.sh
```

A user may install more components than Intel® Extension for TensorFlow* needs, and if required, `setvars.sh` can be customized to point to a specific directory by using a [configuration file](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos/use-a-config-file-for-setvars-sh-on-linux-or-macos.html):

```bash
source /opt/intel/oneapi/setvars.sh --config="full/path/to/your/config.txt"
```
