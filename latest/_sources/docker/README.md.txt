Intel® Extension for TensorFlow* Docker Container Guide
=======================================================

## Description

This document has instructions for running TensorFlow using Intel® Extension for TensorFlow* in a Docker container.

Assumptions:
* Host machine contains an Intel GPU.
* Host machine uses a Linux kernel that is compatible with GPU drivers.
* Host machine has a compatible Intel GPU driver installed
* Host machine has Docker software installed.

Refer to [Install for XPU](../docs/install/install_for_xpu.html) and [Install for CPU](../docs/install/install_for_cpu.html) for detail.

## Binaries Preparation

Download and copy Intel® Extension for TensorFlow* wheel into ./models/binaries directory. You can get the intel-extension-for-tensorflow wheel link from https://pypi.org/project/intel-extension-for-tensorflow/#files, and intel-extension-for-tensorflow-lib wheel link from https://pypi.org/project/intel-extension-for-tensorflow-lib/#files.

To use Intel® Optimization for Horovod* with the Intel® oneAPI Collective Communications Library (oneCCL), copy Horovod wheel into ./models/binaries as well. You can get the intel-optimization-for-horovod wheel link from https://pypi.org/project/intel-optimization-for-horovod/#files.

```
mkdir -p ./models/binaries
cd ./models/binaries
wget <download link from https://pypi.org/project/intel-extension-for-tensorflow/#files>
wget <download link from https://pypi.org/project/intel-extension-for-tensorflow-lib/#files>
wget <download link from https://pypi.org/project/intel-optimization-for-horovod/#files>
```

## Usage of Docker Container
### I. Customize Build Script
We provide [build.sh](./build.sh) as the Docker container build script. The OS version and some software versions (such as Python and TensorFlow) are hard coded inside the script. If you're using a different version, you can edit this script.

For example, to build a Docker container with Python 3.10 and TensorFlow 2.15.1 on an Ubuntu 22.04 layer, update [build.sh](./build.sh) as shown below.

```bash
IMAGE_NAME=intel-extension-for-tensorflow:cpu-ubuntu
        docker build --build-arg UBUNTU_VERSION=22.04 \
                                --build-arg PYTHON=python3.10 \
                                --build-arg TF_VER=2.15.1 \
                                --build-arg WHEELS=*.whl \
                                -t $IMAGE_NAME \
                                -f itex-cpu.Dockerfile .
```

### II. Build the Container

To build the Docker container, enter into [docker](./) folder and run below commands:

```bash
./build.sh [xpu/cpu]
```
### III. Running the Container

Run the following commands to start the Docker container. You can use the `-v` option to mount your local directory into the container. To make the GPU available in the container, attach the GPU to the container using `--device /dev/dri` option and run the container:

```bash
IMAGE_NAME=intel-extension-for-tensorflow:xpu
docker run -v <your-local-dir>:/workspace \
           -v /dev/dri/by-path:/dev/dri/by-path \
           --device /dev/dri \
           --privileged \
           --ipc=host \
           -e http_proxy=$http_proxy \
           -e https_proxy=$https_proxy \
           -e no_proxy=$no_proxy \
           -it \
           $IMAGE_NAME bash
```

>**Note**: Only for distributed training workloads with Intel® Optimization for Horovod*, the following script should be executed to set the required environment variables after entering the container:
```
source /opt/intel/horovod-vars.sh
```


## Verify That Intel GPU is Accessible From TensorFlow
You are inside the container now. Run this command to verify the Intel GPU is visible to TensorFlow:

```
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```
You should see your GPU device in the list of devices. Sample output looks like this:

```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 9266936945121049176
xla_global_id: -1
, name: "/device:XPU:0"
device_type: "XPU"
locality {
bus_id: 1
}
incarnation: 15031084974591766410
physical_device_desc: "device: 0, name: INTEL_XPU, pci bus id: <undefined>"
xla_global_id: -1
, name: "/device:XPU:1"
device_type: "XPU"
locality {
bus_id: 1
}
incarnation: 17448926295332318308
physical_device_desc: "device: 1, name: INTEL_XPU, pci bus id: <undefined>"
xla_global_id: -1
]
```
