Intel® Extension for TensorFlow* Docker Container Guide
=======================================================

## Description

This document has instruction for running TensorFlow using Intel® Extension for TensorFlow* in docker container.

Assumptions:
* Host machine installs Intel GPU.
* Host machine installs Linux kernel that is compatible with GPU drivers.
* Host machine has Intel GPU driver.
* Host machine installs Docker software.

Refer to [Install for GPU](../docs/install/install_for_gpu.md) and [Install for CPU](../docs/install/experimental/install_for_cpu.md) for detail.

## Binaries Preparation

Download and copy Intel® Extension for TensorFlow* wheel into ./models/binaries directory.

```
mkdir ./models/binaries
```

To use Intel® Optimization for Horovod* with the Intel® oneAPI Collective Communications Library (oneCCL), copy Horovod wheel into ./models/binaries as well.

## Usage of Docker Container
### I. Customize build script
[build.sh](./build.sh) is provided as docker container build script. While OS version and some software version (such as Python and TensorFlow) is hard coded inside the script. If you prefer to use newer or later version, you can edit this script.

For example, to build docker container with Python 3.10 and TensorFlow 2.13 on Ubuntu 22.04 layer, update [build.sh](./build.sh) as below.

```bash
IMAGE_NAME=intel-extension-for-tensorflow:cpu-ubuntu
        docker build --build-arg UBUNTU_VERSION=22.04 \
                                --build-arg PYTHON=python3.10 \
                                --build-arg TF_VER=2.13 \
                                --build-arg WHEELS=*.whl \
                                -t $IMAGE_NAME \
                                -f itex-cpu.Dockerfile .
```

### II. Build the container

To build the docker container, enter into [docker](./) folder and run below commands:

```bash
./build.sh [xpu/cpu]
```
### III. Running container

Run following commands to start docker container. You can use -v option to mount your local directory into container. To make GPU available in the container, attach the GPU to the container using --device /dev/dri option and run the container:

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

## Verify if Intel GPU is accessible from TensorFlow
You are inside container now. Run following command to verify Intel GPU is visible to TensorFlow:

```
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```
You should be able to see GPU device in list of devices. Sample output looks like below:

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