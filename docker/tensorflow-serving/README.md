# Intel® Extension for TensorFlow* Serving - Docker Container Guide

## Description

This document has instruction for running TensorFlow Serving using Intel® Extension for TensorFlow* in a Docker container.

## Build the Docker Image

To build the docker container, enter into [docker/tensorflow-serving](./) folder and follow these steps.

### I. Binaries Preparation

Refer to [Install for Tensorflow Serving](../../docs/guide/tf_serving_install.md) to build the TensorFlow Serving binary, and refer to [Install for CPP](../../docs/install/install_for_cpp.md) to build the Intel® Extension for TensorFlow* CC library from source. Then package and copy these binaries into the `./models/binaries` directory, as shown below.

```bash
mkdir -p ./models/binaries

# Package and copy Intel® Extension for TensorFlow* CC library
mkdir -p itex-bazel-bin/
cp -r <path_to_itex>/bazel-out/k8-opt-ST-*/bin/ itex-bazel-bin/
# if you build with threadpool
cp -r <path_to_itex>/bazel-out/k8-opt-ST-*/bin/ itex-bazel-bin/bin_threadpool/
tar cvfh itex-bazel-bin.tar itex-bazel-bin/
cp itex-bazel-bin.tar  ./models/binaries/

# Copy TensorFlow Serving binary
cp <path_to_tensorflow_serving>/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server ./models/binaries/

```

### II. Build the Container

If you build the container using an Intel GPU, make sure you meet these assumptions:

* Host machine has an Intel GPU.
* Host machine uses a Linux kernel that is compatible with GPU drivers.
* Host machine has a compatible Intel GPU driver installed.

Refer to [Install for GPU](../docs/install/install_for_xpu.md) for detail.

Run the [build.sh](./build.sh), specifying either `gpu` or `cpu` as appropriate, to build the target Docker image.
```bash
./build.sh [gpu/cpu]
```

## Running the Container

Run these commands to start the Docker container. You can use the `-v` option to mount your local directory into container. To make a GPU available in the container, attach the GPU to the container using the `--device /dev/dri` option and run the container:

```
IMAGE_NAME=intel-extension-for-tensorflow:serving-gpu
docker run -v <your-local-dir>:/workspace \
           -v /dev/dri/by-path:/dev/dri/by-path \
           --device /dev/dri \
           --privileged \
           --ipc=host \
           -p 8500:8500 \
           -e MODEL_NAME=<your-model-name> \
           -e MODEL_DIR=<your-model-dir> \
           -e http_proxy=$http_proxy \
           -e https_proxy=$https_proxy \
           -e no_proxy=$no_proxy \
           -it \
           $IMAGE_NAME
```
NOTE: If you want to run docker with threadpool, you should add `-e ITEX_OMP_THREADPOOL=0`