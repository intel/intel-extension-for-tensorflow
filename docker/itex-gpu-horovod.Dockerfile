# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ENV LANG=C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

HEALTHCHECK NONE
RUN useradd -d /home/itex -m -s /bin/bash itex

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    apt-utils \
    build-essential \
    ca-certificates \
    clinfo \
    curl \
    git \
    gnupg2 \
    gpg-agent \
    rsync \
    sudo \
    unzip \
    wget && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN no_proxy=$no_proxy wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy max' | \
    tee  /etc/apt/sources.list.d/intel.gpu.jammy.list

ARG ICD_VER
ARG LEVEL_ZERO_GPU_VER
ARG LEVEL_ZERO_VER
ARG LEVEL_ZERO_DEV_VER

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-opencl-icd=${ICD_VER} \
    intel-level-zero-gpu=${LEVEL_ZERO_GPU_VER} \
    level-zero=${LEVEL_ZERO_VER} \
    level-zero-dev=${LEVEL_ZERO_DEV_VER} && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN no_proxy=$no_proxy wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
   | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
   | tee /etc/apt/sources.list.d/oneAPI.list

ARG DPCPP_VER
ARG MKL_VER
ARG CCL_VER

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    intel-oneapi-runtime-dpcpp-cpp=${DPCPP_VER} \
    intel-oneapi-runtime-mkl=${MKL_VER} \
    intel-oneapi-runtime-ccl=${CCL_VER} && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN echo "intelpython=exclude" > $HOME/cfg.txt

ENV LANG=C.UTF-8
ARG PYTHON=python3.10

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON} lib${PYTHON} python3-pip && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

RUN pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/bin/python3

ARG TF_VER="2.12"

RUN pip --no-cache-dir install tensorflow==${TF_VER}

ARG HOROVOD_WHL

COPY models/horovod/${HOROVOD_WHL} /tmp/horovod_whls/

RUN pip install /tmp/horovod_whls/* && \
    rm -rf /tmp/horovod_whls

ARG TF_PLUGIN_WHEEL

COPY models/binaries/$TF_PLUGIN_WHEEL /tmp/tf_whls/

RUN pip install /tmp/tf_whls/* && \
    rm -rf /tmp/tf_whls

ENV LD_LIBRARY_PATH=/opt/intel/oneapi/lib:/opt/intel/oneapi/lib/intel64:/opt/intel/oneapi/lib/intel64/libfabric:$LD_LIBRARY_PATH
ENV PATH=/opt/intel/oneapi/lib/intel64/bin:$PATH
ENV I_MPI_ROOT=/opt/intel/oneapi/lib/intel64/
ENV CCL_ROOT=/opt/intel/oneapi/lib/intel64/

ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/THIRD-PARTY-PROGRAMS.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/dpcpp-third-party-programs.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/oneccl-third-party-programs.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/onemkl-third-party-program-sub-tpp.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/onemkl-third-party-program.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/third-party-program-of-intel-extension-for-tensorflow.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/third-party-programs-of-intel-tensorflow.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/third-party-programs-of-intel-optimization-for-horovod.txt /licenses/
ADD https://raw.githubusercontent.com/oneapi-src/oneCCL/master/third-party-programs.txt /licenses/