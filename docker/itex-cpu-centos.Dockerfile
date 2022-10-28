#copyright (c) 2022 Intel Corporation
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

ARG CENTOS_VER=8 

FROM centos:${CENTOS_VER}

SHELL ["/bin/bash", "-c"]
ENV LANG=C.UTF-8

RUN sed -i.bak '/^mirrorlist=/s/mirrorlist=/#mirrorlist=/g' /etc/yum.repos.d/CentOS-Linux-* && \
    sed -i.bak 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-* && \
    yum distro-sync -y && \
    yum --disablerepo '*' --enablerepo=extras swap centos-linux-repos centos-stream-repos -y && \
    yum distro-sync -y && \
    yum clean all

RUN yum update -y && yum install -y \
    ca-certificates \
    curl \
    git \
    gnupg2 \
    rsync \
    sudo \
    unzip \
    which \
    wget \
    && \
    yum distro-sync -y && \
    yum clean all

ARG PY_VER="39"
ARG PYTHON=python3
RUN yum update -y && yum install -y \
    python${PY_VER} \
    python${PY_VER}-pip && \
    yum clean all

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python

ARG TF_VER="2.10"

RUN pip --no-cache-dir install tensorflow==${TF_VER}

ARG TF_PLUGIN_WHEEL

COPY models/binaries/$TF_PLUGIN_WHEEL /tmp/itex_cpu_whls/

RUN pip install /tmp/itex_cpu_whls/* && \
    rm -rf /tmp/itex_cpu_whls

ENV DNNL_MAX_CPU_ISA="AVX512_CORE_AMX"
ENV TF_ENABLE_MKL_NATIVE_FORMAT=1

ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/THIRD-PARTY-PROGRAMS.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/third-party-program-of-intel-extension-for-tensorflow.txt /licenses/
ADD https://raw.githubusercontent.com/intel/intel-extension-for-tensorflow/master/third-party-programs/dockerlayer/third-party-programs-of-intel-tensorflow.txt /licenses/