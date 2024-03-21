#copyright (c) 2023-2024 Intel Corporation
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

ARG DEBIAN_FRONTEND=noninteractive

HEALTHCHECK NONE
RUN useradd -d /home/itex -m -s /bin/bash itex

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    apt-utils \
    ca-certificates \
    clinfo \
    git \
    gnupg2 \
    gpg-agent \
    rsync \
    sudo \
    unzip \
    wget && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

EXPOSE 8500
EXPOSE 8501

ARG TF_SERVING_BINARY=tensorflow_model_server
ARG TF_PLUGIN_TAR=itex-bazel-bin.tar

COPY models/binaries/${TF_SERVING_BINARY}  /usr/local/bin/tensorflow_model_server
COPY models/binaries/${TF_PLUGIN_TAR} /tmp/itex-bazel-bin.tar

RUN mkdir -p /itex && tar -xvf /tmp/itex-bazel-bin.tar -C /itex && \
    rm /tmp/itex-bazel-bin.tar

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=my_model
RUN mkdir -p ${MODEL_BASE_PATH}/${MODEL_NAME}

ENV ITEX_OMP_THREADPOOL=1
RUN echo '#!/bin/bash \n\n\
if [ ${ITEX_OMP_THREADPOOL} == 1 ]; then \n\
    DIR=/itex/itex-bazel-bin/bin/itex \n\
else \n\
    DIR=/itex/itex-bazel-bin/bin_threadpool/itex \n\
fi \n\
/usr/local/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--tensorflow_plugins=${DIR} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
