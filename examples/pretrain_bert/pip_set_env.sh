#!/bin/bash

#
# Copyright (c) 2021-2022 Intel Corporation
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
#

ENV_NAME=env_itex
deactivate
rm -rf $ENV_NAME
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip install --upgrade pip
pip install --upgrade intel-extension-for-tensorflow[xpu]
pip install git+https://github.com/NVIDIA/dllogger
pip install requests tqdm horovod sentencepiece tensorflow_hub wget progressbar nltk pubmed_parser tf-keras
pip install tensorflow-addons  # Version details in https://github.com/tensorflow/addons
