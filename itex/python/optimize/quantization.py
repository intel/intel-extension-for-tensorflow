# Copyright (c) 2023 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

try:
  from neural_compressor.config import PostTrainingQuantConfig
  from neural_compressor.data import DataLoader, Datasets
  from neural_compressor.quantization import fit
except ImportError:
  raise ImportError("Could not import neural_compressor. Please install with "
                    "pip install neural_compressor")

class default_static_qconfig(object):
  def __init__(self, device='cpu', **kwargs):
    self.inc_config = PostTrainingQuantConfig(
      # TODO: Input and Output parameters.
      # If tensorflow model is used, model's inputs/outputs will be auto inferenced,
      # but sometimes auto inferenced inputs/outputs will not meet your requests, set them manually.

      # TODO: Find other useful parameters and add here.

      # inputs=['input'],
      # outputs=['resnet_v2_50/SpatialSqueeze'],
      device= device,
      quant_format='default',
      approach='static',
      backend='itex',
      calibration_sampling_size=[20],
    )


class dataset(object):
  def __init__(self, dataset, **kwargs):
    self.batch_size = None
    if hasattr(dataset, '_batch_size'):
      self.batch_size = dataset._batch_size
      unbatched_dataset = dataset.unbatch()
    self.dataset = unbatched_dataset


class Converter(object):
  def __init__(self, model, **kwargs):
    self.model=model
    self.optimizations=None
    self.representative_dataset=None

  def convert(self):
    return fit(
        model=self.model,
        conf=self.optimizations.inc_config,
        calib_dataloader=DataLoader(framework='tensorflow', dataset=self.representative_dataset, batch_size=1))


def default_dataset(dataset_type):
    return Datasets('tensorflow')[dataset_type]

def from_model(model):
  return Converter(model)
