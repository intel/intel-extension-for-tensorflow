# Copyright (c) 2022 Intel Corporation
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


#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))

import tensorflow_hub as hub
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import urllib
import os
import sys

def download(url, filename):
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        data = response.read()
        out_file.write(data)

def download_img():
    ImageURL =  "https://github.com/intel/caffe/raw/master/examples/images/"
    image_names = ["cat.jpg"]

    for name in image_names:
        url = ImageURL + name
        if not os.path.exists(name):
            download(url, name)
            print("Downloaded {}".format(name))

def load_data(orig):
    img_width, img_height = 224, 224

    print("Load %s to inference\n" % orig)

    img = image.load_img(orig, target_size=(img_width, img_height))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    x= x/2+ 0.5
    return x

def main():
    module = hub.KerasLayer("https://tfhub.dev/google/supcon/resnet_v1_50/imagenet/classification/1")
    download_img()
    images = load_data("cat.jpg")
    logits = module(images)
    logits = tf.nn.softmax(logits)
    logits = logits.numpy()
    model_index = decode_predictions(logits, top=1)[0]
    print(model_index)


if __name__ == "__main__":
    main()
