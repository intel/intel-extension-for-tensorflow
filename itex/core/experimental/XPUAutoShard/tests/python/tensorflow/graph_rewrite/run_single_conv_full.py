#  Copyright (c) 2023 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

import contextlib
import os

@contextlib.contextmanager
def options(options):
  old_opts = tf.config.optimizer.get_experimental_options()
  tf.config.optimizer.set_experimental_options(options)
  try:
    yield
  finally:
    tf.config.optimizer.set_experimental_options(old_opts)


layers = tf.keras.layers


def run_single_conv( pbtxt_path , hs_round_flag ):


  # Set the generated HS_FILE_NAME
  cwd = os.getcwd()
  path = os.path.join(cwd, pbtxt_path)
  os.environ["HS_FILE_NAME"] = path

  tf.random.set_seed(42)
  input_shape = (64, 28, 28, 3)

  x = tf.random.normal(input_shape)
  y = tf.random.normal([64, 26, 26, 2])

  class ConvModel(tf.keras.Model):
    def __init__(self, activation=None):
      super(ConvModel, self).__init__()
      self.conv_relu = layers.Conv2D(2,
                                      3,
                                      data_format="channels_last",
                                      activation=activation,
                                      use_bias=False,
                                      input_shape=input_shape[1:])

    def call(self, inputs):
      x = self.conv_relu(inputs)
      return x

  model = ConvModel()
  # Explicitly call once for model to initialize, this is bad coding
  model(x, training=False)
  w = model.get_weights()

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, 28, 28, 3], dtype=tf.float32),
      tf.TensorSpec(shape=[None, 26, 26, 2], dtype=tf.float32)
  ])
  def custom_train_single_step(x, y):

    with tf.GradientTape() as tape:
      logits = model(x, training=True)
      loss_val = loss_fn(y, logits)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_val, grads

  optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
  loss_fn = tf.keras.losses.MeanSquaredError(
      reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

  os.environ["HS_ROUND_TRIP"] = hs_round_flag

  print("==========================run_single_conv===================================")
  print("pbtxt_path = ", pbtxt_path)
  with options({'layout_optimizer':False}):
    loss_val, grads = custom_train_single_step(x, y)

  status = { "loss_val" : loss_val,
              "grads" : grads,
            }
  print( "loss: ",loss_val )
  return status

if __name__ == '__main__':
  cwd = os.getcwd()
  pbtxt_path = "pbtxt/fixed_dim/nhwc/conv/conv_training_ori.pbtxt"
  path = os.path.join(cwd, pbtxt_path)
  run_single_conv( path , "false" )