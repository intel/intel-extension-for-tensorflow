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

from copy import deepcopy
import tensorflow as tf

import utils.config_definitions as cfg
from utils.base_test_class import ASUnitTestBase

layers = tf.keras.layers


class TestResNetBlock(ASUnitTestBase):
  """Test ResNet Block model.
  """

  def define_common_config(self):
    '''Define common configs for all testcases in this UT. This will set loss_fn,
    input_shape, etc.
    
    '''
    # Here is only the example of custom `define_common_config` function,
    # This function could be omitted if everything is kept default.
    config = cfg.ASConfig()

    config.input_shape = [64, 28, 28, 3]
    config.label_shape = [64]
    config.input_dtype = tf.float32
    config.label_dtype = tf.float32

    config.reduction = tf.keras.losses.Reduction.NONE

    def loss_fn(logits, y):
      loss = tf.keras.losses.SparseCategoricalCrossentropy(
          reduction=config.reduction)(logits, y)
      avg_loss = tf.nn.compute_average_loss(loss)
      return avg_loss

    config.loss_fn = loss_fn

    return config

  def define_model(self):
    '''Define Models used by all testcases in this UT.
    
    Return a callable keras model object. Note that this function does not 
    defines training/inference function. e.g., the loss_fn/optimizers are not
    defined in this function. They should be construct in each test run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       separate testcase.
    '''

    def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
      return tf.keras.regularizers.L2(
          l2_weight_decay) if use_l2_regularizer else None

    input_shape = self._common_config.input_shape[1:]

    class ResNetBlockModel(tf.keras.Model):

      def __init__(self):
        super(ResNetBlockModel, self).__init__()
        self.conv = layers.Conv2D(2,
                                  3,
                                  kernel_initializer='he_normal',
                                  data_format="channels_last",
                                  use_bias=False,
                                  input_shape=input_shape)
        self.bn = layers.BatchNormalization(axis=3,
                                            momentum=0.9,
                                            epsilon=1e-5,
                                            name='bn_conv1')
        self.dense = layers.Dense(
            10,
            kernel_initializer=tf.initializers.random_normal(stddev=0.01),
            kernel_regularizer=_gen_l2_regularizer(True),
            bias_regularizer=_gen_l2_regularizer(True),
            name='class10')
        self.flatten = tf.keras.layers.Flatten()
        self.activation_relu = layers.Activation('relu', dtype='float32')
        self.activation_softmax = layers.Activation('softmax', dtype='float32')
        self.shortcut = layers.Conv2D(2,
                                      3,
                                      kernel_initializer='he_normal',
                                      data_format="channels_last",
                                      use_bias=False,
                                      input_shape=input_shape)

      def call(self, inputs):
        shortcut = self.shortcut(inputs)
        shortcut = self.bn(shortcut)
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation_relu(x)
        x = layers.add([x, shortcut])
        x = self.flatten(x)
        x = self.dense(x)
        x = self.activation_softmax(x)
        return x

    return ResNetBlockModel()

  def generate_data(self, local_config):
    '''Custom generate data function, which overrides default data genera when user wants 
    to generate one's own data.
    
    '''
    tf.random.set_seed(42)
    input_shape = local_config.input_shape
    label_shape = local_config.label_shape

    inputs = tf.random.normal(input_shape, name='inputs')
    labels = tf.random.uniform(label_shape,
                               minval=0,
                               maxval=10 - 1,
                               dtype=tf.int32,
                               name='labels')
    labels = tf.cast(labels, dtype=local_config.label_dtype)

    return inputs, labels

  def setUp(self):
    super(TestResNetBlock, self).setUp()

  def test_single_device(self):
    # Copy common config and set test_specified config.
    # clone _common_config would trigger error due to uncloneable distributed strategy
    #local_config = self.new_local_config()
    local_config = self._common_config
    # Set specified one, which run the single device.
    local_config.set_graph_run_properties(graph_run_properties_list=[
        cfg.GraphRunProperty(run_mode=cfg.RunMode.AUTO_SHARDING),
        cfg.GraphRunProperty(run_mode=cfg.RunMode.SINGLE_DEVICE),
    ],
                                          test_method_name=self.get_separate_test_name(),
                                          graph_dump_prefix=self.get_separate_test_name()
                                         )

    self.run_and_compare(local_config)
'''
  def test_mirrored_strategy(self):
    # Copy common config and set test_specified config.
    local_config = self.new_local_config()
    # Set specified one, which run the single device.
    local_config.set_graph_run_properties(graph_run_properties_list=[
        cfg.GraphRunProperty(
            cfg.RunMode.AUTO_SHARDING,
            pbtxt_path="pbtxt/fixed_dim/nhwc/resnet/resnet_block_1_1.pbtxt"),
        cfg.GraphRunProperty(cfg.RunMode.DISTRIBUTE_STRATEGY,
                             dist_strategy=tf.distribute.MirroredStrategy(
                                 ["GPU:0", "GPU:1"]))
    ],
                                          test_method_name=self.get_separate_test_name(),
                                          graph_dump_prefix=self.get_separate_test_name()
                                         )
'''

if __name__ == '__main__':
  tf.test.main()
