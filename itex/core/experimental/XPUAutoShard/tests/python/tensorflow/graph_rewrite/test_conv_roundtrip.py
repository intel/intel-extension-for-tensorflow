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


class TestSingleConv(ASUnitTestBase):
  """Test Single Conv Training/Inference model.
  """

  def define_common_config(self):
    '''Define common configs for all testcases in this UT. This will set loss_fn,
    input_shape, etc.
    
    '''
    # Here is only the example of custom `define_common_config` function,
    # This function could be omitted if everything is kept default.
    config = cfg.ASConfig()

    config.input_shape = [64, 28, 28, 3]
    config.label_shape = [64, 26, 26, 2]

    loss_fn = tf.keras.losses.MeanSquaredError(reduction=config.reduction)
    config.loss_fn = loss_fn
    return config

  def define_model(self):
    '''Define Models used by all testcases in this UT.
    
    Return a callable keras model object. Note that this function does not 
    defines training/inference function. e.g., the loss_fn/optimizers are not
    defined in this function. They should be construct in each test run.
    '''

    input_shape = self._common_config.input_shape[1:]

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

    return ConvModel()

  def setUp(self):
    super(TestSingleConv, self).setUp()

  def test_single_device(self):
    # Copy common config and set test_specified config.
    local_config = self.new_local_config()
    # Set specified one, which run the single device.
    local_config.set_graph_run_properties(graph_run_properties_list=[
        cfg.GraphRunProperty(run_mode=cfg.RunMode.SINGLE_DEVICE,
                             generate_original_graph=True),
        cfg.GraphRunProperty(
            cfg.RunMode.RUN_IMPORTED_GRAPH,
            pbtxt_path="pbtxt/dynamic_dim/nhwc/conv/conv_training_ori.pbtxt",
            run_dynamic_shape_flag=True)
    ],
                                          test_method_name=self.get_separate_test_name(),
                                          graph_dump_prefix=self.get_separate_test_name()
                                         )

    self.run_and_compare(local_config)



  def test_mirrored_strategy(self):
    # Copy common config and set test_specified config.
    local_config = self.new_local_config()
    # Set specified one, which run the single device.
    local_config.set_graph_run_properties(graph_run_properties_list=[
        cfg.GraphRunProperty(
            cfg.RunMode.AUTO_SHARDING,
            pbtxt_path="pbtxt/dynamic_dim/nhwc/conv/conv_training_1_1.pbtxt"),
        cfg.GraphRunProperty(cfg.RunMode.DISTRIBUTE_STRATEGY,
                             dist_strategy=tf.distribute.MirroredStrategy(
                                 ["GPU:0", "GPU:1"]))
    ],
                                          test_method_name=self.get_separate_test_name(),
                                          graph_dump_prefix=self.get_separate_test_name()
                                         )

  def test_inference(self):
    # Copy common config and set test_specified config.
    local_config = self.new_local_config()
    # Set specified one, which run the single device.

    local_config.set_graph_run_properties(
        graph_run_properties_list=[
            cfg.GraphRunProperty(run_mode=cfg.RunMode.SINGLE_DEVICE,
                                 generate_original_graph=True),
            cfg.GraphRunProperty(
                cfg.RunMode.RUN_IMPORTED_GRAPH,
                pbtxt_path="pbtxt/fixed_dim/nhwc/conv/conv_fwd_ori.pbtxt",
                run_dynamic_shape_flag=True)
        ],
        test_method_name=self.get_separate_test_name(),
        graph_dump_prefix=self.get_separate_test_name())
    local_config.training = False

    self.run_and_compare(local_config, compare_grads=False)


if __name__ == '__main__':
  tf.test.main()