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
from absl import flags

from utils.base_test_class import ASUnitTestBase

layers = tf.keras.layers
MIN_TOP_1_ACCURACY = 0.76
MAX_TOP_1_ACCURACY = 0.77
FLAGS = flags.FLAGS

class TestRN50Training(ASUnitTestBase):
  """Test RN50 Training model. As this model is defined out of this repo,
  and it is not return "loss" or "grads" as in AUUnitTestBase defined. Thus,
  in this test, we only test the direct run case.
  """

  def define_common_config(self):
    '''Define common configs for all testcases in this UT. This will set loss_fn,
    input_shape, etc.
    
    '''
    # Here is only the example of custom `define_common_config` function,
    # This function could be omitted if everything is kept default.
    config = cfg.ASConfig()

    config.input_shape = [64, 224, 224, 3]
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
    '''

    def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
      return tf.keras.regularizers.L2(
          l2_weight_decay) if use_l2_regularizer else None

    input_shape = self._common_config.input_shape[1:]

    return tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000
    )

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
    super(TestRN50Training, self).setUp()

  def test_resnet50_read_graph(self):

    # Copy common config and set test_specified config.
    local_config = self._common_config
    # Set specified one, which run the single device and run_imported graph.
    local_config.set_graph_run_properties(
        graph_run_properties_list=[
            cfg.GraphRunProperty(run_mode=cfg.RunMode.AUTO_SHARDING),
            cfg.GraphRunProperty(run_mode=cfg.RunMode.SINGLE_DEVICE),
        ],
        test_method_name=self.get_separate_test_name(),
        graph_dump_prefix=self.get_separate_test_name())

    self.run_and_compare(local_config)


if __name__ == '__main__':
  tf.test.main()