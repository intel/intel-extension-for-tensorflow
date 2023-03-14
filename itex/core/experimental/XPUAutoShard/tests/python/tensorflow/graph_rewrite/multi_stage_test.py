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

import os
import tensorflow as tf
import run_single_conv_full

class TestMultiStage(tf.test.TestCase):

  def test_single_conv_multi_stage( self ): 
    pwd = os.getcwd()
    # run original conv 
    relative_path_ori = "pbtxt/fixed_dim/nhwc/conv/conv_training_ori.pbtxt"
    pbtxt_path_ori = os.path.join( pwd, relative_path_ori)
    hs_roundtrip_flag = "false"
    status = run_single_conv_full.run_single_conv( pbtxt_path_ori, hs_roundtrip_flag  )
    loss_ori = status["loss_val"]
    gradient_ori = status["grads"]
    # run 2 gpu auto_sharding_test conv
    relative_path_test = "../../../cpp/tensorflow/pbtxt/test_conv_2_gpu.pbtxt"
    pbtxt_path_test = os.path.join( pwd, relative_path_test)
    hs_roundtrip_flag = "true"
    status = run_single_conv_full.run_single_conv( pbtxt_path_test, hs_roundtrip_flag )
    loss_test = status["loss_val"]
    gradient_test = status["grads"]
    
    self.assertAllClose( loss_ori, loss_test )
    self.assertAllClose( gradient_ori, gradient_test )

if __name__ == '__main__':
  tf.test.main()

