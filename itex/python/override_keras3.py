# Copyright (c) 2023 Intel Corporation
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
"""check keras model if it supports jit compile."""
import keras
import os
import types

XLA_AUTO_CLUSTER = False
if "--tf_xla_auto_jit=1" in os.environ.get("TF_XLA_FLAGS", "").replace(" ", ""):
    XLA_AUTO_CLUSTER = True


def copy_func(f, name=None):
    '''
    return a function with same code, globals, defaults, closure, and
    name (or provide a new name)
    '''
    fn = types.FunctionType(f.__code__, f.__globals__, name or f.__name__,
                            f.__defaults__, f.__closure__)
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    return fn


keras_model_compile = copy_func(keras.src.trainers.trainer.Trainer.compile)
keras_model_predict = copy_func(
    keras.src.backend.tensorflow.trainer.TensorFlowTrainer.predict)


def itex_model_compile(self,
                       optimizer="rmsprop",
                       loss=None,
                       loss_weights=None,
                       metrics=None,
                       weighted_metrics=None,
                       run_eagerly=False,
                       steps_per_execution=1,
                       jit_compile="auto",
                       auto_scale_loss=True,
                       ):
    keras_model_compile(self, # pylint: disable=not-callable
                        optimizer=optimizer,
                        loss=loss,
                        loss_weights=loss_weights,
                        metrics=metrics,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly,
                        steps_per_execution=steps_per_execution,
                        jit_compile=jit_compile,
                        auto_scale_loss=auto_scale_loss,)
    if ((not self.jit_compile) and os.environ.get("ITEX_DISABLE_XLA", "0") in ("false", "0") and (not XLA_AUTO_CLUSTER)):
        print("This keras model does not support jit compile, please use legacy keras or set ITEX_DISABLE_XLA=1")
        quit()


def itex_predict(
    self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
):
    if ((not self.jit_compile) and os.environ.get("ITEX_DISABLE_XLA", "0") in ("false", "0") and (not XLA_AUTO_CLUSTER)):
        print("This keras model is not jit compiled, please compile it or use legacy keras or set ITEX_DISABLE_XLA=1")
        quit()
    return keras_model_predict(self, x, batch_size, verbose, steps, callbacks) # pylint: disable=not-callable


def override_keras3():
    '''
    override model_supports_jit
    '''
    if os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"):
        return
    try:
        from pkg_resources import packaging  # pylint: disable=import-outside-toplevel
        version = packaging.version.parse
        if version(keras.__version__) >= version("3.0.0"):
            keras.src.trainers.trainer.Trainer.compile = itex_model_compile
            keras.src.backend.tensorflow.trainer.TensorFlowTrainer.predict = itex_predict

    except BaseException:  # pylint: disable=broad-except
        import logging
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=format_str)
        logger = logging.getLogger(__name__)
        logger.warning(
            "itex.override_keras3 failed")  # pylint: disable=line-too-long
