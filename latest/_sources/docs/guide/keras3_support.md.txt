# Keras 3 Overview

[Keras](https://keras.io/about/) is a deep learning API written in Python and capable of running on top of either JAX, TensorFlow, or PyTorch. Both JAX and TensorFlow backend compiles the model by XLA and delivers the best training and prediction performance on GPU. But results vary from model to model, as non XLA TensorFlow is occasionaly faster on GPU. The following image show how ITEX works with XLA, Keras 3 TensorFlow backend and legacy Keras.

<p align="center">
  <img src="images/keras3.png" alt="keras3" />
</p>


## Use Case with different performance
There are serval use cases that can lead to diffent performance.

* Default
Users use Keras 3 and the model supports jit, the model will runs into XLA.
If user script does not contains keras related code and does not enables XLA in tensorflow. There will be performance regression. Set environment variable `ITEX_DISABLE_XLA=1` to avoid regression. After ITEX XLA disabled, users can choose wether to use NPD (default) or stream excutor for better performance by environment variable `ITEX_ENABLE_NEXTPLUGGABLE_DEVICE`. 

* Legacy Keras
To continue using Keras 2.0, do the following.
1. Install `tf-keras` via `pip install tf-keras`
2. To switch `tf.keras` to use Keras 2 (`tf-keras`), set the environment variable `TF_USE_LEGACY_KERAS=1` directly or in your python program with `import os;os.environ["TF_USE_LEGACY_KERAS"]="1"`. Please note that this will set it for all packages in your Python runtime program
3. Change the keras import: replace `import keras` with `import tf_keras as keras`. Update any `from keras import ` to `from tf_keras`.

Users can choose wether to use NPD (default) or stream excutor for better performance by environment variable `ITEX_ENABLE_NEXTPLUGGABLE_DEVICE`.

* Keras 3 with jit_compile disabled
Users can disable jit_compile by `model.jit_compile=False` or `model.compile(..., jit_compile=False)`. The use of itex ops override can also lead to disabling jit_compile. In this case, `ITEX_DISABLE_XLA=1` must be set.

* Enable XLA through TensorFlow.
Users can enable XLA through TensorFlow by add environment variable `TF_XLA_FLAGS="--tf_xla_auto_jit=1"`. Use `tf_xla_auto_jit=1` for auto clustering TF ops into XLA, `tf_xla_auto_jit=2` for compiling all into XLA. Users should set `model.jit_compile=False` if keras model is used. If ITEX custom ops is used or `ITEX_OPS_OVERRIDE` is set, users should use `tf_xla_auto_jit=1` to avoid error.





## Situations leads to warning or Error
We list all invalid cases here. Keras version equals to 0 means model script does not use Keras.

Note that in any cases, `import keras` first before `import tensorflow` will cause an error due to circular import in ITEX.

| OPS_OVERRIDE | TF_AUTO_JIT_FLAG | Keras version | NPD | Jit Compile | Warning | Error | Solution |
|--------------|------------------|---------------|-----|-------------|---------|-------|----------|
| Any          | 0                | 0             | 0   | NA          |         | PluggableDevice cannot work with latest Keras. | `ITEX_DISABLE_XLA=1` |
| Any          | 0                | 0             | 1   | NA          | Perf Regression | | `ITEX_DISABLE_XLA=1` |
| Any          | Any              | 2             | Any | 1           |         | | Unkown behavior, not supported. Use `TF_AUTO_JIT_FLAG="--tf_xla_auto_jit=1"` or `2` to enable XLA |
| Any          | 0                | 3             | 0   | Any         |         | Cannot close NPD when keras 3 | `ITEX_DISABLE_XLA=1` |
| Any          | 0                | 3             | 1   | 0           |         | perf regression | `ITEX_DISABLE_XLA=1` |
| Any          | 1                | Any           | 0   | Any         |         | Cannot close NPD | `ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=1` |
| Any          | 2                | Any           | 0   | Any         |         | Cannot close NPD | `ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=1` |
| 1            | 2                | Any           | 1   | Any         | custom op not supported by XLA | | `ITEX_OPS_OVERRIDE=0` |
