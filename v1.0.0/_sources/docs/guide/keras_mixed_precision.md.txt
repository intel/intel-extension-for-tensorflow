# Keras Mixed Precision

## Overview

  Intel® Extension for TensorFlow* supports **[keras mixed precision](https://www.tensorflow.org/guide/mixed_precision)**,  which can run with 16-bit and 32-bit mixed floating-point types during training and inference to make it run faster with less memory consumption. 

  After installing Intel® Extension for TensorFlow*, you need to configure mixed_precision.Policy and identify hardware devices, the models will run with 16-bit and 32-bit mixed floating-point types. The following table shows the types of mixed_precision.Policy supported for Intel hardwares.

| mixed_precision.Policy | Hardware device |
| ---------------------- | --------------- |
| mixed_float16          | GPU             |
| mixed_bfloat16         | CPU, GPU        |



## How to identify different hardware types?

​Enable keras mixed-precision with Intel® Extension for TensorFlow* backend. We provide two ways to distinguish it.  

- Through **tf.config.list_physical_devices*

  For stock TensorFlow, if run with Nvidia GPU,  **tf.config.list_physical_devices('GPU')**  will return true. 

  For Intel® Extension for TensorFlow*, if run with Intel GPU, **tf.config.list_physical_devices('XPU')**  will return true. If run with Intel CPU, it will return false. 

- Through Intel® Extension for TensorFlow* Python API

  If run with GPU, is_gpu_available()  will return true,  if not, it will return false.

  ```python
  from intel_extension_for_tensorflow.python.test_func import test
  
  if test.is_gpu_available():
      ...
  ```

> **Note: Other than the difference in the identification of the hardware device, other behavior is the same, refer to the [keras mixed precision](https://www.tensorflow.org/guide/mixed_precision)** for details.

## Setup

```python
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
```

```python

2022-06-14 02:52:41.061277: W itex/core/profiler/gpu_profiler.cc:111] ******************************Intel TensorFlow Extension profiler Warning***************************************************
2022-06-14 02:52:41.061301: W itex/core/profiler/gpu_profiler.cc:114] Itex profiler not enabled, if you want to enable it, please set environment as :
export ZE_ENABLE_TRACING_LAYER=1
export UseCyclesPerSecondTimer=1
export ENABLE_TF_PROFILER=1
2022-06-14 02:52:41.061306: W itex/core/profiler/gpu_profiler.cc:118] ******************************************************************************************************
2022-06-14 02:52:41.063685: I itex/core/devices/gpu/dpcpp_runtime.cc:100] Selected platform: Intel(R) Level-Zero.
2022-06-14 02:52:41.063851: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-06-14 02:52:41.063865: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (DUT3046-ATSP): /proc/driver/nvidia/version does not exist
```

## Setting the dtype policy

To use mixed precision in Keras, you need to create a [`tf.keras.mixed_precision.Policy`](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/Policy), typically referred to as a *dtype policy*. Dtype policies specify how the dtypes layers will run in. In this guide, you will construct a policy from the string `'mixed_float16'` and set it as the global policy. This will cause subsequently created layers to use mixed precision with a mix of float16 and float32.

```python
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

```
mixed_precision.set_global_policy(policy)WARNING:tensorflow:Mixed precision compatibility check (mixed_float16): WARNING
The dtype policy mixed_float16 may run slowly because this machine does not have a GPU. Only Nvidia GPUs with compute capability of at least 7.0 run quickly with mixed_float16.
If you will use compatible GPU(s) not attached to this host, e.g. by running a multi-worker model, you can ignore this warning. This message will only be logged once
```

For short, you can directly pass a string to `set_global_policy`, which is typically done in practice.

```python
# Equivalent to the two lines above
mixed_precision.set_global_policy('mixed_float16')
```

The policy specifies two important aspects of a layer: the dtype the layer's computations are done in, and the dtype of a layer's variables. Above, you created a `mixed_float16` policy (i.e., a [`mixed_precision.Policy`](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/Policy) created by passing the string `'mixed_float16'` to its constructor). With this policy, layers use float16 computations and float32 variables. Computations are done in float16 for performance, but variables must be kept in float32 for numeric stability. You can directly query these properties of the policy.

```python
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

Compute dtype: float16
Variable dtype: float32
```

As mentioned before,  The policy will run on Intel GPUs and CPUs, and also could improve performance.

## Building the model

Next, let's start to build a simple model. Very small toy models typically do not benefit from mixed precision, because overhead from the TensorFlow runtime typically dominates the execution time, making any performance improvement on the Intel GPU negligible. Therefore, let's build two large `Dense` layers with 4096 units each when a GPU is used.

```python
inputs = keras.Input(shape=(784,), name='digits')
if tf.config.list_physical_devices('XPU'):
  print('The model will run with 4096 units on a Intel GPU')
  num_units = 4096
else:
  # Use fewer units on CPUs so the model finishes in a reasonable amount of time
  print('The model will run with 64 units on a CPU')
  num_units = 64
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)
```

```
The model will run with 4096 units on a Intel GPU
```

Each layer has a policy and uses the global policy by default. Each of the `Dense` layers therefore have the `mixed_float16` policy because you set the global policy to `mixed_float16` previously. This will cause the dense layers to do float16 computations and have float32 variables. They cast their inputs to float16 in order to do float16 computations, which causes their outputs to be float16 as a result. Their variables are float32 and will be cast to float16 when the layers are called to avoid errors from dtype mismatches.

```python
print(dense1.dtype_policy)
print('x.dtype: %s' % x.dtype.name)
# 'kernel' is dense1's variable
print('dense1.kernel.dtype: %s' % dense1.kernel.dtype.name)
```

```
<Policy "mixed_float16">
x.dtype: float16
dense1.kernel.dtype: float32
```

Next, create the output predictions. Normally, you can create the output predictions as follows, but this is not always numerically stable with float16.

```python
# INCORRECT: softmax and model output will be float16, when it should be float32
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)
```

```
Outputs dtype: float16
```

A softmax activation at the end of the model should be float32. Because the dtype policy is `mixed_float16`, the softmax activation would normally have a float16 compute dtype and output float16 tensors.

This can be fixed by separating the Dense and softmax layers, and by passing `dtype='float32'` to the softmax layer:

```python
# CORRECT: softmax and model output are float32
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)
```

```
Outputs dtype: float32 
```

Passing `dtype='float32'` to the softmax layer constructor overrides the layer's dtype policy to be the `float32` policy, which does computations and keeps variables in float32. You could also have passed `dtype=mixed_precision.Policy('float32')`; layers always convert the dtype argument to a policy. Because the `Activation` layer has no variables, the policy's variable dtype is ignored, but the policy's compute dtype of float32 causes softmax and the model output to be float32.

Adding a float16 softmax in the middle of a model is fine, but a softmax at the end of the model should be in float32. The reason is that if the intermediate tensor flowing from the softmax to the loss is float16 or bfloat16, numeric issues may occur.

You can override the dtype of any layer to be float32 by passing `dtype='float32'` if you think it will not be numerically stable with float16 computations. But typically, this is only necessary on the last layer of the model, as most layers have sufficient precision with `mixed_float16` and `mixed_bfloat16`.

Even if the model does not end in a softmax, the outputs should still be float32. While unnecessary for this specific model, the model outputs can be cast to float32 with the following:

```python
# The linear activation is an identity function. So this simply casts 'outputs'
# to float32. In this particular case, 'outputs' is already float32 so this is a
# no-op.
outputs = layers.Activation('linear', dtype='float32')(outputs)
```

Next, finish and compile the model, and generate input data:

```python
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
```

```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 12s 1us/step
```

This example casts the input data from int8 to float32. You don't cast to float16 since the division by 255 is on the CPU, which runs float16 operations slower than float32 operations. In this case, the performance difference in negligible, but in general you should run input processing math in float32 if it runs on the CPU. The first layer of the model will cast the inputs to float16, as each layer casts floating-point inputs to its compute dtype.

The initial weights of the model are retrieved. This will allow training from scratch again by loading the weights.

```python
initial_weights = model.get_weights()
```

## Training the model with Model.fit

Next, train the model:

```python
history = model.fit(x_train, y_train,
                    batch_size=8192,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
```

```
Epoch 1/5
6/6 [==============================] - 111s 2s/step - loss: 5.6240 - accuracy: 0.3359 - val_loss: 0.9755 - val_accuracy: 0.7494
Epoch 2/5
6/6 [==============================] - 0s 83ms/step - loss: 0.7987 - accuracy: 0.7520 - val_loss: 0.3455 - val_accuracy: 0.8972
Epoch 3/5
6/6 [==============================] - 0s 81ms/step - loss: 0.3670 - accuracy: 0.8819 - val_loss: 0.3753 - val_accuracy: 0.8751
Epoch 4/5
6/6 [==============================] - 1s 85ms/step - loss: 0.3555 - accuracy: 0.8863 - val_loss: 0.2155 - val_accuracy: 0.9377
Epoch 5/5
6/6 [==============================] - 0s 84ms/step - loss: 0.1986 - accuracy: 0.9410 - val_loss: 0.4498 - val_accuracy: 0.8534
```

Notice the model prints the time per step in the logs: for example, "84ms/step". The first epoch may be slower as TensorFlow spends some time optimizing the model, but afterwards the time per step should stabilize.

If you are running this guide in Colab, you can compare the performance of mixed precision with float32. To do so, change the policy from `mixed_float16` to `float32` in the "Setting the dtype policy" section, then rerun all the cells up to this point. On GPUs, you should see the time per step significantly increase, indicating mixed precision sped up the model. Make sure to change the policy back to `mixed_float16` and rerun the cells before continuing with the guide.

For many real-world models, mixed precision also allows you to double the batch size without running out of memory, as float16 tensors take half the memory. This does not apply however to this toy model, as you can likely run the model in any dtype where each batch consists of the entire MNIST dataset of 60,000 images.

## Loss scaling

Loss scaling is a technique which [`tf.keras.Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) automatically performs with the `mixed_float16` policy to avoid numeric underflow. This section describes what loss scaling is and the next section describes how to use it with a custom training loop.

### Underflow and Overflow

The float16 data type has a narrow dynamic range compared to float32. This means values above 65504 will overflow to infinity and values below 6.0×10-8 will underflow to zero. float32 and bfloat16 have a much higher dynamic range so that overflow and underflow are not a problem.

For example:

```python
x = tf.constant(256, dtype='float16')
(x ** 2).numpy()  # Overflow
inf
x = tf.constant(1e-5, dtype='float16')
(x ** 2).numpy()  # Underflow
0.0
```

In practice, overflow with float16 rarely occurs. Additionally, underflow also rarely occurs during the forward pass. However, during the backward pass, gradients can underflow to zero. Loss scaling is a technique to prevent this underflow.

### Loss scaling overview

The basic concept of loss scaling is simple: multiply the loss by some large number, say 1024, and you get the *loss scale* value. This will cause the gradients to scale by 1024 as well, greatly reducing the chance of underflow. Once the final gradients are computed, divide them by 1024 to bring them back to their correct values.

The pseudocode for this process is:

```
loss_scale = 1024
loss = model(inputs)
loss *= loss_scale
# Assume `grads` are float32. You do not want to divide float16 gradients.
grads = compute_gradient(loss, model.trainable_variables)
grads /= loss_scale
```

Choosing a loss scale can be tricky. If the loss scale is too low, gradients may still underflow to zero. If too high, the gradients may overflow to infinity.

To solve this, TensorFlow dynamically determines the loss scale so you do not have to choose one manually. If you use [`tf.keras.Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit), loss scaling is done for you so you do not have to do any extra work. If you use a custom training loop, you must explicitly use the special  wrapper [`tf.keras.mixed_precision.LossScaleOptimizer`](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer) in order to use loss scaling. This is described in the next section.

## Training the model with a custom training loop

So far, you have trained a Keras model with mixed precision using [`tf.keras.Model.fit`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit). Next, you will use mixed precision with a custom training loop. If you do not already know what a custom training loop is, read the [Custom training guide](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough) first.

Running a custom training loop with mixed precision requires two changes over running it in float32:

1. Build the model with mixed precision (you already did this)
2. Explicitly use loss scaling if `mixed_float16` is used.

For step (2), you will use the [`tf.keras.mixed_precision.LossScaleOptimizer`](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer) class, which wraps an optimizer and applies loss scaling. By default, it dynamically determines the loss scale so you do not have to choose one. Construct a `LossScaleOptimizer` as follows.

```python
optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
```

You can choose an explicit loss scale or otherwise customize the loss scaling behavior, but it is highly recommended to keep the default loss scaling behavior, as it has been found to work well on all known models. See the [`tf.keras.mixed_precision.LossScaleOptimizer`](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/LossScaleOptimizer) documention if you want to customize the loss scaling behavior.

Next, define the loss object and the [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)s:

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(10000).batch(8192))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(8192)
```

Next, define the training step function. You will use two new methods from the loss scale optimizer to scale the loss and unscale the gradients:

- `get_scaled_loss(loss)`: Multiplies the loss by the loss scale
- `get_unscaled_gradients(gradients)`: Takes in a list of scaled gradients as inputs, and divides each one by the loss scale to unscale them

These functions must be used in order to prevent underflow in the gradients. `LossScaleOptimizer.apply_gradients` will then apply gradients if none of them have `Inf`s or `NaN`s. It will also update the loss scale, halving it if the gradients had `Inf`s or `NaN`s and potentially increasing it otherwise.

```python
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    predictions = model(x)
    loss = loss_object(y, predictions)
    scaled_loss = optimizer.get_scaled_loss(loss)
  scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
```

The `LossScaleOptimizer` will likely skip the first few steps at the start of training. The loss scale starts out high so that the optimal loss scale can quickly be determined. After a few steps, the loss scale will stabilize and very few steps will be skipped. This process happens automatically and does not affect training quality.

Now, define the test step:

```python
@tf.function
def test_step(x):
  return model(x, training=False)
```

Load the initial weights of the model, so you can retrain from scratch:

```python
model.set_weights(initial_weights)
```

Finally, run the custom training loop:

```python
for epoch in range(5):
  epoch_loss_avg = tf.keras.metrics.Mean()
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
  for x, y in train_dataset:
    loss = train_step(x, y)
    epoch_loss_avg(loss)
  for x, y in test_dataset:
    predictions = test_step(x)
    test_accuracy.update_state(y, predictions)
  print('Epoch {}: loss={}, test accuracy={}'.format(epoch, epoch_loss_avg.result(), test_accuracy.result()))
```

```
Epoch 0: loss=3.924008369445801, test accuracy=0.7239000201225281
Epoch 1: loss=0.5294489860534668, test accuracy=0.9168000221252441
Epoch 2: loss=0.3364005982875824, test accuracy=0.9381000399589539
Epoch 3: loss=0.25294047594070435, test accuracy=0.9486000537872314
Epoch 4: loss=0.26531240344047546, test accuracy=0.9536000490188599
```

