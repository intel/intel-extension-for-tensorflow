# Distributed training example with Intel® Optimization for Horovod*

## Dependency
- [TensorFlow](https://pypi.org/project/tensorflow/)
- [Intel® Extension for TensorFlow*](https://pypi.org/project/intel-extension-for-tensorflow/)
- [Intel® Optimization for Horovod*](https://pypi.org/project/intel-optimization-for-horovod/)
- others show as below 
```
pip install gin gin-config tensorflow-addons tensorflow-model-optimization tensorflow-datasets
```

## Model examples preparation

### Model Repo
```
git clone https://github.com/horovod/horovod.git # top commit: 0b19c
cd horovod/examples/tensorflow2
git apply tensorflow2_keras_mnist.patch
```
**Notes**:  
Refer to [tensorflow2_keras_mnist.py](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_mnist.py) for other changes about how to enable horovod.

## Execution
To run on a machine with 4 XPUs:  
**Notes**:  
Check log looking for "### is_mpi" and "### gpus" to get how many XPUs are in your platform. Our example is on a machine with 2 Intel® Data Center GPU Max Series. The 2 tiles in each GPU are taken as 2 independent XPU devices, for a total of 4 XPU devices.
```
horovodrun -np 4 -H localhost:4 python ./tensorflow2_keras_mnist.py
```

## Output
```
...
[0] ### len(sys.argv)  1
[0] ### is_mpi  4
[0] ### gpus  [PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU'), PhysicalDevice(name='/physical_device:XPU:1', device_type='XPU'), PhysicalDevice(name='/physical_device:XPU:2', device_type='XPU'), PhysicalDevice(name='/physical_device:XPU:3', device_type='XPU')]
...
[3] 2022-12-05 10:45:55.830045: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:XPU:0 with 0 MB memory) -> physical PluggableDevice (device: 3, name: XPU, pci bus id: <undefined>)
[2] 2022-12-05 10:45:55.830610: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:XPU:0 with 0 MB memory) -> physical PluggableDevice (device: 2, name: XPU, pci bus id: <undefined>)
[1] 2022-12-05 10:45:55.830696: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:XPU:0 with 0 MB memory) -> physical PluggableDevice (device: 1, name: XPU, pci bus id: <undefined>)
[0] 2022-12-05 10:45:55.831773: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:XPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: XPU, pci bus id: <undefined>)
...
[3] 2022-12-05 10:45:58.971265: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[2] 2022-12-05 10:45:58.982479: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[1] 2022-12-05 10:45:59.003327: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[0] 2022-12-05 10:45:59.070484: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
...
0] Epoch 1/24
1/125 [..............................] - ETA: 20:53 - loss: 2.3020 - accuracy: 0.0859
[0] Epoch 2/24
125/125 [==============================] - xxms/step - loss: 0.1776 - accuracy: 0.9456 - lr: 0.0030
...
```
