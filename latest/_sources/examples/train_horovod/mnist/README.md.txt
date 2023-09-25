# Distributed Training Example with Intel® Optimization for Horovod* on Intel® GPU

## Dependency
- [TensorFlow](https://pypi.org/project/tensorflow/)
- [Intel® Extension for TensorFlow*](https://pypi.org/project/intel-extension-for-tensorflow/)
- [Intel® Optimization for Horovod*](https://pypi.org/project/intel-optimization-for-horovod/)

## Setup Running Environment
### Create Virtual Environment
```
python -m venv env_itex
source source env_itex/bin/activate
```
### Install
```
pip install intel-extension-for-tensorflow[gpu]
pip install intel-optimization-for-horovod
pip install gin gin-config tensorflow-addons tensorflow-model-optimization tensorflow-datasets
```
## Prepare Example Code
### Clone Horovod Repo
```
git clone https://github.com/horovod/horovod.git
cd horovod/examples/tensorflow2
```
### Download Patch
```
wget https://github.com/intel/intel-extension-for-tensorflow/raw/main/examples/train_horovod/mnist/tensorflow2_keras_mnist.patch
```
### Apply Patch for Intel GPU
```
git apply tensorflow2_keras_mnist.patch
```
**Notes**:
Please refer to [tensorflow2_keras_mnist.py](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_mnist.py) for other changes about how to enable horovod.
## Execution
### Enable oneAPI
Run:
```
source /opt/intel/oneapi/setvars.sh
```
Note: to install oneAPI base toolkit, refer to [Intel GPU Software Installation](/docs/install/install_for_xpu.html#install-oneapi-base-toolkit-packages)

### Check Device Count (Optional)
Run:
```
mpirun -np 1 -prepend-rank -ppn 1 python tensorflow2_keras_mnist.py
```

Check how many devices (XPUs) in local machine according output of above command, like:
```
...
XPU count is 2
XPU: PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')
XPU: PhysicalDevice(name='/physical_device:XPU:1', device_type='XPU')
```
The XPUs are **2** according to above log.

**Notes**:
In some Intel GPU (like Intel® Data Center GPU Max Series), there are more than 1 tile per GPU card. A tile is considered as 1 XPU device. So there are total 4 XPUs if there are 2 GPU cards and 2 tiles per GPU card.

### Running Command
For 2 XPUs:
```
mpirun -np 2 -prepend-rank -ppn 2 python ./tensorflow2_keras_mnist.py
```
For 4 XPUs:
```
mpirun -np 4 -prepend-rank -ppn 4 python ./tensorflow2_keras_mnist.py
```
## Output
```
...
[0] Horovod size 2
[0] XPU count is 2
[0] XPU: PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')
[0] XPU: PhysicalDevice(name='/physical_device:XPU:1', device_type='XPU')
...
[1] 2023-05-18 13:54:12.006950: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[0] 2023-05-18 13:54:12.163161: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[1] 2023-05-18 13:54:13.940695: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[0] 2023-05-18 13:54:14.107809: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[0] 2023-05-18 13:54:14.163517: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
[1] 2023-05-18 13:54:14.163517: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
...
0] Epoch 1/24
250/250 [==============================] - Xs YYms/step - loss: X.XXXX - accuracy: Y.YYYY - lr: Z.ZZZZ
[0] Epoch 2/24
250/250 [==============================] - Xs YYms/step - loss: X.XXXX - accuracy: Y.YYYY - lr: Z.ZZZZ

...
```
