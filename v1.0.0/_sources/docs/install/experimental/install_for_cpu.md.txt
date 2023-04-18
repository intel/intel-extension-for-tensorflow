# *Experimental:* Intel CPU Software Installation 


## *Experimental Release*

Intel® Extension for TensorFlow* provides *experimental support* for 2nd Generation Intel® Xeon® Scalable Processors and newer. 

Stock [TensorFlow](https://pypi.org/project/tensorflow/) and [Intel® Optimization for TensorFlow*](https://pypi.org/project/intel-tensorflow/) are recommended if product quality is required.

## Hardware Requirements

Verified Hardware Platforms:
- Cascade Lake
- Cooper Lake
- Ice Lake
- Sapphire Rapids
 
## Software Requirements

- Ubuntu 20.04 (64-bit), Ubuntu 22.04 (64-bit) or CentOS Linux 8 (64-bit), and Sapphire Rapids requires Ubuntu 22.04 or CentOS Linux 8.
- Python 3.7-3.10
- pip 19.0 or later (requires manylinux2014 support)
  
## Install via Docker container

#### Build Docker container from Dockerfile

Run the following [Dockerfile build procedure](./../../../docker/README.md) to build the pip based deployment container.

#### Get docker container from dockerhub

Pre-built docker images are available at [DockerHub](https://hub.docker.com/r/intel/intel-extension-for-tensorflow/tags).
Please run the following command to pull the CPU Docker container image to your local machine.

```
$ docker pull intel/intel-extension-for-tensorflow:cpu
$ docker run -it -p 8888:8888 intel/intel-extension-for-tensorflow:cpu
```
Then go to your browser on http://localhost:8888/

## Install via PyPI wheel in bare metal

#### Install TensorFlow

The Python development and virtual environment setup recommendation by TensorFlow to isolate package installation from the system.

The Intel® Extension for TensorFlow* requires stock TensorFlow, and the version should be == 2.10.0. 


##### Virtual environment install 

You can follow the instructions in [stock tensorflow install](https://www.tensorflow.org/install/pip#step-by-step_instructions) to activate the virtual environment.

On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels.
```bash
(tf)$ pip install --upgrade pip
```

To install in virtual environment, you can run 
```bash
(tf)$ pip install tensorflow==2.10.0
```

##### System environment install 
If want to system install in $HOME, append `--user` to the commands.
```bash
$ pip3 install --user tensorflow==2.10.0
``` 
And the following system environment install for Intel® Extension for TensorFlow* will use the same practice. 

#### Install Intel® Extension for TensorFlow*

To install a CPU-only version in virtual environment, you can run

```bash
(tf)$ pip install intel-extension-for-tensorflow[cpu]==1.0.0
```

##### Verify the Installation 
```
python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
```
