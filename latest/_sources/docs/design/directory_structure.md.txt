# Directory Tree Structure


The directory tree structure of Intel® Extension for TensorFlow*:

```
intel-extension-for-tensorflow/
├── docker
├── docs
│   ├── community
│   ├── design
│   │   └── optimization
│   ├── docs_build
│   ├── guide
│   └── install
├── examples
├── itex
│   ├── core
│   │   ├── devices
│   │   ├── graph
│   │   ├── kernels
│   │   ├── ops
│   │   ├── profiler
│   │   └── utils
│   ├── python
│   │   └── ops
│   └── tools
├── test
├── third_party
└── third-party-programs
```



The key directory is `itex/`, which contains core infrastructure code:

| **Parent directory** | **Sub-directory** | **Description**                                              |
| -------------------- | ----------------- | ------------------------------------------------------------ |
| `core/`              | `devices/`        | CPU/GPU device infrastructure.                               |
|                      | `graph/`          | Graph fusion, OneDNN graph and oneDNN memory format propagation infrastructure. |
|                      | `kernels/`        | CPU/GPU kernels.                                             |
|                      | `ops/`            | CPU/GPU ops.                                                 |
|                      | `profiler/`       | Profiling tool.                                              |
|                      | `utils/`          | Miscellaneous utilities.                                     |
| `python/`            | `ops/`            | Custom layers or ops in Python API.                          |
| `tools/`             |                   | Tools for building the repository.                           |

