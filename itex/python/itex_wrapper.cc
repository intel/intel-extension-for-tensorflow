/* Copyright (c) 2021-2022 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>

#include "Python.h"
#include "itex/core/devices/device_backend_util.h"
#include "itex/core/graph/config_util.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace itex {
static py::object ITEX_GetBackend() {
  std::string backend;
  itex_backend_to_string(itex_get_backend(), &backend);
  PyObject* result = PyBytes_FromStringAndSize(backend.data(), backend.size());
  if (PyErr_Occurred() || result == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(result);
}

static py::bytes ITEX_GetConfig() {
  std::string config_str;
  ConfigProto config_proto = itex_get_config();
  config_proto.SerializeToString(&config_str);
  return py::bytes(config_str);
}

PYBIND11_MODULE(_pywrap_itex, m) {
  m.doc() = "pybind11 front-end api for Intel ® Extension for TensorFlow*";
  m.def("ITEX_SetBackend",
        [](const char* backend) { itex_set_backend(backend); });
  m.def("ITEX_GetBackend", &itex::ITEX_GetBackend);

  m.def("ITEX_SetConfig", [](py::bytes proto) {
    char* c_string;
    Py_ssize_t py_size;
    if (PyBytes_AsStringAndSize(proto.ptr(), &c_string, &py_size) == -1) {
      throw py::error_already_set();
    }
    itex::ConfigProto config;
    config.ParseFromArray(c_string, py_size);

    itex_set_config(config);
  });
  m.def("ITEX_GetConfig", &itex::ITEX_GetConfig);
}

}  // namespace itex
