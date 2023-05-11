load(":build_defs.bzl", "pybind_extension", "pybind_library")

pybind_extension(
    name = "_pywrap_itex",
    srcs = [
        "itex_wrapper.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core:protos_all_cc",
        "//itex/core/devices:device_backend_util_hdr",
        "//itex/core/graph:config_util_hdr",
        "//itex/core/kernels:libitex_common",
        "@com_google_absl//absl/strings",
        "@local_config_python//:python_headers",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "device",
    srcs = ["device.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_itex.so",
        "//itex/core:protos_all_py",
    ],
)

py_binary(
    name = "gen_itex_version",
    srcs = ["gen_itex_version.py"],
    visibility = ["//visibility:public"],
)

# This targe should only be used for API generation.
py_library(
    name = "modules_with_exports",
    srcs = ["modules_with_exports.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)
