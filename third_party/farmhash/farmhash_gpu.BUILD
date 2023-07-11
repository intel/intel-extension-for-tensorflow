licenses(["notice"])  # MIT

exports_files(["COPYING"])

load("@intel_extension_for_tensorflow//itex:itex.bzl", "cc_library")

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)

cc_library(
    name = "farmhash_gpu",
    hdrs = ["src/farmhash_gpu.h"],
    include_prefix = "third_party/farmhash_gpu",
    visibility = ["//visibility:public"],
)
