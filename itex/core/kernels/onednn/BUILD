package(
    licenses = ["notice"],  # Apache 2.0
)

load("//itex:itex.bzl", "itex_xpu_library")

itex_xpu_library(
    name = "onednn_kernels",
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/kernels/onednn/block:block_kernels",
        # TODO(itex): Enable native kernel in future.
        # "//itex/core/kernels/onednn/native:native_kernels",
    ],
    alwayslink = True,
)
