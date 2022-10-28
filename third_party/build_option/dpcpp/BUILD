licenses(["notice"])

config_setting(
    name = "build_with_dpcpp",
    define_values = {
        "build_with_dpcpp": "true",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dpcpp_header",
    srcs = ["runtime/dpcpp_runtime.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "eigen_dpcpp_header",
    srcs = ["runtime/eigen_dpcpp_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [":dpcpp_header"],
)
