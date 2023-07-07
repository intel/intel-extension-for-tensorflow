load("//itex:itex.bzl", "cc_library")

cc_library(
    name = "sparse_tensor",
    srcs = ["sparse_tensor.cc"],
    hdrs = [
        "dim_comparator.h",
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//itex/core/utils:common_utils",
        "//itex/core/utils:errors",
        "//itex/core/utils:status",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
    ],
)
